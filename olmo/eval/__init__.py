from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, Metric

from ..config import EvaluatorConfig, EvaluatorType, TrainConfig
from ..exceptions import OLMoConfigurationError
from ..tokenizer import Tokenizer
from ..torch_util import get_global_rank, get_world_size
from ..data.util import DistributedEvalSampler
from .downstream import ICLMetric, BeamSearchICLMetric, DecomposedICLMetric, label_to_task_map, TGPerplexitySentenceLevelMetric, TGPerplexityDocumentLevelMetric, TerminalDocumentPerplexityMetric, SyntacticGeneralizationMetric, BLiMPMetric, RougeMetric
from .evaluator import Evaluator
from olmo.data import get_TG_generate_bias_func

__all__ = [
    "Evaluator",
    "ICLMetric",
    "BeamSearchICLMetric",
    "DecomposedICLMetric",
    "label_to_task_map",
    "build_downstream_evaluator",
    "build_evaluator",
    "build_evaluators",
    "resolve_structure_mode",
]

beam_search_tasks = {
    "syntactic_generalization",
    "xsum"
}

STRUCTURE_MODES = {"auto", "terminal", "gold", "beam"}


def resolve_structure_mode(
    eval_cfg: EvaluatorConfig,
    transformer_grammar_type: str,
) -> str:
    """Resolve one evaluator's syntax protocol, including legacy configs.

    ``structure_mode`` is the canonical switch. The older ``beam_search`` flag
    remains supported only when ``structure_mode`` is left at ``auto``; making
    both explicit with conflicting values is almost certainly a configuration
    error and must not silently select a branch.
    """
    mode = getattr(eval_cfg, "structure_mode", "auto")
    if mode not in STRUCTURE_MODES:
        expected = ", ".join(sorted(STRUCTURE_MODES))
        raise OLMoConfigurationError(
            f"Unknown evaluator structure_mode={mode!r}; expected one of {expected}"
        )

    legacy_beam = bool(getattr(eval_cfg, "beam_search", False))
    if legacy_beam:
        if mode not in {"auto", "beam"}:
            raise OLMoConfigurationError(
                "Evaluator sets beam_search=true but "
                f"structure_mode={mode!r}; remove beam_search or use "
                "structure_mode='beam'"
            )
        mode = "beam"

    # Preserve the historical Pushdown SG default while allowing an explicit
    # terminal run for the requested protocol comparison.
    if (
        mode == "auto"
        and eval_cfg.label == "syntactic_generalization"
        and transformer_grammar_type == "pushdown"
    ):
        mode = "beam"

    if mode == "gold" and eval_cfg.label != "BLiMP":
        raise OLMoConfigurationError(
            "structure_mode='gold' is currently supported only for BLiMP"
        )
    if mode == "beam" and eval_cfg.label not in beam_search_tasks | {"BLiMP"}:
        raise OLMoConfigurationError(
            f"structure_mode='beam' is not supported for evaluator {eval_cfg.label!r}"
        )
    return mode

def build_downstream_evaluator(
    train_config: TrainConfig,
    eval_cfg: EvaluatorConfig,
    tokenizer: Tokenizer,
    device: torch.device,
    is_unit_test=False,
) -> Evaluator:
    task_kwargs = {}
    task_spec = label_to_task_map[eval_cfg.label]
    if isinstance(task_spec, tuple):
        task_class, default_task_kwargs = task_spec
        # Registry defaults are shared process-wide. Evaluator-specific options
        # (force_terminal, pushdown_gold, pair_per_task, etc.) must never mutate
        # that shared dictionary or leak into the next evaluator with the same
        # label.
        task_kwargs = dict(default_task_kwargs)
        if eval_cfg.type in (EvaluatorType.tg_doc, EvaluatorType.terminal_doc) or eval_cfg.label=="BLiMP":
            task_kwargs["device_eval_batch_size"] = (
                1
                if eval_cfg.type == EvaluatorType.terminal_doc
                else eval_cfg.device_eval_batch_size or train_config.device_eval_batch_size
            )
    else:
        task_class = task_spec
    if (
        eval_cfg.label == "syntactic_generalization"
        and getattr(eval_cfg, "sg_dataset_path", None) is not None
    ):
        task_kwargs["dataset_path"] = eval_cfg.sg_dataset_path
    task_kwargs["model_ctx_len"] = train_config.model.max_sequence_length
    task_kwargs["vocab_path"] = train_config.tokenizer.vocabulary
    task_kwargs["generate_TG_attention_bias"] = get_TG_generate_bias_func(train_config)
    task_kwargs["transformer_grammar_type"] = train_config.model.transformer_grammar_type
    task_kwargs["pause_token_id"] = train_config.model.pause_token_id
    if eval_cfg.samples_per_sent is not None:
        task_kwargs["samples_per_sent"] = eval_cfg.samples_per_sent
    if eval_cfg.tree_eval_type is not None:
        task_kwargs["tree_eval_type"] = eval_cfg.tree_eval_type
    structure_mode = resolve_structure_mode(
        eval_cfg, train_config.model.transformer_grammar_type
    )
    if eval_cfg.label == "BLiMP":
        # Terminal and beam protocols both consume one terminal sequence per
        # sentence. Beam search supplies its own latent structure.
        if structure_mode in {"terminal", "beam"}:
            task_kwargs["force_terminal"] = True
        # Pushdown gold300 consumes the supplied tree_300 parses, converted to
        # terminal tokens and terminal-coordinate spans by the dataset.
        if structure_mode == "gold":
            if train_config.model.transformer_grammar_type == "pushdown":
                task_kwargs["pushdown_gold"] = True
                task_kwargs["parse_binarize_direction"] = (
                    train_config.model.parse_binarize_direction
                )
            elif train_config.model.transformer_grammar_type == "treereg":
                # TreeReg trees affect only the training loss. Avoid evaluating
                # 300 identical forward distributions per sentence.
                task_kwargs["force_terminal"] = True
    # BLiMP subset: reduce pairs/task (and the compute() denominator) for a
    # meaningful partial-run accuracy. None => 1000 (full BLiMP).
    if eval_cfg.label == "BLiMP" and getattr(eval_cfg, "pair_per_task", None) is not None:
        task_kwargs["pair_per_task"] = eval_cfg.pair_per_task
    if train_config.finetune_task is not None:
        task_kwargs["shots_num"] = 0
    ds_eval_dataset = task_class(tokenizer=tokenizer, **task_kwargs)  # type: ignore
    data_config = eval_cfg.data
    if is_unit_test:
        ds_eval_sampler = None
    else:
        # Use a partitioning sampler that evaluates every sample exactly once
        # across ranks (no padding duplicates, no tail truncation). The previous
        # setup used DistributedSampler (drop_last=False → padded duplicates
        # that corrupted sum/len metrics) for some evaluators and
        # SequentialDistributedSampler (tail truncation → skipped cases) for
        # others. DistributedEvalSampler fixes both.
        #
        # tg_doc / tg_sent / BLiMP need contiguous blocks: tg_doc accumulates a
        # KV cache within a document (samples of one doc must arrive on one
        # rank, in order); tg_sent/BLiMP assume sequential sent_id arrival and
        # index by position. The other evaluators (SG, Rouge, ICL,
        # beam_search_icl) have no order dependence and use strided partitioning.
        contiguous = eval_cfg.type in [
            EvaluatorType.tg_doc,
            EvaluatorType.terminal_doc,
            EvaluatorType.tg_sent,
        ] or eval_cfg.label == "BLiMP"
        # tg_doc additionally needs whole-document partitioning: its KV cache
        # accumulates across every sentence of a document, so a document must
        # not be split across ranks (count-based contiguous splitting would cut
        # mid-sentence/mid-document). Pass group_starts = per-document tree-index
        # boundaries so the sampler partitions DOCUMENTS, not samples. Each
        # document is an integer number of sentences (SENT_SIZE trees), so this
        # also keeps the per-sentence 300-sync in TG_doc_eval_step aligned.
        group_starts = None
        if eval_cfg.type in (EvaluatorType.tg_doc, EvaluatorType.terminal_doc):
            ds = ds_eval_dataset
            sent_size = getattr(ds, "SENT_SIZE", None) or getattr(ds, "samples_per_sent", 300)
            # The dataset keeps sentence-level cumulative document ends before
            # prep_examples mutates doc_index. Convert those exact boundaries to
            # flat candidate-record offsets.
            document_ends = getattr(ds, "document_ends", None)
            if document_ends is not None:
                starts = np.concatenate(
                    (np.zeros(1, dtype=np.int64), np.asarray(document_ends))
                ) * int(sent_size)
                group_starts = torch.from_numpy(starts)
        elif eval_cfg.label == "BLiMP" and ds_eval_dataset.SENT_SIZE > 1:
            # Gold-K BLiMP batches and BLiMPMetric rows are sentence groups of
            # exactly SENT_SIZE parses. A plain contiguous N/world_size split
            # can cut a group (e.g. gold300 on 3 ranks), causing a batch to span
            # two metric rows. Partition whole sentences instead.
            sent_size = int(ds_eval_dataset.SENT_SIZE)
            if len(ds_eval_dataset) % sent_size != 0:
                raise OLMoConfigurationError(
                    f"BLiMP dataset length {len(ds_eval_dataset)} is not divisible "
                    f"by SENT_SIZE={sent_size}"
                )
            group_starts = torch.arange(
                0, len(ds_eval_dataset) + 1, sent_size, dtype=torch.long
            )
        ds_eval_sampler = DistributedEvalSampler(
            ds_eval_dataset,
            num_replicas=get_world_size(),
            rank=get_global_rank(),
            contiguous=contiguous,
            group_starts=group_starts,
        )
    eval_batch_size = eval_cfg.device_eval_batch_size or train_config.device_eval_batch_size
    if eval_cfg.type == EvaluatorType.terminal_doc:
        # One terminal path per sentence. Sequential batch size 1 lets the
        # shared document scorer commit that path to its KV cache immediately.
        # Pause models receive a document-phase-aware expanded sequence and a
        # label mask that excludes the inserted pause targets.
        eval_batch_size = 1
    if eval_cfg.label in beam_search_tasks or eval_cfg.type == EvaluatorType.beam_search_icl \
        or (eval_cfg.label == "BLiMP" and (
            structure_mode == "beam"
        )):
        eval_batch_size = 1
    
    loader_num_workers = (
        0 if eval_cfg.type == EvaluatorType.terminal_doc else data_config.num_workers
    )
    ds_eval_dataloader = DataLoader(
        ds_eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=ds_eval_dataset.collate_fn,
        num_workers=loader_num_workers,
        sampler=ds_eval_sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=(
            None if loader_num_workers == 0 else data_config.prefetch_factor
        ),
        persistent_workers=(
            False if loader_num_workers == 0 else data_config.persistent_workers
        ),
        timeout=data_config.timeout,
    )
    if eval_cfg.type == EvaluatorType.tg_sent:
        metric = TGPerplexitySentenceLevelMetric(
            vocab_path=train_config.tokenizer.vocabulary,
            metric_type=ds_eval_dataset.metric_type, 
            term_length=ds_eval_dataset.get_term_length(),
            device_eval_batch_size = eval_batch_size
        )
    elif eval_cfg.type == EvaluatorType.terminal_doc:
        metric = TerminalDocumentPerplexityMetric(
            term_length=ds_eval_dataset.get_term_length(),
            dataset_length=len(ds_eval_dataset),
        )
    elif eval_cfg.type == EvaluatorType.tg_doc:
        metric = TGPerplexityDocumentLevelMetric(
            vocab_path=train_config.tokenizer.vocabulary,
            metric_type=ds_eval_dataset.metric_type, 
            term_length=ds_eval_dataset.get_term_length(),
            device_eval_batch_size = eval_batch_size,
            dataset_length=len(ds_eval_dataset),
            samples_per_sent=ds_eval_dataset.SENT_SIZE,
        )
    elif eval_cfg.label == "syntactic_generalization":
        metric = SyntacticGeneralizationMetric(metric_type=ds_eval_dataset.metric_type,
                                               tree_eval_type=getattr(ds_eval_dataset, "tree_eval_type", "default"))
    elif eval_cfg.label == "BLiMP":
        # Opt-in beam-tree dump for offline comparison vs blimp_tree_300.
        # Off by default so normal eval runs are unaffected.
        import os as _bos
        beam_dump_path = None
        if _bos.environ.get("OLMO_BEAM_DUMP") == "1" and train_config.save_folder:
            beam_dump_path = _bos.path.join(
                train_config.save_folder, "beam_trees_BLiMP.jsonl"
            )
        metric = BLiMPMetric(vocab_path=train_config.tokenizer.vocabulary,
                             metric_type=ds_eval_dataset.metric_type,
                             dataset_name=train_config.model.transformer_grammar_type,
                             device_eval_batch_size = eval_batch_size,
                             dataset_length=len(ds_eval_dataset),
                             samples_per_sent=ds_eval_dataset.SENT_SIZE,
                             pair_per_task=ds_eval_dataset.pair_per_task,
                             tree_eval_type=ds_eval_dataset.tree_eval_type,
                             save_beam_trees_path=beam_dump_path)
    elif eval_cfg.type == EvaluatorType.rouge:
        metric = RougeMetric(tokenizer=tokenizer)
    elif eval_cfg.type == EvaluatorType.beam_search_icl:
        metric = BeamSearchICLMetric(
            metric_type=ds_eval_dataset.metric_type,
            doc_group=ds_eval_dataset.doc_group,
        )
    elif eval_cfg.label.endswith("_decomp"):
        save_path = None
        if train_config.save_folder:
            import os as _os
            save_path = _os.path.join(
                train_config.save_folder,
                f"per_example_{eval_cfg.label}.json"
            )
        metric = DecomposedICLMetric(
            metric_type=ds_eval_dataset.metric_type,
            vocab_path=train_config.tokenizer.vocabulary,
            tree_eval_type=ds_eval_dataset.tree_eval_type,
            doc_group=ds_eval_dataset.doc_group,
            tokenizer=tokenizer,
            save_per_example_path=save_path,
        )
    else:
        # Opt-in per-instance prediction dump for post-hoc bootstrap significance
        # testing. Set OLMO_BOOTSTRAP_DUMP=1 to write per_example_<label>.json
        # into save_folder. Off by default so normal eval runs are unaffected.
        import os as _bos
        save_path = None
        if _bos.environ.get("OLMO_BOOTSTRAP_DUMP") == "1" and train_config.save_folder:
            save_path = _bos.path.join(
                train_config.save_folder, f"per_example_{eval_cfg.label}.json"
            )
        metric = ICLMetric(metric_type=ds_eval_dataset.metric_type,
                           vocab_path=train_config.tokenizer.vocabulary,
                            tree_eval_type=ds_eval_dataset.tree_eval_type,
                            doc_group=ds_eval_dataset.doc_group,
                            save_per_example_path=save_path)

    if eval_cfg.type in (EvaluatorType.tg_doc, EvaluatorType.terminal_doc) or eval_cfg.label == "BLiMP":
        assert (
            ds_eval_dataset.SENT_SIZE % eval_batch_size == 0
            or ds_eval_dataset.TASK_SIZE % eval_batch_size == 0
        ), (
            f"SENT_SIZE={ds_eval_dataset.SENT_SIZE} and "
            f"TASK_SIZE={ds_eval_dataset.TASK_SIZE} not divisible by "
            f"eval_batch_size={eval_batch_size}"
        )

    evaluator = Evaluator(
        label=eval_cfg.label,
        type=eval_cfg.type,
        eval_loader=ds_eval_dataloader,
        eval_metric=metric.to(device),
        subset_num_batches=eval_cfg.subset_num_batches,
    )
    return evaluator


def build_evaluator(
    train_config: TrainConfig, eval_config: EvaluatorConfig, tokenizer: Tokenizer, device: torch.device
) -> Evaluator:
    from ..data import build_eval_dataloader

    if eval_config.type in [EvaluatorType.tg_doc, EvaluatorType.terminal_doc, EvaluatorType.tg_sent, EvaluatorType.downstream, EvaluatorType.rouge, EvaluatorType.beam_search_icl]:
        # Downstream evaluation.
        return build_downstream_evaluator(train_config, eval_config, tokenizer, device)
    elif eval_config.type == EvaluatorType.lm:
        # Language modeling evaluation.
        eval_loader = build_eval_dataloader(
            train_config,
            eval_config.data,
            eval_config.device_eval_batch_size or train_config.device_eval_batch_size,
        )

        def make_metric():
            return MeanMetric(nan_strategy="error").to(device)

        eval_metric: Union[Metric, Dict[str, Metric]]
        if eval_config.data.paths:
            eval_metric = make_metric()
        elif eval_config.data.datasets:
            eval_metric = {label: make_metric() for label in eval_config.data.datasets.keys()}
        else:
            raise OLMoConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")

        return Evaluator(
            label=eval_config.label,
            type=eval_config.type,
            eval_loader=eval_loader,
            eval_metric=eval_metric,
            subset_num_batches=eval_config.subset_num_batches,
        )
    else:
        raise ValueError(f"Unexpected evaluator type '{eval_config.type}'")


def build_evaluators(cfg: TrainConfig, device: torch.device) -> List[Evaluator]:
    evaluators = []
    tokenizer = Tokenizer.from_train_config(cfg)
    for eval_cfg in cfg.evaluators:
        evaluators.append(build_evaluator(cfg, eval_cfg, tokenizer, device))
    return evaluators
