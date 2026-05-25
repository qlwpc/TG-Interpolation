from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import MeanMetric, Metric

from ..config import EvaluatorConfig, EvaluatorType, TrainConfig
from ..exceptions import OLMoConfigurationError
from ..tokenizer import Tokenizer
from ..torch_util import get_global_rank, get_world_size
from ..data.util import SequentialDistributedSampler
from .downstream import ICLMetric, BeamSearchICLMetric, DecomposedICLMetric, label_to_task_map, TGPerplexitySentenceLevelMetric, TGPerplexityDocumentLevelMetric, SyntacticGeneralizationMetric, BLiMPMetric, RougeMetric
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
]

beam_search_tasks = {
    "syntactic_generalization",
    "xsum"
}

def build_downstream_evaluator(
    train_config: TrainConfig,
    eval_cfg: EvaluatorConfig,
    tokenizer: Tokenizer,
    device: torch.device,
    is_unit_test=False,
) -> Evaluator:
    task_kwargs = {}
    task_class = label_to_task_map[eval_cfg.label]
    if isinstance(task_class, tuple):
        task_class, task_kwargs = task_class
        if eval_cfg.type == EvaluatorType.tg_doc or eval_cfg.label=="BLiMP":
            task_kwargs["device_eval_batch_size"] = eval_cfg.device_eval_batch_size or train_config.device_eval_batch_size
    task_kwargs["model_ctx_len"] = train_config.model.max_sequence_length
    task_kwargs["vocab_path"] = train_config.tokenizer.vocabulary
    task_kwargs["generate_TG_attention_bias"] = get_TG_generate_bias_func(train_config)
    task_kwargs["transformer_grammar_type"] = train_config.model.transformer_grammar_type
    task_kwargs["pause_token_id"] = train_config.model.pause_token_id
    if eval_cfg.samples_per_sent is not None:
        task_kwargs["samples_per_sent"] = eval_cfg.samples_per_sent
    if eval_cfg.tree_eval_type is not None:
        task_kwargs["tree_eval_type"] = eval_cfg.tree_eval_type
    if train_config.finetune_task is not None:
        task_kwargs["shots_num"] = 0
    ds_eval_dataset = task_class(tokenizer=tokenizer, **task_kwargs)  # type: ignore
    data_config = eval_cfg.data
    if is_unit_test:
        ds_eval_sampler = None
    elif eval_cfg.type in [EvaluatorType.tg_doc, EvaluatorType.tg_sent] or eval_cfg.label == "BLiMP":
        ds_eval_sampler = SequentialDistributedSampler(
            ds_eval_dataset,
            num_replicas=get_world_size(),
            rank=get_global_rank(),
        )
    else:
        ds_eval_sampler = DistributedSampler(
            ds_eval_dataset,
            drop_last=data_config.drop_last,
            shuffle=False,
            num_replicas=get_world_size(),
            rank=get_global_rank(),
            seed=train_config.seed,
        )
    eval_batch_size = eval_cfg.device_eval_batch_size or train_config.device_eval_batch_size
    if eval_cfg.label in beam_search_tasks or eval_cfg.type == EvaluatorType.beam_search_icl:
        eval_batch_size = 1
    
    ds_eval_dataloader = DataLoader(
        ds_eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=ds_eval_dataset.collate_fn,
        num_workers=data_config.num_workers,
        sampler=ds_eval_sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=data_config.prefetch_factor,
        persistent_workers=data_config.persistent_workers,
        timeout=data_config.timeout,
    )
    if eval_cfg.type == EvaluatorType.tg_sent:
        metric = TGPerplexitySentenceLevelMetric(
            vocab_path=train_config.tokenizer.vocabulary,
            metric_type=ds_eval_dataset.metric_type, 
            term_length=ds_eval_dataset.get_term_length(),
            device_eval_batch_size = eval_batch_size
        )
    elif eval_cfg.type == EvaluatorType.tg_doc:
        metric = TGPerplexityDocumentLevelMetric(
            vocab_path=train_config.tokenizer.vocabulary,
            metric_type=ds_eval_dataset.metric_type, 
            term_length=ds_eval_dataset.get_term_length(),
            device_eval_batch_size = eval_batch_size,
            dataset_length=len(ds_eval_dataset)
        )
    elif eval_cfg.label == "syntactic_generalization":
        metric = SyntacticGeneralizationMetric(metric_type=ds_eval_dataset.metric_type,
                                               tree_eval_type=getattr(ds_eval_dataset, "tree_eval_type", "default"))
    elif eval_cfg.label == "BLiMP":
        metric = BLiMPMetric(vocab_path=train_config.tokenizer.vocabulary,
                             metric_type=ds_eval_dataset.metric_type,
                             dataset_name=train_config.model.transformer_grammar_type,
                             device_eval_batch_size = eval_batch_size,
                             dataset_length=len(ds_eval_dataset),
                             samples_per_sent=ds_eval_dataset.SENT_SIZE,
                             tree_eval_type=ds_eval_dataset.tree_eval_type)
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
        metric = ICLMetric(metric_type=ds_eval_dataset.metric_type, 
                           vocab_path=train_config.tokenizer.vocabulary,
                            tree_eval_type=ds_eval_dataset.tree_eval_type,
                            doc_group=ds_eval_dataset.doc_group)

    if eval_cfg.type == EvaluatorType.tg_doc or eval_cfg.label == "BLiMP":
        assert(ds_eval_dataset.SENT_SIZE % eval_batch_size == 0 or
               ds_eval_dataset.TASK_SIZE % eval_batch_size == 0,
               f"SENT_SIZE={ds_eval_dataset.SENT_SIZE} and TASK_SIZE={ds_eval_dataset.TASK_SIZE} "
               f"not divisible by eval_batch_size={eval_batch_size}")

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

    if eval_config.type in [EvaluatorType.tg_doc, EvaluatorType.tg_sent, EvaluatorType.downstream, EvaluatorType.rouge, EvaluatorType.beam_search_icl]:
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
