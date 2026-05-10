from pathlib import Path
from typing import Any, Dict, List, Optional, cast, Callable, Tuple

from torch.utils.data import DataLoader, DistributedSampler

from ..aliases import PathOrStr
from ..config import DataConfig, TrainConfig, TGConfig
from ..exceptions import OLMoConfigurationError
from ..tokenizer import Tokenizer
from ..torch_util import barrier, get_global_rank, get_world_size
from .collator import DataCollator
from .iterable_dataset import IterableDataset
from .memmap_dataset import MemMapDataset
from .tg_mask import TG_attention_bias, KProximal_TG_attention_bias, Height_TG_attention_bias, SentencepieceVocab
import torch 
__all__ = ["MemMapDataset", "DataCollator", "IterableDataset", "build_eval_dataloader", "build_train_dataloader", "SentencepieceVocab", "get_TG_generate_bias_func"]
        
class Soft_Alibilike_bias:
    def __init__(self, vocab_path:str = None, max_token_length:int = 2048, type:str = None) -> None:
        self.prox = KProximal_TG_attention_bias(vocab_path, max_token_length, max_token_length)

    def reset_state(self) -> None:
        self.prox.reset_state()
    
    def __call__(self, input_ids:torch.Tensor, update_state=False) -> Tuple[torch.Tensor, torch.Tensor]:
        mask, rel_pos, label_mask = self.prox.get_alibi_rel_pos(input_ids, update_state)
        mask += rel_pos * (1/2)
        return mask, label_mask


class HeadMixingBias:
    def __init__(self, config:List[TGConfig], train_config:TrainConfig, max_length:int) -> None:
        self.config = config
        self.TG_biases = []
        for head_config in self.config:
            self.TG_biases.append(get_TG_generate_bias_func(train_config, max_length, head_config.grammar_type))
    
    def reset_state(self) -> None:
        for TG_bias in self.TG_biases:
            TG_bias.reset_state()

    def __call__(self, input_ids:torch.Tensor, update_state=False) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, label_mask = [], None
        for gen_TG_bias, head_config in zip(self.TG_biases, self.config):
            mask, label_mask_head = gen_TG_bias(input_ids, update_state)
            masks.append(mask.unsqueeze(0).expand(head_config.n_heads, -1, -1))
            if label_mask_head is not None:
                label_mask = label_mask_head
        mask = torch.cat(masks, dim=0)
        return mask, label_mask

class TGCausalBias:
    def __init__(self, vocab_path:str, max_length:int) -> None:
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.max_length = max_length
        self.cur_length = 0
        self.causal_cache = None
    
    def reset_state(self) -> None:
        self.cur_length = 0

    def __call__(self, input_ids:torch.Tensor, update_state=False) -> Tuple[torch.Tensor, torch.Tensor]:
        pads_cnt = (input_ids == self.vocab.pad).sum().item()
        mask = None
        T = input_ids.shape[0]
        remove_len = (self.cur_length + T - self.max_length) if self.cur_length + T > self.max_length else 0
        pastT = (self.max_length - T)  if self.cur_length + T > self.max_length else self.cur_length
        update_T = T - pads_cnt
        if self.causal_cache is None:
            self.causal_cache = torch.tril(
                torch.ones(self.max_length, self.max_length, dtype=torch.bool),
            )
        mask = self.causal_cache[pastT:pastT+T, :pastT+T]
        if update_state:
            self.cur_length += - remove_len + update_T
        return mask, None

#TODO: promote parameter forwarding 
# Types that use tree-format data with standard causal attention (no TG bias)
_CAUSAL_TREE_TYPES = {"tree", "tgtree", "tree_shuffle", "tree_shuffle_mask",
                      "tree_noont", "tree_compress", "tree_triplecnt",
                      "terminal"}


def get_TG_generate_bias_func(train_config: TrainConfig, max_length:Optional[int] = None, TG_type:Optional[str]=None) -> TG_attention_bias:
    generate_TG_attention_bias = None
    vocab_path = train_config.tokenizer.vocabulary
    max_length = max_length if max_length is not None else train_config.model.max_sequence_length
    if TG_type is None:
        TG_type = train_config.model.transformer_grammar_type
        # Causal-only types (including new tree_noont, tree_compress, tree_triplecnt)
        if TG_type in _CAUSAL_TREE_TYPES:
            return None
        if TG_type[:5] == "pause":
            return None

    if TG_type == "mixing":
        generate_TG_attention_bias = HeadMixingBias(train_config.model.mix_head_type, train_config, max_length)
    elif TG_type=="tg":
        generate_TG_attention_bias = TG_attention_bias(vocab_path, max_length)
    elif TG_type[0:10]=="tgproximal":
        generate_TG_attention_bias = KProximal_TG_attention_bias(vocab_path, max_length, train_config.model.tg_proximal_k, TG_type[-3:]=="aug")
    elif TG_type[0:8]=="tgnomask":
        generate_TG_attention_bias = KProximal_TG_attention_bias(vocab_path, max_length, max_length, TG_type[-3:]=="aug")
    elif TG_type=="tgtree":
        generate_TG_attention_bias = TGCausalBias(vocab_path, max_length)

    return generate_TG_attention_bias

def build_memmap_dataset(
    train_config: TrainConfig, data_config: DataConfig, include_instance_metadata: bool = True
) -> MemMapDataset:
    paths: List[str]
    metadata: List[Dict[str, Any]] = []
    if data_config.paths:
        if data_config.datasets:
            raise OLMoConfigurationError("DataConfig.paths is mutually exclusive with DataConfig.datasets")
        paths = data_config.paths
        for path in paths:
            metadata.append({"path": str(path)})
    elif data_config.datasets:
        paths = []
        for label in sorted(data_config.datasets.keys()):
            label_paths = data_config.datasets[label]
            paths.extend(label_paths)
            metadata.extend([{"label": label}] * len(label_paths))
    else:
        raise OLMoConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")
    seq_length = train_config.model.max_sequence_length
    if train_config.model.transformer_grammar_type[:5]=="pause": 
        assert seq_length % (1+train_config.model.ispause) == 0,  f"model_ctx_len {seq_length} should be divided by pause length {(1+train_config.model.ispause)}"
        seq_length //= 1 + train_config.model.ispause 
    return MemMapDataset(
        *paths,
        chunk_size=seq_length,
        memmap_dtype=data_config.effective_memmap_dtype,
        metadata=metadata,
        include_instance_metadata=include_instance_metadata,
        pad_token_id=train_config.model.pad_token_id,
        eos_token_id=train_config.model.eos_token_id,
        pause_token_id=train_config.model.pause_token_id,
        generate_attention_mask=data_config.generate_attention_mask,
        generate_doc_lengths=data_config.generate_doc_lengths,
        label_mask_paths=cast(Optional[List[PathOrStr]], data_config.label_mask_paths),
        instance_filter_config=data_config.instance_filter,
        generate_TG_attention_bias=get_TG_generate_bias_func(train_config),
        transformer_grammar_type=train_config.model.transformer_grammar_type
    )


def build_eval_dataloader(
    train_config: TrainConfig,
    data_config: DataConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = build_memmap_dataset(train_config, data_config, include_instance_metadata=True)
    collator = DataCollator(pad_direction=data_config.pad_direction, pad_token_id=train_config.model.pad_token_id, 
                            generate_attention_mask=False, shuffle_tree=train_config.model.transformer_grammar_type)
    collator.vocab = SentencepieceVocab.from_vocab_file(train_config.tokenizer.vocabulary)
    if data_config.drop_last:
        # Make sure batch size is small enough.
        samples_per_device = len(dataset) // get_world_size()
        batch_size = min(batch_size, samples_per_device)
        assert batch_size > 0, f"dataset for {data_config.paths} is too small"
    seed = data_config.seed if data_config.seed is not None else train_config.seed
    sampler = DistributedSampler(
        dataset,
        drop_last=data_config.drop_last,
        shuffle=shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=data_config.num_workers,
        sampler=sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=None if data_config.num_workers == 0 else data_config.prefetch_factor,
        persistent_workers=False if data_config.num_workers == 0 else data_config.persistent_workers,
        timeout=data_config.timeout,
    )


def build_train_dataloader(
    train_config: TrainConfig,
    *,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    fs_local_rank: Optional[int] = None,
    include_instance_metadata: bool = False,
) -> DataLoader:
    assert train_config.device_train_batch_size is not None
    collator = DataCollator.from_train_config(train_config)
    if train_config.finetune_task is None:
        dataset = build_memmap_dataset(
            train_config, train_config.data, include_instance_metadata=include_instance_metadata
        )
    else:
        task_kwargs = {}
        from ..eval.downstream import label_to_task_map, Super_GLUE
        task_class = label_to_task_map[train_config.finetune_task]
        if isinstance(task_class, tuple):
            task_class, task_kwargs = task_class
        tokenizer = Tokenizer.from_train_config(train_config)
        dataset = task_class(tokenizer=tokenizer, 
                             generate_TG_attention_bias=get_TG_generate_bias_func(train_config, train_config.model.max_sequence_length),
                             transformer_grammar_type=train_config.model.transformer_grammar_type,
                             vocab_path=train_config.tokenizer.vocabulary,
                             split="train",
                             **task_kwargs)  # type: ignore
        if train_config.finetune_task in Super_GLUE:
            collator = dataset.collate_fn
    
    work_dir = Path(train_config.save_folder) / "train_data"
    if get_global_rank() == 0:
        if work_dir.is_dir() and not train_config.save_overwrite:
            raise OLMoConfigurationError(
                "train data working directory already exists, use --save_overwrite to overwrite"
            )
        else:
            work_dir.mkdir(exist_ok=True, parents=True)
    barrier()
    seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
    return DataLoader(
        IterableDataset(
            dataset,  # type: ignore
            train_config.global_train_batch_size,
            seed=seed,
            epoch=train_config.epoch or 0,
            shuffle=True,
            drop_last=train_config.data.drop_last,
            world_size=world_size,
            rank=rank,
            fs_local_rank=fs_local_rank,
            work_dir=work_dir,
        ),
        batch_size=train_config.device_train_batch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=collator,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )
