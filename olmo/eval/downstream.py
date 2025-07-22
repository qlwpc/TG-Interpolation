import abc
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Union, Callable

import datasets
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torchmetrics import Metric
import numpy as np
import json, os

from olmo.util import load_hf_dataset, load_oe_eval_requests

from ..data.tg_mask import TG_attention_bias, SentencepieceVocab
from ..tokenizer import Tokenizer

log = logging.getLogger(__name__)

# Map from oe-eval metrics to metrics used here
METRIC_FROM_OE_EVAL = {"acc_raw": "acc", "acc_per_char": "len_norm", "acc_uncond": "pmi_dc"}
LOG_2_OF_E = 1.44269504089


class ICLMetric(Metric):
    # update method does not require access to global metric state
    full_state_update: bool = False

    def __init__(self, metric_type="acc") -> None:
        """metric_type: f1, acc, len_norm, pmi_dc, ce_loss, bpb"""
        super().__init__(sync_on_compute=True)

        self.metric_type = metric_type

        self.add_state("loglikelihoods", default=[], dist_reduce_fx=None)
        self.add_state("labels", default=[], dist_reduce_fx=None)

    def reset(
        self,
    ):
        self.loglikelihoods = []
        self.labels = []

    def update(self, batch: Dict[str, Any], lm_logits: torch.Tensor, dc_lm_logits=None):
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        if self.metric_type == "pmi_dc":
            assert dc_lm_logits is not None, "PMI_DC acc type selected but no domain conditional logits provided"

        for idx, (doc_id, cont_id) in enumerate(zip(batch["doc_id"], batch["cont_id"])):
            # [cont_len]: continuation is padded for batching
            cont_tokens = batch["continuation"][idx][: batch["cont_len"][idx]]
            # get logits from LM for the continuation: [cont_len, vocab]
            # batch['input_ids'][idx] -> ctx + cont + padding
            # -1 in both indices: lm_logits will be left shited 1 pos as 0th pos in input generates next token in the 0th pos of lm_logits
            lm_cont_logits = lm_logits[idx][
                batch["ctx_len"][idx] - 1 : batch["ctx_len"][idx] + batch["cont_len"][idx] - 1
            ]

            log_likelihood: torch.Tensor
            if self.metric_type == "pmi_dc":
                assert dc_lm_logits is not None
                # get domain conditional continuation logits: [cont_len, vocab]
                dc_lm_cont_logits = dc_lm_logits[idx][
                    batch["dc_len"][idx] - 1 : batch["dc_len"][idx] + batch["cont_len"][idx] - 1
                ]

                # gather log-probs at continuation token indices but divide by domain conditional prob
                log_likelihood = (
                    torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                    / torch.gather(dc_lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                )
            elif self.metric_type == "acc" or self.metric_type == "f1":
                # gather log-probs at continuation token indices
                log_likelihood = torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
            elif self.metric_type == "len_norm" or self.metric_type == "ce_loss":
                log_likelihood = (
                    torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum() / batch["cont_str_len"][idx]
                )
                if self.metric_type == "ce_loss":
                    log_likelihood = -log_likelihood
            elif self.metric_type == "bpb":
                # bits per byte
                log_likelihood = (
                    -torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                    / batch["cont_byte_len"][idx]
                    * LOG_2_OF_E
                )
            else:
                raise ValueError(self.metric_type)

            # because metric states cannot be dict/list of tuples, store this tuple as tensor: (doc_id, cont_id, metric_state)
            self.loglikelihoods.append(
                torch.Tensor((doc_id, cont_id, log_likelihood)).to(batch["continuation"][idx].device)
            )
            self.labels.append(
                torch.LongTensor((doc_id, cont_id, batch["label_id"][idx])).to(batch["label_id"][idx].device)
            )

    def compute(self) -> torch.Tensor:
        # states should have been synced from all accelerators at this point
        # account for duplicates here because of DistributedSampler compensating for drop_last=False
        loglikelihood_dict: Dict[int, Dict[int, float]] = {}
        label_dict = {}

        # collect labels
        for doc_id, cont_id, label_id in self.labels:
            if doc_id.item() not in label_dict:
                label_dict[doc_id.item()] = label_id.item()

        # collect loglikelihoods
        for doc_id, cont_id, loglikelihood in self.loglikelihoods:
            if int(doc_id.item()) not in loglikelihood_dict:
                loglikelihood_dict[int(doc_id.item())] = {}

            if int(cont_id.item()) not in loglikelihood_dict[int(doc_id.item())]:
                loglikelihood_dict[int(doc_id.item())][int(cont_id.item())] = loglikelihood

        # compute acc
        correct = []
        preds: Optional[List[float]] = None
        labels: Optional[List[int]] = None
        if self.metric_type == "f1":
            preds = []
            labels = []

        for doc_id in loglikelihood_dict:
            # each doc_id might have a different number of continuation
            num_continuations = len(loglikelihood_dict[doc_id].keys())
            loglikelihoods = torch.tensor([-float("inf")] * num_continuations)

            skip_document = False
            for cont_id in loglikelihood_dict[doc_id]:
                try:
                    loglikelihoods[cont_id] = loglikelihood_dict[doc_id][cont_id]
                except IndexError:
                    # We didn't process all of the continuations, so skip this document.
                    skip_document = True
                    break

            if skip_document:
                continue
            if self.metric_type in ["ce_loss", "bpb"]:
                correct.append(loglikelihoods[0])  # Only one answer is scored
            else:
                correct.append(1.0 if torch.argmax(loglikelihoods).item() == label_dict[doc_id] else 0.0)

            if self.metric_type == "f1":
                assert preds is not None
                assert labels is not None
                preds.append(torch.argmax(loglikelihoods).item())
                labels.append(label_dict[doc_id])

        if self.metric_type == "f1":
            assert preds is not None
            assert labels is not None
            # for NLI tasks, continuations are yes, no, neither, so idx=0 assigned to pos label
            score = f1_score(labels, preds, pos_label=0)
        else:
            score = sum(correct) / len(correct)

        return torch.tensor(score)

class TGperplexityDocumentLevelMetric(Metric):
    full_state_update: bool = False
    
    def __init__(
            self, 
            metric_type="doc_ppl", 
            vocab_path = None,
            term_length = None
        ) -> None:
        """metric_type: f1, acc, len_norm, pmi_dc, ce_loss, bpb"""
        super().__init__(sync_on_compute=True)

        self.metric_type = "doc_ppl"
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.term_length = term_length
        self.add_state("loglikelihoods", default=[], dist_reduce_fx=None)

    def reset(
        self,
    ):
        self.loglikelihoods = []


    def update(self, batch: Dict[str, Any], ce_loss:torch.Tensor, lm_logits: Optional[torch.Tensor] = None, dc_lm_logits=None):
        device = batch["input_ids"].device

        for idx, sent_id in enumerate(batch["sent_id"]):
            self.loglikelihoods.append(
                torch.Tensor((sent_id, ce_loss[idx])).to(device)
            )

    def compute(self) -> torch.Tensor: 
        # states should have been synced from all accelerators at this point
        # account for duplicates here because of DistributedSampler compensating for drop_last=False
        samples_per_sent = 300
        sent_cnt = len(self.loglikelihoods)//samples_per_sent
        loglikelihood_dict = torch.zeros(sent_cnt, dtype=torch.int32)
        loglikelihood_tensor = torch.empty(
            sent_cnt, 
            samples_per_sent, 
            dtype=torch.float32,
            device=self.loglikelihoods[0][0].device
        )
        # collect loglikelihoods
        for sent_id, loglikelihood in self.loglikelihoods:
            sent_id = int(sent_id.item()) - 1  # data sent_id count from 1
            loglikelihood_tensor[sent_id, loglikelihood_dict[sent_id]] = loglikelihood
            loglikelihood_dict[sent_id] += 1
        ppl = 0.0
        data_numwords = sum(self.term_length)
        ppl = torch.logsumexp(-loglikelihood_tensor, dim=1).sum().item()

        ppl = np.exp(-ppl / data_numwords)
        return torch.tensor(ppl)

# def print_tensor_data(tensor, precision=4, suppress_small=True):
#     """
#     打印PyTorch Tensor中的所有数据
    
#     参数:
#         tensor (torch.Tensor): 要打印的PyTorch张量
#         precision (int): 浮点数打印精度，默认为4位小数
#         suppress_small (bool): 是否抑制非常小的数用科学计数法显示，默认为True
#     """
#     # 将Tensor转换为numpy数组
#     # 设置numpy打印选项
#     # np.set_printoptions(
#     #     precision=precision,
#     #     threshold=np.inf,  # 显示所有元素
#     #     linewidth=np.inf,   # 不换行
#     #     suppress=suppress_small  # 抑制科学计数法
#     # )
#     torch.set_printoptions(
#         precision=4,    # 小数位数
#         threshold=10000000, # 触发缩略显示的阈值（元素数量）
#         edgeitems=3,    # 缩略时显示的首尾元素数量
#         linewidth=100000,  # 每行的字符宽度
#         sci_mode=False  # 是否禁用科学计数法
#     )
    
#     print(tensor)
    
#     # 恢复默认打印选项
#     torch.set_printoptions()

class TGperplexitySentenceLevelMetric(Metric):
    # update method does not require access to global metric state
    full_state_update: bool = False

    def __init__(
            self, 
            metric_type="sent_ppl", 
            vocab_path : str = None,
            term_length = None
        ) -> None:
        """metric_type: f1, acc, len_norm, pmi_dc, ce_loss, bpb"""
        super().__init__(sync_on_compute=True)

        self.metric_type = "sent_ppl"
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.term_length = term_length
        self.add_state("loglikelihoods", default=[], dist_reduce_fx=None)

    def reset(
        self,
    ):
        self.loglikelihoods = []

    def update(self, batch: Dict[str, Any], lm_logits: torch.Tensor, dc_lm_logits=None):
        logits_for_loss = lm_logits[..., :-1, :].contiguous()
        # print_tensor_data(batch["input_ids"])
        # print_tensor_data(batch["attention_bias"])
        # shape: (batch_size * seq_len, vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len)
        labels, label_mask, attention_mask, instance_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
            batch.get("instance_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, self.vocab.pad)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, self.vocab.pad)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), value=self.vocab.pad)
        labels = labels[..., 1:].contiguous()
        # shape: (batch_size * seq_len,)
        labels = labels.view(-1)
        ce_loss = F.cross_entropy(
            logits_for_loss, labels, ignore_index=self.vocab.pad, reduction="none")
        # print_tensor_data(ce_loss.view(batch["input_ids"].shape[0], -1))
        ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1).sum(dim=1)
        device = batch["input_ids"].device

        for idx, sent_id in enumerate(batch["sent_id"]):
            # [cont_len]: continuation is padded for batching
            # sent = tokens[idx]
            # sent_length = sum([self.vocab.is_terminal(sent[i]) for i in range(batch["input_ids"].shape[1])]) + 1 # add predict eos

            # because metric states cannot be dict/list of tuples, store this tuple as tensor: (doc_id, cont_id, metric_state)
            self.loglikelihoods.append(
                torch.Tensor((sent_id, ce_loss[idx])).to(device)
            )

    def compute(self) -> torch.Tensor:
        # states should have been synced from all accelerators at this point
        # account for duplicates here because of DistributedSampler compensating for drop_last=False
        samples_per_sent = 300
        sent_cnt = len(self.loglikelihoods)//samples_per_sent
        loglikelihood_dict = torch.zeros(sent_cnt, dtype=torch.int32)
        loglikelihood_tensor = torch.empty(
            sent_cnt, 
            samples_per_sent, 
            dtype=torch.float32,
            device=self.loglikelihoods[0][0].device
        )
        # collect loglikelihoods
        for sent_id, loglikelihood in self.loglikelihoods:
            sent_id = int(sent_id.item()) - 1  # data sent_id count from 1
            loglikelihood_tensor[sent_id, loglikelihood_dict[sent_id]] = loglikelihood
            # log.info(f"eval likeli is {loglikelihood}")
            loglikelihood_dict[sent_id] += 1
        
        ppl = 0.0
        data_numwords = sum(self.term_length)

        ppl = torch.logsumexp(-loglikelihood_tensor, dim=1).sum().item()
        # for sent_id in loglikelihood_dict:
        #     sent_logLs = torch.tensor(loglikelihood_dict[sent_id])
        #     cur_loss = torch.logsumexp(-sent_logLs, dim=0).item()
        #     ppl += cur_loss
        #     data_numwords += self.term_length[sent_id]

        ppl = np.exp(-ppl / data_numwords)
        return torch.tensor(ppl)


class TGPerplexityApproximationDataset(metaclass=abc.ABCMeta):
    metric_type: str

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: Union[str, Sequence[str], None] = None,
        model_ctx_len: int = 2048,
        split="validation",
        metric_type="sent",  # Override default metric type, whether be sent/doc
        generate_TG_attention_bias: Optional[TG_attention_bias] = None,
        vocab_path: str = None,
        device_eval_batch_size: int = 60, 
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len
        self.metric_type = metric_type 
        self.log_instances = 2  # Set to > 0 to log the first few instances as a sanity check
        self.batch_size = device_eval_batch_size
        self.SENT_SIZE = 300

        self.samples: List[Dict[str, Any]] = []
        self.term_len: List[int] = []
        dataset_names: Sequence[Optional[str]]
        if isinstance(dataset_name, str) or dataset_name is None:
            dataset_names = [dataset_name]
        else:
            dataset_names = dataset_name

        log.info(
                f"Starting loading TG_approx_ppl dataset"
            )

        import json, os
        for ds_name in dataset_names:
            with open(os.path.join(dataset_path,ds_name), 'r', encoding='utf-8') as file:
                dataset = json.load(file)

        self.dataset = dataset #datasets.concatenate_datasets(dataset_list)
        self._generate_TG_attention_bias = generate_TG_attention_bias
        self.prep_examples()
        self.reset()
        log.info(f"Loading Dataset finished")

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def get_term_length(self):
        return self.term_len

    def reset(self) -> None:
        self.cur_doc_id = 0
        self.sent_to_add = None
        self.num_evaled = 0

    def prep_examples(self):
        """Append doc_ids to each example so that they are processed together in the metric"""
        doc_id = 0
        cnt = 2 * 300
        cur = 0
        for sent in self.dataset:
            # if cur >= cnt:
            #     break
            # cur += 1
            # if cur % 100 == 0:
            #     log.info(self.token_decode(sent["input_ids"]))
            if (sent["sent_id"] >= len(self.term_len)):
                self.term_len.extend([0]*(sent["sent_id"] + 1 - len(self.term_len)))
                self.term_len[sent["sent_id"]] = sum([self.vocab.is_terminal(token) or token==self.vocab.eos for token in sent["input_ids"]])
            self.samples.append(
                {
                    "sent_id" : sent["sent_id"],
                    "doc_id": sent.get("doc_id") if sent.get("doc_id") is not None else 0,
                    "input_ids": sent["input_ids"], 
                }
            )
            doc_id += 1

    def pad_tokens_until_max(self, tokens, max_len=2048):
        """truncate from left if len(tokens) > model_ctx_len, max_len is not considered then
        queries are already truncated at max length of model_ctx_len
        this acts as additional check for all types of sequences in the batch
        """
        if len(tokens) > self.model_ctx_len:
            return tokens[-self.model_ctx_len :]
        else:
            # pad to max_len, but check again if this padding exceeded self.model_ctx_len
            # this time truncate from right side of the sequence because additional padding caused len(tokens) > self.model_ctx_len
            tokens = tokens + [self.vocab.pad] * (max_len - len(tokens))

            if len(tokens) > self.model_ctx_len:
                tokens = tokens[: self.model_ctx_len]

            return tokens

    def collate_fn(self, data):
        # pad to max length
        if self.metric_type=="doc" and data[0]["doc_id"] > self.cur_doc_id:
            self.cur_doc_id = data[0]["doc_id"]
            if self._generate_TG_attention_bias is not None:
                self._generate_TG_attention_bias.reset_state()
        
        if self._generate_TG_attention_bias is not None:
            for sample in data:
                sample["input_ids"] = self._generate_TG_attention_bias.convert_input_to_TG_format(sample["input_ids"])

        self.num_evaled += len(data)
        max_input_len = 0
        for sample in data:
            if len(sample["input_ids"]) > max_input_len:
                max_input_len = len(sample["input_ids"])

        sent_ids = []
        input_ids = []
        all_attention_bias = []
        all_label_mask = []
        # pad according to max_lengths
        for sample in data:
            pad_shape = (
                0, (max_input_len - len(sample["input_ids"]))
            )
            sent_ids.append(sample["sent_id"])
            cur_input_id = torch.LongTensor(self.pad_tokens_until_max(sample["input_ids"], max_len=max_input_len))

            attention_bias, label_mask = None, None
            if self._generate_TG_attention_bias is not None:
                attention_bias, label_mask = self._generate_TG_attention_bias(cur_input_id)
            input_ids.append(cur_input_id)
            
            if attention_bias is not None:
                if not isinstance(attention_bias, torch.Tensor):
                    attention_bias = torch.tensor(attention_bias)
                # Reshape to `(1, seq_len, seq_len)`
                while len(attention_bias.shape) < 3:
                    attention_bias = attention_bias.unsqueeze(0)
                all_attention_bias.append(attention_bias)
                # pad_value = False if attention_bias.dtype == torch.bool else float("-inf")
                # all_attention_bias.append(
                #     F.pad(
                #         attention_bias,
                #         pad_shape + pad_shape,
                #         value=pad_value,
                #     )
                # )

            # label_mask = sample.get("label_mask")
            if label_mask is not None:
                if not isinstance(label_mask, torch.Tensor):
                    label_mask = torch.tensor(label_mask)
                all_label_mask.append(label_mask)
                # all_label_mask.append(
                #     F.pad(
                #         label_mask.to(dtype=torch.bool),
                #         pad_shape,
                #         value=False,
                #     )
                # )

        batch = {
            "doc_id": data[0]["doc_id"] if self.metric_type=="doc" else None,
            "sent_id": torch.LongTensor(sent_ids),
            "input_ids": torch.stack(input_ids),
        }
        if all_attention_bias:
            batch["attention_bias"] = torch.stack(all_attention_bias)
        if all_label_mask:
            batch["label_mask"] = torch.stack(all_label_mask)

        if self.metric_type=="doc":
            if self.num_evaled % self.SENT_SIZE == 0:
                if self._generate_TG_attention_bias is not None:
                    self._generate_TG_attention_bias(self.sent_to_add, True)
            elif self.num_evaled % self.SENT_SIZE == self.batch_size:
                self.sent_to_add = torch.LongTensor(data[0]["input_ids"])
                batch["add_len"] = self.sent_to_add.shape[0]
        return batch

    def token_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def token_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

formula_dict = {
    'center_embed': ['[ (%plaus%) ] < [ (%implaus%) ]'],
    'center_embed_mod': ['[ (%plaus%) ] < [ (%implaus%) ]'],
    
    'cleft': ['[ (%np_mismatch%) - (%np_match%) ] + [ [ (%vp_mismatch%) ] - [ (%vp_match%) ] ]>0'],
    'cleft_modifier': ['[ (%np_mismatch%) - (%np_match%) ]+[ [ (%vp_mismatch%) ] - [ (%vp_match%) ] ]>0'],
    
    'fgd_subject': ['[ (%what_nogap%) > (%that_nogap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd_object': ['[ (%what_nogap%) > (%that_nogap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd_pp': ['[ (%what_nogap%) > (%that_nogap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd-embed3': ['[ (%what_no-gap%) > (%that_no-gap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd-embed4': ['[ (%what_no-gap%) > (%that_no-gap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd_hierarchy': ['[ (%what_nogap%) > (%that_nogap%)] & [ (%what_subjgap%) <  (%that_subjgap%) ]', '[ (%what_nogap%) = (%that_nogap%) ] & [ (%what_subjgap%) = (%that_subjgap%) ]'],   
    #TODO: why two formulas in fgd_hierarchy? we use the first formula
    
    'mvrr': ['[ (%reduced_ambig%) > (%unreduced_ambig%) ] & [ (%reduced_ambig%) > (%reduced_unambig%) ] & [ [ (%reduced_ambig%) - (%unreduced_ambig%) ] > [ (%reduced_unambig%) - (%unreduced_unambig%) ] ]'],
    'mvrr_mod': ['[ (%reduced_ambig%) > (%unreduced_ambig%) ] & [ (%reduced_ambig%) > (%reduced_unambig%) ] & [ [ (%reduced_ambig%) - (%unreduced_ambig%)] > [(%reduced_unambig%) - (%unreduced_unambig%)] ]'],
    
    'nn-nv-rpl': ['(%nn_ambig%)>(%nn_unambig%)', '(%nv_ambig%)>(%nv_unambig%)'], 
    
    'npi_orc_any': ['[ (%neg_pos%) < (%pos_pos%) ] & [ (%neg_neg%) < (%pos_neg%) ] & [ (%neg_pos%) < (%pos_neg%) ]'],
    'npi_orc_ever': ['[ (%neg_pos%) < (%pos_pos%) ] & [ (%neg_neg%) < (%pos_neg%) ] & [ (%neg_pos%) < (%pos_neg%) ]'],
    'npi_src_any': ['[ (%neg_pos%) < (%pos_pos%) ] & [ (%neg_neg%) < (%pos_neg%) ] & [ (%neg_pos%) < (%pos_neg%) ]'],
    'npi_src_ever': ['[ (%neg_pos%) < (%pos_pos%) ] & [ (%neg_neg%) < (%pos_neg%) ] & [ (%neg_pos%) < (%pos_neg%) ]'], 
    
    'npz_ambig': ['[ (%ambig_nocomma%) > (%ambig_comma%) ] &  [ (%ambig_nocomma%) > (%unambig_nocomma%) ]  & [ [ (%ambig_nocomma%) - (%ambig_comma%) ] > [ (%unambig_nocomma%) - (%unambig_comma%) ] ]'],
    'npz_ambig_mod': ['[ (%ambig_nocomma%) > (%ambig_comma%) ] &  [ (%ambig_nocomma%) > (%unambig_nocomma%) ]  & [ [ (%ambig_nocomma%) - (%ambig_comma%) ] > [ (%unambig_nocomma%) - (%unambig_comma%) ] ]'],
    'npz_obj': ['[ (%no-obj_no-comma%) > (%no-obj_comma%) ] &  [ (%no-obj_no-comma%) > (%obj_no-comma%) ] & [ [ (%no-obj_no-comma%) - (%no-obj_comma%) ] > [ (%obj_no-comma%) - (%obj_comma%) ] ]'],
    'npz_obj_mod': ['[ (%no-obj_no-comma%) > (%no-obj_comma%) ] &  [ (%no-obj_no-comma%) > (%obj_no-comma%) ] & [ [ (%no-obj_no-comma%) - (%no-obj_comma%) ] > [ (%obj_no-comma%) - (%obj_comma%) ] ]'],
    
    'number_orc': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'number_prep': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'number_src': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    
    'reflexive_orc_fem': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'reflexive_orc_masc': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'reflexive_prep_fem': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'], 
    'reflexive_prep_masc': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'reflexive_src_fem': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'], 
    'reflexive_src_masc': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    
    'subordination': ['[ (%sub_no-matrix%) > (%no-sub_no-matrix%) ] & [ (%sub_matrix%) < (%no-sub_matrix%) ]'], 
    'subordination_orc-orc': ['[ (%sub_no-matrix%) > (%no-sub_no-matrix%) ] & [ (%sub_matrix%) < (%no-sub_matrix%) ]'], 
    'subordination_pp-pp': ['[ (%sub_no-matrix%) > (%no-sub_no-matrix%) ] & [ (%sub_matrix%) < (%no-sub_matrix%) ]'], 
    'subordination_src-src': ['[ (%sub_no-matrix%) > (%no-sub_no-matrix%) ] & [ (%sub_matrix%) < (%no-sub_matrix%) ]']
}
test_suite_dict = {
    "Agreement" : ["number_orc", "number_prep", "number_src"], 
    "Center_Embedding" : ["center_embed", "center_embed_mod"],
    "Garden_Path_Effects" : ["mvrr", "mvrr_mod", "npz_ambig", "npz_ambig_mod", "npz_obj", "npz_obj_mod"],
    "Gross_Syntactic_Expectation" : ["subordination", "subordination_orc-orc", "subordination_pp-pp", "subordination_src-src"],
    "Licensing" : ["npi_orc_any", "npi_orc_ever", "npi_src_any", "npi_src_ever", \
            "reflexive_orc_fem", "reflexive_orc_masc", "reflexive_prep_fem", "reflexive_prep_masc", "reflexive_src_fem", "reflexive_src_masc"],
    "Long_Distance_Dependencies" : ["fgd_subject", "fgd_object", "fgd_pp", "fgd-embed3", "fgd-embed4", "fgd_hierarchy", "cleft", "cleft_modifier"],
    # "nn-nv-rpl" : ["nn-nv-rpl"] # extra test in SG but not in test suites
}

class SyntacticGeneralizationMetric(Metric):
    def __init__(
            self, 
            metric_type="syntactic_generation", 
        ) -> None:
        super().__init__(sync_on_compute=True)

        self.metric_type = metric_type
        self.test_suite_dict = test_suite_dict
        self.map_task_dict = {}
        for key in test_suite_dict:
            self.add_state(key, default=[], dist_reduce_fx=None)
            for task in test_suite_dict[key]:
                self.map_task_dict[task] = key

    def reset(self):
        for key in test_suite_dict:
            setattr(self, key, [])
    
    def update(self, task, score_dict):
        '''
        input: task, condition probability variables, then eval with formula
        '''
        print(f"task is {task} score is {score_dict}")
        formula = formula_dict[task][0]
        keys = re.findall(r"%([\w|-]+)%", formula)
        keys = set(keys)
        for key in keys:
            formula = formula.replace(
                "(%{}%)".format(key),
                str(score_dict[key]),
                )
        formula = formula.replace("[", "(")
        formula = formula.replace("]", ")")
        
        result = eval(formula)
        print(f"result is {result}")
        getattr(self, self.map_task_dict[task]).append(result)

    def compute(self) -> Dict[str, float]:
        acc_dict = {}
        avg_acc = 0.0
        for key in self.test_suite_dict:
            acc_dict[key] = sum(getattr(self, key))
            if acc_dict[key]>0:
                acc_dict[key] /= len(getattr(self, key))
            if key != 'nn-nv-rpl':
                avg_acc += acc_dict[key]
        acc_dict["avg"] = avg_acc / len(self.test_suite_dict)
        return acc_dict

class SGDataset(metaclass=abc.ABCMeta):
    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: Union[str, Sequence[str], None] = None,
        model_ctx_len: int = 2048,
        split="validation",
        metric_type="SG",
        vocab_path: str = None,
    ):
        self.task_list = ["center_embed", "center_embed_mod", "cleft", "cleft_modifier", "fgd_subject", "fgd_object", "fgd_pp", "fgd-embed3", \
            "fgd-embed4", "fgd_hierarchy", "mvrr", "mvrr_mod", "npi_orc_any", "npi_orc_ever", "npi_src_any", \
            "npi_src_ever", "npz_ambig", "npz_ambig_mod", "npz_obj", "npz_obj_mod", "number_orc", "number_prep", "number_src", \
            "reflexive_orc_fem", "reflexive_orc_masc", "reflexive_prep_fem", "reflexive_prep_masc", "reflexive_src_fem", "reflexive_src_masc", \
            "subordination", "subordination_orc-orc", "subordination_pp-pp", "subordination_src-src",   
            # "nn-nv-rpl" don't include this test
        ]

        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.metric_type = metric_type
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.prep_examples()

    def prep_examples(self):
        self.samples : List[List[Dict[str, List]]] = [] 
        for task in self.task_list:
            # if task not in test_suite_dict["Agreement"]: continue
            with open(os.path.join(self.dataset_path, task+".json"), 'r', encoding='utf-8') as file:
                dataset = json.load(file)
                for case in dataset["data"]:
                    cur = []
                    for sent in case:
                        sent["task"] = task
                        sent["input_ids"] = torch.LongTensor([self.vocab.bos] + self.tokenizer.encode(" " + sent["input"], add_special_tokens=False)).unsqueeze(0)
                        start = end = -1
                        for i,x in enumerate(sent["tag"][0]):
                            if x == 1:
                                end = i
                                if start == -1:
                                    start = i - 1
                        sent["tag_start"] = start + 1  # add bos token position
                        sent["tag_end"] = end + 1      # add bos token position
                        assert(sum(sent['tag'][0]) == end-start)
                        if sent["condition_name"] in formula_dict[task][0]:
                            cur.append(sent)
                    self.samples.append(cur)
    
    def __getitem__(self, index):
        return self.samples[index]
    
    def reset(self) -> None:
        return
    
    def __len__(self):
        return len(self.samples)
    
    def collate_fn(self, data):
        return data[0]

class ICLMultiChoiceTaskDataset(metaclass=abc.ABCMeta):
    """Only supports zero-shot for now."""

    metric_type: str

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: Union[str, Sequence[str], None] = None,
        model_ctx_len: int = 2048,
        split="validation",
        metric_type=None,  # Override default metric type
        prompts=[None],  # List of prompt variants to use
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len
        self.prompts = prompts
        self.current_prompt = None
        if metric_type is not None:
            self.metric_type = metric_type
        self.log_instances = 0  # Set to > 0 to log the first few instances as a sanity check

        self.samples: List[Dict[str, Any]] = []
        dataset_names: Sequence[Optional[str]]
        if isinstance(dataset_name, str) or dataset_name is None:
            dataset_names = [dataset_name]
        else:
            dataset_names = dataset_name

        dataset_list = []
        for ds_name in dataset_names:
            dataset = load_hf_dataset(self.dataset_path, ds_name, split)
            dataset_list.append(dataset)
        self.dataset = datasets.concatenate_datasets(dataset_list)

        # prep examples
        self.prep_examples()

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def prep_examples(self):
        """Append doc_ids to each example so that they are processed together in the metric"""
        doc_id = 0
        for doc in self.dataset:
            for prompt in self.prompts:
                self.current_prompt = prompt
                # from EAI harness
                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                continuations = self.doc_to_continuations(doc)
                label_id = self.doc_to_label(doc)
                doc_text = self.doc_to_text(doc)
                ctx = self.token_encode(doc_text)
                dc = self.token_encode(self.doc_to_domain_conditional(doc))
                if self.log_instances > 0:
                    self.log_instances -= 1
                    ds_name = self.dataset_name
                    if isinstance(ds_name, list):
                        ds_name = ds_name[0]
                    log.info(
                        f"Sample doc from ({self.dataset_path}, {ds_name}, {self.current_prompt}):"
                        + f"\ndoc_text: {doc_text}\ncontinuations: {continuations}"
                    )

                for cont_id, continuation_str in enumerate(continuations):
                    cont_str_len = len(continuation_str) - 1  # continuation contain leading blank
                    cont_byte_len = len(continuation_str[1:].encode("utf-8"))
                    continuation = self.token_encode(continuation_str)

                    # query, remove last token from continuation, truncate from left is longer than model ctx length
                    query = ctx + continuation[:-1]
                    query = query[-self.model_ctx_len :]
                    # this will be different from len(ctx) when truncated by model_ctx_len
                    actual_ctx_len = len(query) - len(continuation) + 1

                    # get domain conditional query
                    # we don't expect this to be longer than self.model_ctx_len and it won't make sense to truncate from left
                    dc_query = dc + continuation[:-1]

                    # form a sample
                    self.samples.append(
                        {
                            "doc_id": doc_id,
                            "cont_id": cont_id,
                            "ctx": ctx,
                            "continuation": continuation,
                            "ctx_len": actual_ctx_len,
                            "dc_len": len(dc),
                            "cont_len": len(
                                continuation
                            ),  # even if query has last token removed, LM will output same cont len
                            "cont_str_len": cont_str_len,
                            "cont_byte_len": cont_byte_len,
                            "query": query,  # remove last token from continuation
                            "dc_query": dc_query,
                            "label_id": label_id,
                        }
                    )

                doc_id += 1

    def pad_tokens_until_max(self, tokens, max_len=2048):
        """truncate from left if len(tokens) > model_ctx_len, max_len is not considered then
        queries are already truncated at max length of model_ctx_len
        this acts as additional check for all types of sequences in the batch
        """
        if len(tokens) > self.model_ctx_len:
            return tokens[-self.model_ctx_len :]
        else:
            # pad to max_len, but check again if this padding exceeded self.model_ctx_len
            # this time truncate from right side of the sequence because additional padding caused len(tokens) > self.model_ctx_len
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))

            if len(tokens) > self.model_ctx_len:
                tokens = tokens[: self.model_ctx_len]

            return tokens

    def collate_fn(self, data):
        # pad to max length
        # 'ctx', 'continuation', 'query' can all have variable length
        max_ctx_len = 0
        max_cont_len = 0
        max_query_len = 0
        max_dc_query_len = 0

        for sample in data:
            if len(sample["ctx"]) > max_ctx_len:
                max_ctx_len = len(sample["ctx"])

            if len(sample["continuation"]) > max_cont_len:
                max_cont_len = len(sample["continuation"])

            if len(sample["query"]) > max_query_len:
                max_query_len = len(sample["query"])

            if len(sample["dc_query"]) > max_dc_query_len:
                max_dc_query_len = len(sample["dc_query"])

        doc_ids = []
        cont_ids = []
        ctxs = []
        continuations = []
        ctx_lens = []
        dc_lens = []
        cont_lens = []
        cont_str_lens = []
        cont_byte_lens = []
        queries = []
        dc_queries = []
        label_ids = []

        # pad according to max_lengths
        for sample in data:
            doc_ids.append(sample["doc_id"])
            cont_ids.append(sample["cont_id"])

            ctxs.append(torch.LongTensor(self.pad_tokens_until_max(sample["ctx"], max_len=max_ctx_len)))
            continuations.append(
                torch.LongTensor(self.pad_tokens_until_max(sample["continuation"], max_len=max_cont_len))
            )

            ctx_lens.append(sample["ctx_len"])
            dc_lens.append(sample["dc_len"])
            cont_lens.append(sample["cont_len"])
            cont_str_lens.append(sample["cont_str_len"])
            cont_byte_lens.append(sample["cont_byte_len"])

            queries.append(torch.LongTensor(self.pad_tokens_until_max(sample["query"], max_len=max_query_len)))
            dc_queries.append(
                torch.LongTensor(self.pad_tokens_until_max(sample["dc_query"], max_len=max_dc_query_len))
            )

            label_ids.append(sample["label_id"])

        batch = {
            "doc_id": torch.LongTensor(doc_ids),
            "cont_id": torch.LongTensor(cont_ids),
            "ctx": torch.stack(ctxs),
            "continuation": torch.stack(continuations),
            "ctx_len": torch.LongTensor(ctx_lens),
            "dc_len": torch.LongTensor(dc_lens),
            "cont_len": torch.LongTensor(cont_lens),  # since query has last token removed from continuation
            "cont_str_len": torch.LongTensor(cont_str_lens),
            "cont_byte_len": torch.LongTensor(cont_byte_lens),
            "input_ids": torch.stack(queries),
            "dc_input_ids": torch.stack(dc_queries),
            "label_id": torch.LongTensor(label_ids),
        }

        return batch

    def token_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def token_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    @abc.abstractmethod
    def doc_to_text(self, doc) -> str:
        """Match EAI eval harness
        returns a single context string
        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_continuations(self, doc) -> List[str]:
        """Match EAI eval harness
        returns a list of continuations
        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_label(self, doc) -> int:
        """Match EAI eval harness
        returns continuation id which corresponds to true label
        """
        raise NotImplementedError

    def doc_to_domain_conditional(self, doc) -> str:
        """Provide string for domain conditional normalization
        by default its blank string, continuation normalized by prob conditioned on a blank
        """
        del doc
        return " "


class PIQA(ICLMultiChoiceTaskDataset):
    """PIQA sends context in the following fashion: "Question: GOAL\nAnswer:"
    space added as prefix to each continuation

    implement PMI_DC

    {
        'goal': "How do I ready a guinea pig cage for it's new occupants?",
        'sol1': 'Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.',
        'sol2': 'Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish.',
        'label': 0
    }
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="piqa",
        dataset_name="plain_text",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return "Question: " + doc["goal"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + doc["sol1"], " " + doc["sol2"]]

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class HellaSwag(ICLMultiChoiceTaskDataset):
    """HellaSwag concats "ACTIVITY_LABEL: CTX_A CTX_B.capitalize()" to form context and then sends endings as continuations
        space added as prefix to each continuation

    {
        'activity_label': 'Roof shingle removal',
        'ctx_a': 'A man is sitting on a roof.',
        'ctx_b': 'he',
        'ctx': 'A man is sitting on a roof. he',
        'endings': ['is using wrap to wrap a pair of skis.', 'is ripping level tiles off.', "is holding a rubik's cube.", 'starts pulling up roofing on a roof.'],
        'label': '3'
    }
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="hellaswag",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")

        return text

    def doc_to_text(self, doc):
        return self.preprocess(doc["activity_label"] + ": " + doc["ctx_a"] + " " + doc["ctx_b"].capitalize())

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + self.preprocess(ending) for ending in doc["endings"]]

    def doc_to_label(self, doc):
        return int(doc["label"])

    def doc_to_domain_conditional(self, doc):
        domain_conditional = self.preprocess(doc["ctx_b"].capitalize())

        # ensure non 0 len domain conditional
        if len(domain_conditional) == 0:
            return self.preprocess(doc["ctx_a"]).split(" ")[-1]

        return domain_conditional


class WinoGrande(ICLMultiChoiceTaskDataset):
    """Prompt: split sentence at _ "SENTENCE[:idx] + OPTION1/OPTION2", where idx = SENTENCE.index("_")
        implement PMI_DC
        acc, random at 50%
        continuation is everything in setnence after '_' (" SENTENCE[idx:].strip()")

        Req_loglikelihood('People think Samantha', ' is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.')
        Req_loglikelihood('People think Rebecca', ' is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.')

    {
        'sentence': 'People think _ is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.',
        'option1': 'Samantha',
        'option2': 'Rebecca',
        'answer': '2'
    }

    TODO: might need to write custom metric for Winogrande
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="winogrande",
        dataset_name="winogrande_xl",
    ):
        # all winogrande datasets have same val set
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def prep_examples(self):
        """Overwrite for WinoGrande as multiple ctx, single continuation"""
        doc_id = 0
        for doc in self.dataset:
            # here ctx is a list
            ctxs = self.doc_to_text(doc)
            dcs = self.doc_to_domain_conditional(doc)

            continuation_str = self.doc_to_continuations(doc)
            label_id = self.doc_to_label(doc)
            cont_str_len = len(continuation_str) - 1  # continuations contain leading blank space
            cont_byte_len = len(continuation_str[1:].encode("utf-8"))

            # tokenize
            continuation = self.token_encode(continuation_str)

            for cont_id, (ctx, dc) in enumerate(zip(ctxs, dcs)):
                ctx = self.token_encode(ctx)
                dc = self.token_encode(dc)

                # query, remove last token from continuation, truncate from left is longer than model ctx length
                query = ctx + continuation[:-1]
                query = query[-self.model_ctx_len :]

                # get domain conditional query
                # we don't expect this to be longer than self.model_ctx_len and it won't make sense to truncate from left
                dc_query = dc + continuation[:-1]

                # form a sample
                self.samples.append(
                    {
                        "doc_id": doc_id,
                        "cont_id": cont_id,
                        "ctx": ctx,
                        "continuation": continuation,
                        "ctx_len": len(ctx),
                        "dc_len": len(dc),
                        "cont_len": len(
                            continuation
                        ),  # even if query has last token removed, LM will output same cont len
                        "cont_str_len": cont_str_len,
                        "cont_byte_len": cont_byte_len,
                        "query": query,  # remove last token from continuation
                        "dc_query": dc_query,
                        "label_id": label_id,
                    }
                )

            doc_id += 1

    def doc_to_text(self, doc):
        # special case where there are multiple ctx and single continuation
        pronoun_loc = doc["sentence"].index("_")

        ctx = []
        for option in [doc["option1"], doc["option2"]]:
            ctx.append(doc["sentence"][:pronoun_loc] + option)

        return ctx

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        pronoun_loc = doc["sentence"].index("_") + 1
        return " " + doc["sentence"][pronoun_loc:].strip()

    def doc_to_label(self, doc):
        return int(doc["answer"]) - 1

    def doc_to_domain_conditional(self, doc):
        """same number of domain conditionals as context"""
        return [doc["option1"], doc["option2"]]


class OpenBookQA(ICLMultiChoiceTaskDataset):
    """OBQA: question_stem is sent as context (no special prompt format) and choices are sent as continuation
        space added as prefix to each continuation

        implement PMI_DC

    {
        'question_stem': 'Frilled sharks and angler fish live far beneath the surface of the ocean, which is why they are known as',
        'choices': {'text': ['Deep sea animals', 'fish', 'Long Sea Fish', 'Far Sea Animals'],
        'label': ['A', 'B', 'C', 'D']},
        'answerKey': 'A'
    }
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="openbookqa",
        dataset_name="main",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return doc["question_stem"]

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + choice for choice in doc["choices"]["text"]]

    def doc_to_label(self, doc):
        return ["A", "B", "C", "D"].index(doc["answerKey"].strip())

    def doc_to_domain_conditional(self, doc):
        return doc["question_stem"].strip().split(" ")[-1]


class BoolQ(ICLMultiChoiceTaskDataset):
    """Prompt: "PASSAGE\nQuestion: QUESTION?\nAnswer:"
    acc, random at 50% (SuperGLUE)
    continuation: yes, no

    {
        'question': 'is ncis new orleans over for the season',
        'passage': 'NCIS: New Orleans (season 4) -- The fourth season of NCIS: New Orleans premiered on September 26, 2017 on CBS. The series continues to air following Bull, Tuesday at 10:00 p.m. (ET) and contained 24 episodes. The season concluded on May 15, 2018.',
        'label': 1
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="boolq",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return doc["passage"] + "\nQuestion: " + doc["question"] + "?\nAnswer:"

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        return [" yes", " no"]

    def doc_to_label(self, doc):
        # if doc['answer'] is True, return index of " yes" which is 0
        if doc["answer"]:
            return 0
        else:
            return 1

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class SciQ(ICLMultiChoiceTaskDataset):
    """SciQ sends context as "SUPPORT\nQuestion: QUESTION\nAnswer:" and then distractors + correct_answer as continuations
        space added as prefix to each continuation

        implement PMI_DC

    {
        'question': 'Who proposed the theory of evolution by natural selection?',
        'distractor3': 'Scopes',
        'distractor1': 'Linnaeus',
        'distractor2': 'shaw',
        'correct_answer': 'darwin',
        'support': ''
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="sciq",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return doc["support"].strip() + "\nQuestion: " + doc["question"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [
            " " + doc["distractor1"],
            " " + doc["distractor2"],
            " " + doc["distractor3"],
            " " + doc["correct_answer"],
        ]

    def doc_to_label(self, doc):
        del doc
        return 3

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class ArcEasy(ICLMultiChoiceTaskDataset):
    """ArcEasy creates context with "Question: QUESTION\nAnswer:" and sends the choices as continuations
        space added as prefix to each continuation

    {
        'question': 'Which technology was developed most recently?',
        'choices': {'text': ['cellular telephone', 'television', 'refrigerator', 'airplane'],
        'label': ['A', 'B', 'C', 'D']},
        'answerKey': 'A'
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="ai2_arc",
        dataset_name="ARC-Easy",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return "Question: " + doc["question"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + choice for choice in doc["choices"]["text"]]

    def doc_to_label(self, doc):
        # some doc["answerKey"] are stored as numbers
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

        if doc["answerKey"] in num_to_letter:
            doc["answerKey"] = num_to_letter[doc["answerKey"]]

        return ["A", "B", "C", "D", "E"].index(doc["answerKey"])

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class ArcChallenge(ArcEasy):
    """ArcChallenge follows the same prompt format as ArcEasy.
    implement PMI_DC
    """

    metric_type = "len_norm"  # Ideally "pmi_dc"

    def __init__(
        self,
        tokenizer,
        dataset_path="ai2_arc",
        dataset_name="ARC-Challenge",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )


class ArcEasyCELoss(ArcEasy):
    """ArcEasyCELoss is ARCEasy using an alternate ce_loss metric"""

    metric_type = "ce_loss"

    def doc_to_continuations(self, doc):
        # We only consider the correct answer for this metric
        answer = doc["choices"]["text"][self.doc_to_label(doc)]
        return [" " + answer]

    def doc_to_label(self, doc):
        return 0


class BasicArithmetic(ArcEasy):
    """This is a basic arithmetic task follows the same prompt format as ArcEasy.
    Example:
    {"id": "q85_1d1d_max1d_plus",
    "question": "Calculate 2 + 5 =",
    "choices": {"text": ["8", "7", "6", "17"],
    "label": ["A", "B", "C", "D"]},
    "answerKey": "B", "type_tag": "easy"}

    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="allenai/basic_arithmetic",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )


class CommonsenseQA(ArcEasy):
    """CommonsenseQA
    Example:
    {'id': 'e68fb2448fd74e402aae9982aa76e527',
    'question': 'Where are  you likely to find a hamburger?',
    'question_concept': 'hamburger',
    'choices': {'label': ['A', 'B', 'C', 'D', 'E'],
    'text': ['fast food restaurant', 'pizza', 'ground up dead cows', 'mouth', 'cow carcus']},
    'answerKey': 'A'}
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="tau/commonsense_qa",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )


class SocialIQa(ICLMultiChoiceTaskDataset):
    """SocialIQa
    Example:
    {'context': 'Jordan was in charge of taking the food on the camping trip and left all the food at home.',
     'question': 'How would Jordan feel afterwards?',
     'answerA': 'horrible that he let his friends down on the camping trip',
     'answerB': "happy that he doesn't need to do the cooking on the trip",
     'answerC': 'very proud and accomplished about the camping trip', 'label': '1'}
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="social_i_qa",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return "Question: " + doc["context"] + " " + doc["question"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [
            " " + doc["answerA"],
            " " + doc["answerB"],
            " " + doc["answerC"],
        ]

    def doc_to_label(self, doc):
        return int(doc["label"]) - 1

    def doc_to_domain_conditional(self, doc):
        return "Answer:"


class COPA(ICLMultiChoiceTaskDataset):
    """Prompt: "PREMISE.strip()[:-1] because/therefore"
    Req_loglikelihood('The pair of students came under scrutiny by the teacher because', ' the students both received excellent grades.'
    continuations: CHOICE1/CHOICE2

    "cause": "because",
    "effect": "therefore",

    implement PMI_DC
    acc, random at 50%

    {
        'premise': 'The pair of students came under scrutiny by the teacher.',
        'choice1': 'The students both received excellent grades.',
        'choice2': 'Their responses on the assignment were identical.',
        'question': 'cause',
        'label': 1
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="super_glue",
        dataset_name="copa",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        connector = "because" if doc["question"] == "cause" else "therefore"

        # remove the period
        return doc["premise"].strip()[:-1] + " " + connector

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        def convert_choice(choice):
            return choice[0].lower() + choice[1:]

        return [" " + convert_choice(doc["choice1"]), " " + convert_choice(doc["choice2"])]

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        return "because" if doc["question"] == "cause" else "therefore"


class RTE(ICLMultiChoiceTaskDataset):
    """Prompt: "SENTENCE1\nQuestion: SENTENCE2 True or False?\nAnswer:"
    implement PMI_DC
    acc, random at 50% (GLUE)
    continuations: True, False

    {
        'sentence1': 'The number of Danes opposed to swapping the krone for the euro has increased slightly to 35.3 percent, up from 34.6 percent in April, according to a poll published on Thursday by Danske Bank.',
        'sentence2': 'The introduction of the euro has been opposed.',
        'label': 0,
    }
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="glue",
        dataset_name="rte",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return doc["sentence1"] + "\nQuestion: " + doc["sentence2"] + " True or False?\nAnswer:"

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        return [" True", " False"]

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class CommitmentBank(ICLMultiChoiceTaskDataset):
    """Prompt: "PREMISE\nQuestion: HYPOTHESIS. True, False or Neither?\nAnswer:"
    continuations: True, False, Neither

        implement PMI_DC
        acc/F1, random at 33% acc. (SuperGLUE)

    {
        'premise': 'Then they would awake, terrified and sweating, to find themselves in white starched linen, in a comfortable bed, in peaceful England. And all would be well. It may be said that although he survived it the siege nevertheless had a bad effect on the Collector.',
        'hypothesis': 'the siege nevertheless had a bad effect on the Collector',
        'label': 0
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="super_glue",
        dataset_name="cb",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return doc["premise"] + "\nQuestion: " + doc["hypothesis"] + ". True, False or Neither?\nAnswer:"

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        return [" True", " False", " Neither"]

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class MRPC(ICLMultiChoiceTaskDataset):
    """Prompt for MRPC is formed using "Sentence 1: SENTENCE1\nSentence 2: SENTENCE2\nQuestion: Do both sentences mean the same thing?\nAnswer:"
    acc/F1, random at 50% acc. (GLUE)
    continuations: yes and no

    {
        'sentence1': 'In fiction : Edward P. Jones ( " The Known World " ) and Scott Spencer ( " A Ship Made of Paper " ) .',
        'sentence2': 'The fifth nominee for fiction is Scott Spencer , for A Ship Made of Paper .',
        'label': 0
    }
    """

    metric_type = "f1"

    def __init__(
        self,
        tokenizer,
        dataset_path="glue",
        dataset_name="mrpc",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    @classmethod
    def preprocess(cls, string: str) -> str:
        string = string.replace(" n't", "n't")
        string = string.replace(" )", ")")
        string = string.replace("( ", "(")
        string = string.replace('" ', '"')
        string = string.replace(' "', '"')

        string = re.sub(r" (['.,])", r"\1", string)

        return string

    def doc_to_text(self, doc):
        return (
            "Sentence 1: "
            + self.preprocess(doc["sentence1"])
            + "\nSentence 2: "
            + self.preprocess(doc["sentence2"])
            + "\nQuestion: Do both sentences mean the same thing?\nAnswer:"
        )

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        return [" yes", " no"]

    def doc_to_label(self, doc):
        # if doc['label'] is True, return index of " yes" which is 0
        if doc["label"]:
            return 0
        else:
            return 1

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class SST2(ICLMultiChoiceTaskDataset):
    """SST2 task formats prompts as "SENTENCE\nQuestion: Is this sentence positive or negative?\nAnswer:"
    some preprocessing done on sentence

    constructs 2 requests, 1 for positive and another for negative
    positive and negative have just 1 token in tokenizer
    positive: 1313
    negative: 2430

    implement PMI_DC
    acc, random at 50% (GLUE)

    {
        'sentence': "harrison 's flowers puts its heart in the right place , but its brains are in no particular place at all . ",
        'label': 1,
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="glue",
        dataset_name="sst2",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    @classmethod
    def preprocess(cls, string: str) -> str:
        string = string.replace(" n't", "n't")
        string = string.replace(" )", ")")
        string = string.replace("( ", "(")
        string = string.replace('" ', '"')
        string = string.replace(' "', '"')

        string = re.sub(r" (['.,])", r"\1", string)

        return string

    def doc_to_text(self, doc):
        return self.preprocess(doc["sentence"]) + "\nQuestion: Is this sentence positive or negative?\nAnswer:"

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        # # {1: "positive", 0: "negative"}
        return [" negative", " positive"]

    def doc_to_label(self, doc):
        # {1: "positive", 0: "negative"}
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class MMLU(ICLMultiChoiceTaskDataset):
    """MMLU creates context with "Question: QUESTION\nAnswer:" and sends the choices as continuations
           space added as prefix to each continuation

       {
           'question': "Which of the following terms describes the body's ability to maintain its normal state?",
           'subject': 'anatomy',
           'choices': ['Anabolism', 'Catabolism', 'Tolerance', 'Homeostasis'],
    '       answer': 3
        }
    """

    metric_type = "len_norm"  # Ideally pmi_dc

    _subcategories = {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"],
    }

    _categories = {
        "stem": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
        "humanities": ["history", "philosophy", "law"],
        "social_sciences": ["politics", "culture", "economics", "geography", "psychology"],
        "other": ["other", "business", "health"],
    }

    def __init__(
        self,
        tokenizer,
        dataset_path="hails/mmlu_no_train",
        dataset_name=None,
        split="validation",
        prompt_variations=None,
        mc_labels=False,
        metric_type=None,
    ):
        dataset_names = []
        # Collect the relevant categories
        if dataset_name in MMLU._categories:
            for sub_cat in MMLU._categories[dataset_name]:
                for name, cats in MMLU._subcategories.items():
                    if sub_cat in cats:
                        dataset_names.append(name)
        elif dataset_name in MMLU._subcategories:
            dataset_names.append(dataset_name)
        else:  # E.g., "math"
            for name, cats in MMLU._subcategories.items():
                if dataset_name in cats:
                    dataset_names.append(name)
        self.dev_set = {}
        self.mc_labels = mc_labels
        prompts: List[Union[None, str]] = [None]
        if prompt_variations is not None:
            if prompt_variations == 1:
                prompts = [None, "inst", "inst+1", "inst+2", "inst+3", "inst+4", "inst+5"]
            elif prompt_variations == 2:
                prompts = ["inst+5"]
            else:
                raise ValueError(f"Unknown prompt variations: {prompt_variations}")
            # Need to grab the dev set for the few-shot prompts
            for name in dataset_names:
                dev_set = load_hf_dataset(dataset_path, name, "dev")
                self.dev_set[name] = dev_set
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_names,
            split=split,
            prompts=prompts,
            metric_type=metric_type,
        )

    def doc_to_text(self, doc):
        def format_example(doc, keys):
            question_prefix = ""
            if not self.mc_labels:
                question_prefix = "Question: "  # To make context more clear
            question = question_prefix + doc["question"].strip()
            choices = ""
            if self.mc_labels:
                choices = "".join([f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])])
            prompt = f"{question}\n{choices}Answer:"
            return prompt

        keys = ["A", "B", "C", "D"]
        output_text = format_example(doc, keys)

        if self.current_prompt is not None:
            prefix = ""
            if "inst" in self.current_prompt:
                subject = doc.get("subject").replace("_", " ")
                prefix = f"The following are multiple choice questions (with answers) about {subject}:\n\n"
            num_shots = re.findall("\\+(\\d+)", self.current_prompt)
            if num_shots:
                dev_set = self.dev_set.get(doc.get("subject"), [])
                num_shots_int = int(num_shots[0])
                for idx, dev_doc in enumerate(dev_set):
                    if idx >= num_shots_int:
                        break
                    if self.mc_labels:
                        answer = keys[dev_doc["answer"]]
                    else:
                        answer = dev_doc["choices"][dev_doc["answer"]]
                    prefix += format_example(dev_doc, keys) + " " + answer + "\n\n"
            output_text = prefix + output_text
        return output_text

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        if self.mc_labels:
            choices = [" A", " B", " C", " D"]
        else:
            choices = [" " + choice for choice in doc["choices"]]
        if self.metric_type in ["ce_loss", "bpb"]:
            # Only need correct answer for these metrics
            return [choices[doc["answer"]]]
        else:
            return choices

    def doc_to_label(self, doc):
        if self.metric_type in ["ce_loss", "bpb"]:
            # Only the correct answer is provided for these metrics
            return 0
        return doc["answer"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class TriviaQACELoss(ICLMultiChoiceTaskDataset):
    """Sample TriviaQA entity with some fields suppressed. For CE Loss we only consider the "value"
    field as the answer to score.

    {
        'question': 'Which Lloyd Webber musical premiered in the US on 10th December 1993?',
        'question_id': 'tc_33',
        'answer': {
            'aliases': ['Sunset Blvd', ...],
            'normalized_aliases': ['sunset boulevard', ...],
            'normalized_value': 'sunset boulevard',
            'value': 'Sunset Boulevard'
        }
    }
    """

    metric_type = "ce_loss"

    def __init__(
        self,
        tokenizer,
        dataset_path="trivia_qa",
        dataset_name="rc.wikipedia.nocontext",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return "\nQuestion: " + doc["question"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        return [" " + doc["answer"]["value"]]

    def doc_to_label(self, doc):
        return 0

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class NaturalQuestionsCELoss(ICLMultiChoiceTaskDataset):
    """Sample NaturalQuestions entity. For CE Loss we only consider the first answer entry to score.

    {
        'question': 'when was the last time anyone was on the moon',
        'answer': ['14 December 1972 UTC', 'December 1972']
    }
    """

    metric_type = "ce_loss"

    def __init__(
        self,
        tokenizer,
        dataset_path="nq_open",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return "\nQuestion: " + doc["question"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        return [" " + doc["answer"][0]]

    def doc_to_label(self, doc):
        return 0

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class OEEvalTask(ICLMultiChoiceTaskDataset):
    """Generic class for OE evaluation tasks"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: Union[str, Sequence[str], None] = None,
        model_ctx_len: int = 2048,
        split=None,
        metric_type=None,
        prompts=[None],  # List of prompt variants to use
    ):
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len
        self.log_instances = 0  # Set to > 0 to log the first few instances as a sanity check

        self.samples: List[Dict[str, Any]] = []
        dataset_names: Sequence[Optional[str]]
        if isinstance(dataset_name, str) or dataset_name is None:
            dataset_names = [dataset_name]
        else:
            dataset_names = dataset_name

        requests_list = []
        configs = []
        for ds_name in dataset_names:
            config, requests = load_oe_eval_requests(self.dataset_path, ds_name, split)
            requests_list.append(requests)
            configs.append(config)
        if metric_type is not None:
            self.metric_type = metric_type
        else:
            # Use metric type from associated task config
            for config in configs:
                if config is not None:
                    metric_type_raw = config["task_config"].get("primary_metric")
                    if metric_type_raw is not None:
                        # acc, len_norm, pmi_dc
                        metric_type = METRIC_FROM_OE_EVAL[metric_type_raw]
                        if self.metric_type is not None and self.metric_type != metric_type:
                            raise ValueError(f"Conflicting metric types: {self.metric_type} and {metric_type}")
                        self.metric_type = metric_type
        self.dataset = requests_list

        # prep examples
        self.prep_examples()

    def prep_examples(self):
        current_doc_id_offset = 0
        max_doc_id = 0
        for requests in self.dataset:
            current_doc_id_offset += max_doc_id
            max_doc_id = 0  # Max doc id seen in this dataset
            for request in requests:
                doc = request["doc"]
                doc_id = request["doc_id"]
                if doc_id >= 1000000:
                    # Hacky implementation of unconditional requests in oe-eval
                    # Not supported here for now
                    continue
                if doc_id > max_doc_id:
                    max_doc_id = doc_id
                assert (
                    request["request_type"] == "loglikelihood"
                ), f"Unsupported request type: {request['request_type']}"

                # from EAI harness
                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                request_dict = request["request"]
                continuation_str = request_dict["continuation"]
                label_id = request["label"]
                cont_id = request["idx"]
                if self.metric_type in ["ce_loss", "bpb"]:
                    if label_id != cont_id:
                        # Skip non-target continuations for ce_loss and bpb
                        continue
                    else:
                        # Treat as instance with just one continuation
                        cont_id = 0
                        label_id = 0
                doc_text = request_dict["context"]
                ctx = self.token_encode(doc_text)
                dc = self.token_encode(self.doc_to_domain_conditional(doc))
                if self.log_instances > 0:
                    self.log_instances -= 1
                    ds_name = self.dataset_name
                    if isinstance(ds_name, list):
                        ds_name = ds_name[0]
                    log.info(
                        f"Sample doc from ({self.dataset_path}, {ds_name}):"
                        + f"\ndoc_text: {doc_text}\ncontinuation: {continuation_str}"
                    )
                cont_str_len = len(continuation_str) - 1  # continuation contain leading blank
                cont_byte_len = len(continuation_str[1:].encode("utf-8"))
                continuation = self.token_encode(continuation_str)

                # query, remove last token from continuation, truncate from left is longer than model ctx length
                query = ctx + continuation[:-1]
                query = query[-self.model_ctx_len :]
                # this will be different from len(ctx) when truncated by model_ctx_len
                actual_ctx_len = len(query) - len(continuation) + 1

                # get domain conditional query
                # we don't expect this to be longer than self.model_ctx_len and it won't make sense to truncate from left
                dc_query = dc + continuation[:-1]

                # form a sample
                self.samples.append(
                    {
                        "doc_id": doc_id + current_doc_id_offset,
                        "cont_id": cont_id,
                        "ctx": ctx,
                        "continuation": continuation,
                        "ctx_len": actual_ctx_len,
                        "dc_len": len(dc),
                        "cont_len": len(
                            continuation
                        ),  # even if query has last token removed, LM will output same cont len
                        "cont_str_len": cont_str_len,
                        "cont_byte_len": cont_byte_len,
                        "query": query,  # remove last token from continuation
                        "dc_query": dc_query,
                        "label_id": label_id,
                    }
                )

    def doc_to_text(self, doc) -> str:
        raise NotImplementedError

    def doc_to_continuations(self, doc) -> List[str]:
        raise NotImplementedError

    def doc_to_label(self, doc) -> int:
        raise NotImplementedError


TG_path = "./TG-LLaMA/OLMoData/TG"
TXLTREE_path = "./TG-LLaMA/OLMoData/txltree"
label_to_task_map = {
    "tg_approx_sent": (TGPerplexityApproximationDataset, {"dataset_path": TG_path, "dataset_name": "newppl.json", "metric_type": "sent"}),
    "tg_approx_sent_testor": (TGPerplexityApproximationDataset, {"dataset_path": TG_path, "dataset_name": "smallppl.json", "metric_type": "sent"}),
    "txl_approx_sent": (TGPerplexityApproximationDataset, {"dataset_path": TXLTREE_path, "dataset_name": "newppl.json", "metric_type": "sent"}),
    "txl_approx_sent_testor": (TGPerplexityApproximationDataset, {"dataset_path": TXLTREE_path, "dataset_name": "smallppl.json", "metric_type": "sent"}),
    "tg_approx_doc": (TGPerplexityApproximationDataset, {"dataset_path": TG_path, "dataset_name": "docppl.json", "metric_type": "doc"}),
    "tg_approx_doc_testor": (TGPerplexityApproximationDataset, {"dataset_path": TG_path, "dataset_name": "docsmallppl.json", "metric_type": "doc"}),
    "txl_approx_doc": (TGPerplexityApproximationDataset, {"dataset_path": TXLTREE_path, "dataset_name": "docppl.json", "metric_type": "doc"}),
    "txl_approx_doc_testor": (TGPerplexityApproximationDataset, {"dataset_path": TXLTREE_path, "dataset_name": "docsmallppl.json", "metric_type": "doc"}),
    "syntactic_generalization": (SGDataset, {"dataset_path": "./evaluation/SG/tokenized"}),
    "piqa": PIQA,
    "hellaswag": HellaSwag,
    "winogrande": WinoGrande,
    "openbook_qa": OpenBookQA,
    "boolq": BoolQ,
    "sciq": SciQ,
    "arc_easy": ArcEasy,
    "arc_easy_ppl": ArcEasyCELoss,
    "arc_challenge": ArcChallenge,
    "basic_arithmetic": BasicArithmetic,
    "copa": COPA,
    "rte": RTE,
    "commitment_bank": CommitmentBank,
    "mrpc": MRPC,
    "sst2": SST2,
    "commonsense_qa": CommonsenseQA,
    "social_iqa": SocialIQa,
    "trivia_qa_wiki_ppl": TriviaQACELoss,
    "natural_qs_open_ppl": NaturalQuestionsCELoss,
    "mmlu_stem_test": (MMLU, {"dataset_name": "stem", "split": "test"}),
    "mmlu_humanities_test": (MMLU, {"dataset_name": "humanities", "split": "test"}),
    "mmlu_social_sciences_test": (MMLU, {"dataset_name": "social_sciences", "split": "test"}),
    "mmlu_other_test": (MMLU, {"dataset_name": "other", "split": "test"}),
    "mmlu_stem": (MMLU, {"dataset_name": "stem"}),
    "mmlu_humanities": (MMLU, {"dataset_name": "humanities"}),
    "mmlu_social_sciences": (MMLU, {"dataset_name": "social_sciences"}),
    "mmlu_other": (MMLU, {"dataset_name": "other"}),
    "mmlu_stem_bpb": (MMLU, {"dataset_name": "stem", "metric_type": "bpb"}),
    "mmlu_humanities_bpb": (MMLU, {"dataset_name": "humanities", "metric_type": "bpb"}),
    "mmlu_social_sciences_bpb": (MMLU, {"dataset_name": "social_sciences", "metric_type": "bpb"}),
    "mmlu_other_bpb": (MMLU, {"dataset_name": "other", "metric_type": "bpb"}),
    "mmlu_stem_var": (MMLU, {"dataset_name": "stem", "prompt_variations": 1}),
    "mmlu_humanities_var": (MMLU, {"dataset_name": "humanities", "prompt_variations": 1}),
    "mmlu_social_sciences_var": (MMLU, {"dataset_name": "social_sciences", "prompt_variations": 1}),
    "mmlu_other_var": (MMLU, {"dataset_name": "other", "prompt_variations": 1}),
    "mmlu_stem_var_bpb": (MMLU, {"dataset_name": "stem", "prompt_variations": 1, "metric_type": "bpb"}),
    "mmlu_humanities_var_bpb": (
        MMLU,
        {"dataset_name": "humanities", "prompt_variations": 1, "metric_type": "bpb"},
    ),
    "mmlu_social_sciences_var_bpb": (
        MMLU,
        {"dataset_name": "social_sciences", "prompt_variations": 1, "metric_type": "bpb"},
    ),
    "mmlu_other_var_bpb": (MMLU, {"dataset_name": "other", "prompt_variations": 1, "metric_type": "bpb"}),
    "mmlu_stem_mc_5shot": (MMLU, {"dataset_name": "stem", "prompt_variations": 2, "mc_labels": True}),
    "mmlu_humanities_mc_5shot": (MMLU, {"dataset_name": "humanities", "prompt_variations": 2, "mc_labels": True}),
    "mmlu_social_sciences_mc_5shot": (
        MMLU,
        {"dataset_name": "social_sciences", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_other_mc_5shot": (MMLU, {"dataset_name": "other", "prompt_variations": 2, "mc_labels": True}),
    "mmlu_stem_mc_5shot_test": (
        MMLU,
        {"dataset_name": "stem", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_humanities_mc_5shot_test": (
        MMLU,
        {"dataset_name": "humanities", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_social_sciences_mc_5shot_test": (
        MMLU,
        {"dataset_name": "social_sciences", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_other_mc_5shot_test": (
        MMLU,
        {"dataset_name": "other", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    # Paste in all oe-eval tasks from output of scripts/list_evals_from_oe_eval.py
    "arc_challenge_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "arc_challenge_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "arc_challenge_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "rc_0shot", "metric_type": "len_norm"},
    ),
    "arc_challenge_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "arc_challenge_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "rc_5shot", "metric_type": "len_norm"},
    ),
    "arc_challenge_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "arc_easy_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "rc_0shot", "metric_type": "acc"},
    ),
    "arc_easy_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "arc_easy_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "rc_5shot", "metric_type": "acc"},
    ),
    "arc_easy_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "boolq_mc_5shot": (OEEvalTask, {"dataset_path": "boolq", "dataset_name": "mc_5shot", "metric_type": "acc"}),
    "boolq_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "boolq_rc_0shot": (OEEvalTask, {"dataset_path": "boolq", "dataset_name": "rc_0shot", "metric_type": "acc"}),
    "boolq_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "boolq_rc_5shot": (OEEvalTask, {"dataset_path": "boolq", "dataset_name": "rc_5shot", "metric_type": "acc"}),
    "boolq_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "copa_rc_0shot": (OEEvalTask, {"dataset_path": "copa", "dataset_name": "rc_0shot", "metric_type": "acc"}),
    "copa_rc_0shot_bpb": (OEEvalTask, {"dataset_path": "copa", "dataset_name": "rc_0shot", "metric_type": "bpb"}),
    "copycolors_10way": (
        OEEvalTask,
        {"dataset_path": "copycolors", "dataset_name": "10way", "metric_type": "acc"},
    ),
    "copycolors_10way_bpb": (
        OEEvalTask,
        {"dataset_path": "copycolors", "dataset_name": "10way", "metric_type": "bpb"},
    ),
    "copycolors_xl_10way": (
        OEEvalTask,
        {"dataset_path": "copycolors", "dataset_name": "xl_10way", "metric_type": "acc"},
    ),
    "copycolors_xl_10way_bpb": (
        OEEvalTask,
        {"dataset_path": "copycolors", "dataset_name": "xl_10way", "metric_type": "bpb"},
    ),
    "csqa_mc_5shot": (OEEvalTask, {"dataset_path": "csqa", "dataset_name": "mc_5shot", "metric_type": "acc"}),
    "csqa_mc_5shot_bpb": (OEEvalTask, {"dataset_path": "csqa", "dataset_name": "mc_5shot", "metric_type": "bpb"}),
    "csqa_rc_0shot": (OEEvalTask, {"dataset_path": "csqa", "dataset_name": "rc_0shot", "metric_type": "len_norm"}),
    "csqa_rc_0shot_bpb": (OEEvalTask, {"dataset_path": "csqa", "dataset_name": "rc_0shot", "metric_type": "bpb"}),
    "csqa_rc_5shot": (OEEvalTask, {"dataset_path": "csqa", "dataset_name": "rc_5shot", "metric_type": "len_norm"}),
    "csqa_rc_5shot_bpb": (OEEvalTask, {"dataset_path": "csqa", "dataset_name": "rc_5shot", "metric_type": "bpb"}),
    "hellaswag_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "hellaswag_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "hellaswag_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "rc_0shot", "metric_type": "len_norm"},
    ),
    "hellaswag_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "hellaswag_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "rc_5shot", "metric_type": "len_norm"},
    ),
    "hellaswag_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "openbookqa_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "rc_0shot", "metric_type": "len_norm"},
    ),
    "openbookqa_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "openbookqa_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "rc_5shot", "metric_type": "len_norm"},
    ),
    "openbookqa_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "piqa_mc_5shot": (OEEvalTask, {"dataset_path": "piqa", "dataset_name": "mc_5shot", "metric_type": "acc"}),
    "piqa_mc_5shot_bpb": (OEEvalTask, {"dataset_path": "piqa", "dataset_name": "mc_5shot", "metric_type": "bpb"}),
    "piqa_rc_0shot": (OEEvalTask, {"dataset_path": "piqa", "dataset_name": "rc_0shot", "metric_type": "len_norm"}),
    "piqa_rc_0shot_bpb": (OEEvalTask, {"dataset_path": "piqa", "dataset_name": "rc_0shot", "metric_type": "bpb"}),
    "piqa_rc_5shot": (OEEvalTask, {"dataset_path": "piqa", "dataset_name": "rc_5shot", "metric_type": "len_norm"}),
    "piqa_rc_5shot_bpb": (OEEvalTask, {"dataset_path": "piqa", "dataset_name": "rc_5shot", "metric_type": "bpb"}),
    "sciq_rc_0shot": (OEEvalTask, {"dataset_path": "sciq", "dataset_name": "rc_0shot", "metric_type": "acc"}),
    "sciq_rc_0shot_bpb": (OEEvalTask, {"dataset_path": "sciq", "dataset_name": "rc_0shot", "metric_type": "bpb"}),
    "socialiqa_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "socialiqa_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "socialiqa_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "rc_0shot", "metric_type": "len_norm"},
    ),
    "socialiqa_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "socialiqa_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "rc_5shot", "metric_type": "len_norm"},
    ),
    "socialiqa_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "winogrande_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "winogrande_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "winogrande_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "rc_0shot", "metric_type": "acc"},
    ),
    "winogrande_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "winogrande_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "rc_5shot", "metric_type": "acc"},
    ),
    "winogrande_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
}

# This standardizes the metrics we should eval for the ladder.
# Train and test sets are added when applicable.
# No subsampling happens in these sets.
label_to_task_map_new = {
    "arc_challenge_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "train_rc_5shot", "metric_type": "len_norm"},
    ),
    "arc_challenge_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "arc_challenge_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "arc_challenge_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "arc_challenge_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "arc_challenge_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "arc_challenge_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "arc_challenge_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "arc_challenge_test_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "test_rc_5shot", "metric_type": "len_norm"},
    ),
    "arc_challenge_test_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "test_rc_5shot", "metric_type": "bpb"},
    ),
    "arc_challenge_test_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "test_mc_5shot", "metric_type": "acc"},
    ),
    "arc_challenge_test_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "test_mc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "train_rc_5shot", "metric_type": "len_norm"},
    ),  # this used to be acc
    "arc_easy_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "arc_easy_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "arc_easy_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "arc_easy_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_test_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "test_rc_5shot", "metric_type": "len_norm"},
    ),
    "arc_easy_test_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "test_rc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_test_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "test_mc_5shot", "metric_type": "acc"},
    ),
    "arc_easy_test_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "test_mc_5shot", "metric_type": "bpb"},
    ),
    "boolq_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "train_rc_5shot", "metric_type": "acc"},
    ),  # kept acc here, since len_norm can bias towards "yes"
    "boolq_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "boolq_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "boolq_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "boolq_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "val_rc_5shot", "metric_type": "acc"},
    ),
    "boolq_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "boolq_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "boolq_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "csqa_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "train_rc_5shot", "metric_type": "len_norm"},
    ),
    "csqa_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "csqa_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "csqa_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "csqa_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "csqa_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "csqa_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "csqa_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "hellaswag_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "train_rc_5shot", "metric_type": "len_norm"},
    ),
    "hellaswag_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "hellaswag_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "hellaswag_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "hellaswag_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "hellaswag_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "hellaswag_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "hellaswag_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "train_rc_5shot", "metric_type": "len_norm"},
    ),
    "openbookqa_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "openbookqa_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "openbookqa_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "openbookqa_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_test_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "test_rc_5shot", "metric_type": "len_norm"},
    ),
    "openbookqa_test_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "test_rc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_test_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "test_mc_5shot", "metric_type": "acc"},
    ),
    "openbookqa_test_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "test_mc_5shot", "metric_type": "bpb"},
    ),
    "piqa_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "train_rc_5shot", "metric_type": "len_norm"},
    ),
    "piqa_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "piqa_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "piqa_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "piqa_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "piqa_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "piqa_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "piqa_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "socialiqa_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "train_rc_5shot", "metric_type": "len_norm"},
    ),
    "socialiqa_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "socialiqa_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "socialiqa_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "socialiqa_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "socialiqa_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "socialiqa_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "socialiqa_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "winogrande_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "train_rc_5shot", "metric_type": "len_norm"},
    ),  # this used to be acc
    "winogrande_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "winogrande_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "winogrande_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "winogrande_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "winogrande_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "winogrande_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "winogrande_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "mmlu_stem_val_rc_var": (MMLU, {"dataset_name": "stem", "prompt_variations": 1}),
    "mmlu_stem_val_rc_var_bpb": (MMLU, {"dataset_name": "stem", "prompt_variations": 1, "metric_type": "bpb"}),
    "mmlu_stem_val_rc_5shot": (MMLU, {"dataset_name": "stem", "prompt_variations": 2}),
    "mmlu_stem_val_rc_5shot_bpb": (MMLU, {"dataset_name": "stem", "prompt_variations": 2, "metric_type": "bpb"}),
    "mmlu_stem_val_mc_5shot": (MMLU, {"dataset_name": "stem", "prompt_variations": 2, "mc_labels": True}),
    "mmlu_stem_val_mc_5shot_bpb": (
        MMLU,
        {"dataset_name": "stem", "prompt_variations": 2, "mc_labels": True, "metric_type": "bpb"},
    ),
    "mmlu_stem_test_rc_var": (MMLU, {"dataset_name": "stem", "split": "test", "prompt_variations": 1}),
    "mmlu_stem_test_rc_var_bpb": (
        MMLU,
        {"dataset_name": "stem", "split": "test", "prompt_variations": 1, "metric_type": "bpb"},
    ),
    "mmlu_stem_test_rc_5shot": (MMLU, {"dataset_name": "stem", "split": "test", "prompt_variations": 2}),
    "mmlu_stem_test_rc_5shot_bpb": (
        MMLU,
        {"dataset_name": "stem", "split": "test", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_stem_test_mc_5shot": (
        MMLU,
        {"dataset_name": "stem", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_stem_test_mc_5shot_bpb": (
        MMLU,
        {"dataset_name": "stem", "split": "test", "prompt_variations": 2, "mc_labels": True, "metric_type": "bpb"},
    ),
    "mmlu_humanities_val_rc_var": (MMLU, {"dataset_name": "humanities", "prompt_variations": 1}),
    "mmlu_humanities_val_rc_var_bpb": (
        MMLU,
        {"dataset_name": "humanities", "prompt_variations": 1, "metric_type": "bpb"},
    ),
    "mmlu_humanities_val_rc_5shot": (MMLU, {"dataset_name": "humanities", "prompt_variations": 2}),
    "mmlu_humanities_val_rc_5shot_bpb": (
        MMLU,
        {"dataset_name": "humanities", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_humanities_val_mc_5shot": (
        MMLU,
        {"dataset_name": "humanities", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_humanities_val_mc_5shot_bpb": (
        MMLU,
        {"dataset_name": "humanities", "prompt_variations": 2, "mc_labels": True, "metric_type": "bpb"},
    ),
    "mmlu_humanities_test_rc_var": (MMLU, {"dataset_name": "humanities", "split": "test", "prompt_variations": 1}),
    "mmlu_humanities_test_rc_var_bpb": (
        MMLU,
        {"dataset_name": "humanities", "split": "test", "prompt_variations": 1, "metric_type": "bpb"},
    ),
    "mmlu_humanities_test_rc_5shot": (
        MMLU,
        {"dataset_name": "humanities", "split": "test", "prompt_variations": 2},
    ),
    "mmlu_humanities_test_rc_5shot_bpb": (
        MMLU,
        {"dataset_name": "humanities", "split": "test", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_humanities_test_mc_5shot": (
        MMLU,
        {"dataset_name": "humanities", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_humanities_test_mc_5shot_bpb": (
        MMLU,
        {
            "dataset_name": "humanities",
            "split": "test",
            "prompt_variations": 2,
            "mc_labels": True,
            "metric_type": "bpb",
        },
    ),
    "mmlu_social_sciences_val_rc_var": (MMLU, {"dataset_name": "social_sciences", "prompt_variations": 1}),
    "mmlu_social_sciences_val_rc_var_bpb": (
        MMLU,
        {"dataset_name": "social_sciences", "prompt_variations": 1, "metric_type": "bpb"},
    ),
    "mmlu_social_sciences_val_rc_5shot": (MMLU, {"dataset_name": "social_sciences", "prompt_variations": 2}),
    "mmlu_social_sciences_val_rc_5shot_bpb": (
        MMLU,
        {"dataset_name": "social_sciences", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_social_sciences_val_mc_5shot": (
        MMLU,
        {"dataset_name": "social_sciences", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_social_sciences_val_mc_5shot_bpb": (
        MMLU,
        {"dataset_name": "social_sciences", "prompt_variations": 2, "mc_labels": True, "metric_type": "bpb"},
    ),
    "mmlu_social_sciences_test_rc_var": (
        MMLU,
        {"dataset_name": "social_sciences", "split": "test", "prompt_variations": 1},
    ),
    "mmlu_social_sciences_test_rc_var_bpb": (
        MMLU,
        {"dataset_name": "social_sciences", "split": "test", "prompt_variations": 1, "metric_type": "bpb"},
    ),
    "mmlu_social_sciences_test_rc_5shot": (
        MMLU,
        {"dataset_name": "social_sciences", "split": "test", "prompt_variations": 2},
    ),
    "mmlu_social_sciences_test_rc_5shot_bpb": (
        MMLU,
        {"dataset_name": "social_sciences", "split": "test", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_social_sciences_test_mc_5shot": (
        MMLU,
        {"dataset_name": "social_sciences", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_social_sciences_test_mc_5shot_bpb": (
        MMLU,
        {
            "dataset_name": "social_sciences",
            "split": "test",
            "prompt_variations": 2,
            "mc_labels": True,
            "metric_type": "bpb",
        },
    ),
    "mmlu_other_val_rc_var": (MMLU, {"dataset_name": "other", "prompt_variations": 1}),
    "mmlu_other_val_rc_var_bpb": (MMLU, {"dataset_name": "other", "prompt_variations": 1, "metric_type": "bpb"}),
    "mmlu_other_val_rc_5shot": (MMLU, {"dataset_name": "other", "prompt_variations": 2}),
    "mmlu_other_val_rc_5shot_bpb": (MMLU, {"dataset_name": "other", "prompt_variations": 2, "metric_type": "bpb"}),
    "mmlu_other_val_mc_5shot": (MMLU, {"dataset_name": "other", "prompt_variations": 2, "mc_labels": True}),
    "mmlu_other_val_mc_5shot_bpb": (
        MMLU,
        {"dataset_name": "other", "prompt_variations": 2, "mc_labels": True, "metric_type": "bpb"},
    ),
    "mmlu_other_test_rc_var": (MMLU, {"dataset_name": "other", "split": "test", "prompt_variations": 1}),
    "mmlu_other_test_rc_var_bpb": (
        MMLU,
        {"dataset_name": "other", "split": "test", "prompt_variations": 1, "metric_type": "bpb"},
    ),
    "mmlu_other_test_rc_5shot": (MMLU, {"dataset_name": "other", "split": "test", "prompt_variations": 2}),
    "mmlu_other_test_rc_5shot_bpb": (
        MMLU,
        {"dataset_name": "other", "split": "test", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_other_test_mc_5shot": (
        MMLU,
        {"dataset_name": "other", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_other_test_mc_5shot_bpb": (
        MMLU,
        {
            "dataset_name": "other",
            "split": "test",
            "prompt_variations": 2,
            "mc_labels": True,
            "metric_type": "bpb",
        },
    ),
}

label_to_task_map = {
    **label_to_task_map,
    **label_to_task_map_new,
}
