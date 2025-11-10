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
import evaluate

from olmo.util import load_hf_dataset, load_oe_eval_requests

from ..data.tg_mask import TG_attention_bias, SentencepieceVocab
from ..tokenizer import Tokenizer
from ..data.util import encode_TG_string, convert_TG_format
from ..data.collator import DataCollator
from ..config import PaddingDirection

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

    def reset(self):
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
    

class ICLMultiChoiceTaskDataset(metaclass=abc.ABCMeta):
    """Only supports zero-shot for now."""

    metric_type: str

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: Union[str, Sequence[str], None] = None,
        model_ctx_len: int = 2048,
        split="test",
        metric_type=None,  # Override default metric type
        prompts=[None],  # List of prompt variants to use
        local_datasets=True,
        transformer_grammar_type="",
        generate_TG_attention_bias=None,
        vocab_path=None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len
        if transformer_grammar_type == "pause1/2":
            self.model_ctx_len //= 2
        self.prompts = prompts
        self.current_prompt = None
        self.split = split
        if metric_type is not None:
            self.metric_type = metric_type
        self.log_instances = 5  # Set to > 0 to log the first few instances as a sanity check
        self.generate_TG_attention_bias = generate_TG_attention_bias
        self.transformer_grammar_type = transformer_grammar_type
        print(f"vocab path is {vocab_path}")
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)

        self.samples: List[Dict[str, Any]] = []
        dataset_names: Sequence[Optional[str]]
        if isinstance(dataset_name, str) or dataset_name is None:
            dataset_names = [dataset_name]
        else:
            dataset_names = dataset_name

        if not local_datasets:
            dataset_list = []
            for ds_name in dataset_names:
                dataset = load_hf_dataset(self.dataset_path, ds_name, split)
                dataset_list.append(dataset)
            self.dataset = datasets.concatenate_datasets(dataset_list)
        else:
            self.load_local_datasets()

        # prep examples
        self.prep_examples()

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
    
    def convert_grammar_input(self, input_ids) -> List[int]:
        if not isinstance(input_ids, np.ndarray):
            input_ids = np.array(input_ids)
        if self.transformer_grammar_type == "terminal":
            input_ids = self.vocab.convert_treenpy_to_terminal(input_ids)
        elif self.transformer_grammar_type[:4] == "tree":
            input_ids = self.vocab.convert_TGnpy_to_tree(input_ids)
        return input_ids.tolist()

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
                dc = self.convert_grammar_input(dc)

                for cont_id, continuation_str in enumerate(continuations):
                    cont_str_len = len(continuation_str) - 1  # continuation contain leading blank
                    cont_byte_len = len(continuation_str[1:].encode("utf-8"))
                    continuation = self.token_encode(continuation_str)

                    # query, remove last token from continuation, truncate from left is longer than model ctx length
                    # when train, should keep last token
                    if self.split=="train":
                        query = ctx + continuation
                    else:
                        query = ctx + continuation[:-1]
                        
                    query = query[-self.model_ctx_len :]
                    query = self.convert_grammar_input(query)
                    continuation = self.convert_grammar_input(continuation)
                    # this will be different from len(ctx) when truncated by model_ctx_len
                    actual_ctx_len = len(query) - len(continuation) + 1
                    
                    # get domain conditional query
                    # we don't expect this to be longer than self.model_ctx_len and it won't make sense to truncate from left
                    if self.split=="train":
                        dc_query = dc + continuation
                    else:
                        dc_query = dc + continuation[:-1]

                    # form a sample
                    self.samples.append(
                        {
                            "doc_id": doc_id,
                            "cont_id": cont_id,
                            # "ctx": ctx,
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
                if self.log_instances > 0:
                    self.log_instances -= 1
                    ds_name = self.dataset_name
                    if isinstance(ds_name, list):
                        ds_name = ds_name[0]
                    log.info(
                        f"Sample doc from ({self.dataset_path}, {ds_name}, {self.current_prompt}):"
                        + f"\ndoc_text: {doc_text}\ncontinuations: {continuations}\n" +
                        f"input_ids is {self.token_decode(query)}"
                    )

                doc_id += 1

    def pad_tokens_until_max(self, tokens, max_len=2048, max_model_len=None):
        """truncate from left if len(tokens) > model_ctx_len, max_len is not considered then
        queries are already truncated at max length of model_ctx_len
        this acts as additional check for all types of sequences in the batch
        """
        model_ctx_len = max_model_len or self.model_ctx_len
        if len(tokens) > model_ctx_len:
            return tokens[-model_ctx_len :]
        else:
            # pad to max_len, but check again if this padding exceeded model_ctx_len
            # this time truncate from right side of the sequence because additional padding caused len(tokens) > model_ctx_len
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))

            if len(tokens) > model_ctx_len:
                tokens = tokens[: model_ctx_len]

            return tokens

    def collate_fn(self, data):
        # pad to max length
        # 'ctx', 'continuation', 'query' can all have variable length
        max_ctx_len = 0
        max_cont_len = 0
        max_query_len = 0
        max_dc_query_len = 0

        for sample in data:
            # if len(sample["ctx"]) > max_ctx_len:
            #     max_ctx_len = len(sample["ctx"])

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
        all_attention_bias = []
        all_label_mask = []
        if self.transformer_grammar_type == "pause1/2":
            max_query_len *= 2

        # pad according to max_lengths
        for sample in data:
            input_ids = sample["query"]
            if self.transformer_grammar_type[:12] == "tree_shuffle":
                if not isinstance(input_ids, np.ndarray):
                    input_ids = np.array(input_ids)
                input_ids = self.vocab.random_shuffle_tree(input_ids)
                input_ids = input_ids.tolist()
            elif self.transformer_grammar_type == "pause1/2":
                paused_input = input_ids + input_ids
                paused_input[::2] = input_ids
                paused_input[1::2] = [50260] * len(input_ids)  # Special token
                input_ids = paused_input
                sample["ctx_len"] *= 2
                sample["dc_len"] *= 2
                sample["cont_len"] *= 2
            input_ids = torch.LongTensor(self.pad_tokens_until_max(input_ids, max_len=max_query_len, max_model_len=None if self.transformer_grammar_type!="pause1/2" else self.model_ctx_len * 2))
            queries.append(input_ids)

            label_mask = None
            if self.generate_TG_attention_bias is not None:
                attention_bias, label_mask = self.generate_TG_attention_bias(input_ids)
                while len(attention_bias.shape) < 3:
                    attention_bias = attention_bias.unsqueeze(0)
                all_attention_bias.append(attention_bias)
            if self.transformer_grammar_type[-4:] == "mask":
                cur_label_mask = self.vocab.get_non_terminal_mask(input_ids)
                if label_mask is not None:
                    label_mask = torch.bitwise_and(label_mask, torch.tensor(cur_label_mask))
                else:
                    label_mask = cur_label_mask
            if label_mask is not None:
                all_label_mask.append(label_mask)
            
            doc_ids.append(sample["doc_id"])
            cont_ids.append(sample["cont_id"])

            # ctxs.append(torch.LongTensor(self.pad_tokens_until_max(sample["ctx"], max_len=max_ctx_len)))
            continuations.append(
                torch.LongTensor(self.pad_tokens_until_max(sample["continuation"], max_len=max_cont_len))
            )

            ctx_lens.append(sample["ctx_len"])
            dc_lens.append(sample["dc_len"])
            cont_lens.append(sample["cont_len"])
            cont_str_lens.append(sample["cont_str_len"])
            cont_byte_lens.append(sample["cont_byte_len"])

            dc_queries.append(
                torch.LongTensor(self.pad_tokens_until_max(sample["dc_query"], max_len=max_dc_query_len))
            )
            label_ids.append(sample["label_id"])

        batch = {
            "doc_id": torch.LongTensor(doc_ids),
            "cont_id": torch.LongTensor(cont_ids),
            # "ctx": torch.stack(ctxs),
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
        if all_attention_bias:
            batch["attention_bias"] = torch.stack(all_attention_bias)
        if all_label_mask:
            batch["label_mask"] = torch.stack(all_label_mask)

        return batch

    def token_encode(self, string: str) -> List[int]:
        ids = encode_TG_string(self.tokenizer, string, string_with_POS_tags=False)
        ids = self.vocab.convert_treenpy_to_TG(ids)
        return ids.tolist()

    def token_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

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


class XsumDataset(metaclass=abc.ABCMeta):
    def __init__(self,
        tokenizer: Tokenizer,
        dataset_path: str,
        model_ctx_len: int = 2048,
        split="test",
        metric_type="sent",
        generate_TG_attention_bias: Optional[Callable] = None,
        transformer_grammar_type:str = "",
        vocab_path: str = None):

        self.tokenizer = tokenizer
        self.transformer_grammar_type = transformer_grammar_type
        self.collator = DataCollator(pad_direction=PaddingDirection.left, pad_token_id=self.tokenizer.pad_token_id, 
                                        generate_attention_mask=True, shuffle_tree=transformer_grammar_type)
        self.MAX_SUMMARY_LENGTH = 150
        self.collator.vocab = self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.model_ctx_len = model_ctx_len
        self.generate_TG_attention_bias = generate_TG_attention_bias
        self.prompts = "<|SEP|> (S (VP Summarize (NP the above article NP) (PP in (NP 1 sentence NP) PP) VP) . S) <|SEP|>"
        self.prompts_tokens = encode_TG_string(self.tokenizer, self.prompts, string_with_POS_tags=False)
        self.prompts_TG_tokens = self.vocab.convert_treenpy_to_TG(self.prompts_tokens)

        if transformer_grammar_type=="pause1/2":
            self.model_ctx_len //= 2
            self.prompts_TG_tokens = self.vocab.convert_treenpy_to_terminal(self.prompts_TG_tokens)
        passages = []
        gold_summary = []
        with open(os.path.join(dataset_path, f"gold_{split}_summary.jsonl"), 'r') as file:
            for line in file:
                summary = json.loads(line.strip())
                gold_summary.append(summary)

        with open(os.path.join(dataset_path, f"xsum_{split}.txt"), 'r') as file:
            for line in file:
                passages.append(line.strip())

        if split=="train":
            train_summary = []
            with open(os.path.join(dataset_path, f"save_ids.json"), 'r') as file:
                train_ids = json.load(file)
            with open(os.path.join(dataset_path, "xsum_train_summary.txt"), 'r') as file:
                for line in file:
                    train_summary.append(line.strip())
            train_ids = set(train_ids)
            self.passages = []
            self.train_summary = []
            self.gold_summary = []
            for passage, summary, gold in zip(passages, train_summary, gold_summary):
                if gold["id"] in train_ids:
                    self.passages.append(passage)
                    self.train_summary.append(summary)
                    self.gold_summary.append(gold["summary"])
        else:
            self.passages = passages
            self.gold_summary = [line["summary"] for line in gold_summary]
            self.train_summary = None


    def __getitem__(self, index: int) -> Dict[str, Any]:
        '''
        truncate input as the length of TG.
        '''
        passage = self.passages[index]
        passage_tokens = encode_TG_string(self.tokenizer, passage)
        passage_TG_tokens = self.vocab.convert_treenpy_to_TG(passage_tokens)
        if self.transformer_grammar_type == "pause1/2":
            passage_TG_tokens = self.vocab.convert_treenpy_to_terminal(passage_tokens)
        if self.train_summary is not None:
            train_summary = self.train_summary[index]
            train_summary_tokens = encode_TG_string(self.tokenizer, train_summary)
            train_summary_TG_tokens = self.vocab.convert_treenpy_to_TG(train_summary_tokens)
            if self.transformer_grammar_type == "pause1/2":
                train_summary_TG_tokens = self.vocab.convert_treenpy_to_terminal(train_summary_TG_tokens)
            passage_truncate_length = self.model_ctx_len - len(train_summary_TG_tokens) - len(self.prompts_TG_tokens) - 1 - 1 # one for bos and one for eos
            input_ids = np.concatenate([
                np.array([self.vocab.bos]),
                passage_TG_tokens[:passage_truncate_length],
                self.prompts_TG_tokens,
                train_summary_TG_tokens,
                np.array([self.vocab.eos])
            ])
            loss_tokens = np.concatenate([
                train_summary_TG_tokens, 
                np.array([self.vocab.eos])
            ])
        else:
            passage_truncate_length = self.model_ctx_len - self.MAX_SUMMARY_LENGTH - len(self.prompts_TG_tokens) - 1 - 1
            input_ids = np.concatenate([
                np.array([self.vocab.bos]),
                passage_TG_tokens[:passage_truncate_length],
                self.prompts_TG_tokens,
            ])
            loss_tokens = None
        
        attention_bias, label_mask, TG_label_mask = None, None, None
        if self.transformer_grammar_type == "terminal":
            input_ids = self.vocab.convert_treenpy_to_terminal(input_ids)
            if loss_tokens is not None:
                loss_tokens = self.vocab.convert_treenpy_to_terminal(loss_tokens)
        elif self.transformer_grammar_type[:4] == "tree":
            input_ids = self.vocab.convert_TGnpy_to_tree(input_ids)
            if loss_tokens is not None:
                loss_tokens = self.vocab.convert_TGnpy_to_tree(loss_tokens)
        elif self.transformer_grammar_type == "pause1/2":
            paused_input = np.zeros(2 * len(input_ids))
            paused_input[::2] = input_ids
            paused_input[1::2] = 50260  # Special token
            input_ids = paused_input
        elif self.generate_TG_attention_bias is not None:            
            input_ids = torch.tensor(input_ids)
            attention_bias, TG_label_mask = self.generate_TG_attention_bias(input_ids)
        
        if loss_tokens is not None:
            label_mask = torch.zeros((input_ids.shape[0], ), dtype=torch.bool)
            label_mask[label_mask.shape[0] - loss_tokens.shape[0]:] = True
            if TG_label_mask is not None:
                label_mask = torch.bitwise_and(label_mask, TG_label_mask)
        return {
            "attention_bias": attention_bias,
            "gold_summary" : self.gold_summary[index],
            "label_mask": label_mask,
            "input_ids": input_ids,
        }

    def __len__(self):
        return len(self.passages)
    
    def collate_fn(self, data):
        return self.collator(data)

class RougeMetric(Metric):
    def __init__(self, 
                 metric_type:str = "rouge",
                 vocab_path:str = None,
                 tokenizer:Tokenizer = None
        ) -> None:
        super().__init__(sync_on_compute=True)
        self.add_state("predictions", default=[], dist_reduce_fx=None)
        self.add_state("references", default=[], dist_reduce_fx=None)
        self.tokenizer = tokenizer

    def update(self, batch, predictions, references):
        input_ids = batch["input_ids"].cpu()
        for b in range(predictions.shape[0]):
            # pred_summary = self.tokenizer.decode(predictions[b].tolist())
            passage = self.tokenizer.decode(input_ids[b].tolist(), skip_special_tokens=False)
            print(f"<New Passage>: {passage} {self.tokenizer.decode(predictions[b].tolist(), skip_special_tokens=False)}")
            self.predictions.append(predictions[b])

        for gold in references:
            self.references.append(torch.tensor(self.tokenizer.encode(gold, add_special_tokens=False), device=self.device))

    def reset(self):
        self.predictions = []
        self.references = []

    def compute(self):
        rouge = evaluate.load('rouge')
        predictions_str = []
        references_str = []
        for prediction in self.predictions:
            predictions_str.append(self.tokenizer.decode(prediction.tolist()))
        for reference in self.references:
            references_str.append(self.tokenizer.decode(reference.tolist()))
        results = rouge.compute(
            predictions=predictions_str,
            references=references_str,
            use_stemmer=True,
            rouge_types=['rouge1', 'rouge2', 'rougeL'],  
            use_aggregator=True,  # ave scores
        )
        results["R-AVG"] = sum(results.values()) / 3
        return results

class TGPerplexityDocumentLevelMetric(Metric):
    full_state_update: bool = False
    
    def __init__(
            self, 
            metric_type="doc_ppl", 
            vocab_path = None,
            term_length = None, 
            device_eval_batch_size = None, 
            dataset_length = None,
            samples_per_sent = 300,
        ) -> None:
        """metric_type: f1, acc, len_norm, pmi_dc, ce_loss, bpb"""
        super().__init__(sync_on_compute=False) # since we use one device to eval, sync could be false

        self.metric_type = "doc_ppl"
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.term_length = term_length
        self.samples_per_sent = samples_per_sent
        self.cur_sent = 0
        self.cur_batch = 0
        self.device_eval_batch_size = device_eval_batch_size 
        self.add_state("loglikelihoods", default=torch.zeros((dataset_length//self.samples_per_sent, self.samples_per_sent), dtype=torch.float32), dist_reduce_fx=None)

    def reset(self):
        self.cur_sent = 0
        self.cur_batch = 0

    def update(self, batch: Dict[str, Any], ce_loss:torch.Tensor, lm_logits: Optional[torch.Tensor] = None, dc_lm_logits=None):
        self.loglikelihoods[self.cur_sent, self.cur_batch:self.cur_batch + self.device_eval_batch_size] = ce_loss
        self.cur_batch += self.device_eval_batch_size
        if self.cur_batch == self.samples_per_sent:
            self.cur_batch = 0
            self.cur_sent += 1

    def compute(self) -> torch.Tensor: 
        data_numwords = sum(self.term_length)
        ppl = torch.logsumexp(-self.loglikelihoods, dim=1).sum().item()
        ppl = np.exp(-ppl / data_numwords)
        return torch.tensor(ppl)

# Deprecated please use Document Level ppl metric
class TGPerplexitySentenceLevelMetric(Metric):
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

    def reset(self):
        self.loglikelihoods = []

    def update(self, batch: Dict[str, Any], lm_logits: torch.Tensor, dc_lm_logits=None):
        logits_for_loss = lm_logits[..., :-1, :].to(dtype=torch.float32).contiguous()
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
        dataset_name: str = None, # tg or tree
        model_ctx_len: int = 2048,
        split="validation",
        metric_type="sent",  # Override default metric type, whether be sent/doc
        generate_TG_attention_bias: Optional[TG_attention_bias] = None,
        vocab_path: str = None,
        device_eval_batch_size: int = 60, 
        **kwargs
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

        log.info(
                f"Starting loading {self.dataset_name}_approx_ppl dataset"
            )

        self.dataset = np.load(os.path.join(self.dataset_path, f"{self.dataset_name}_300.npy"), mmap_mode='r')
        self.sent_index = torch.LongTensor(np.load(os.path.join(self.dataset_path, f"{self.dataset_name}_sent_index.npy")))
        self.doc_index = torch.LongTensor(np.load(os.path.join(self.dataset_path, f"{self.dataset_name}_doc_index.npy")))
        self.length = len(self.sent_index)
        self.generate_TG_attention_bias = generate_TG_attention_bias
        self.prep_examples()
        self.reset()
        log.info(f"Loading Dataset finished")

    def __getitem__(self, index):
        return {
            "sent_id" : index//self.SENT_SIZE + 1, 
            "doc_id": self.sent_doc_id[index//self.SENT_SIZE],
            "input_ids": self.dataset[self.sent_index[index]:self.sent_index[index+1]], 
        }

    def __len__(self):
        return self.length

    def get_term_length(self):
        return self.term_len

    def reset(self) -> None:
        self.cur_doc_id = 0
        self.sent_to_add = None
        self.num_evaled = 0

    def prep_examples(self):
        """Append doc_ids to each example so that they are processed together in the metric"""
        self.sent_index = torch.cat([torch.LongTensor([0]), torch.cumsum(self.sent_index, dim=0)])
        self.doc_index = torch.cumsum(self.doc_index, dim=0)
        self.sent_doc_id = torch.zeros((self.length // self.SENT_SIZE + 1), dtype=torch.int)
        self.term_len = [0] * (self.length // self.SENT_SIZE + 1)
        self.sent_doc_id[0] = 1
        self.sent_doc_id[self.doc_index] = 1
        self.sent_doc_id = torch.cumsum(self.sent_doc_id, dim=0)

        for i in range(1, len(self.term_len)):
            sent = self[self.SENT_SIZE * (i-1)]
            self.term_len[i] = sum([self.vocab.is_terminal(token) or token==self.vocab.eos for token in sent["input_ids"]])

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
            tokens = np.concatenate([tokens, [self.vocab.pad] * (max_len - len(tokens))], axis=0)

            if len(tokens) > self.model_ctx_len:
                tokens = tokens[: self.model_ctx_len]

            return tokens

    def collate_fn(self, data):
        # pad to max length
        if self.metric_type=="doc" and data[0]["doc_id"] > self.cur_doc_id:
            self.cur_doc_id = data[0]["doc_id"]
            if self.generate_TG_attention_bias is not None:
                self.generate_TG_attention_bias.reset_state()
        
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
            # make sure Gen TG bias have the correct length
            cur_input_id = torch.LongTensor(self.pad_tokens_until_max(sample["input_ids"], max_len=max_input_len))

            attention_bias, label_mask = None, None
            if self.generate_TG_attention_bias is not None:
                attention_bias, label_mask = self.generate_TG_attention_bias(cur_input_id)
            input_ids.append(cur_input_id)
            
            if attention_bias is not None:
                if not isinstance(attention_bias, torch.Tensor):
                    attention_bias = torch.tensor(attention_bias)
                # Reshape to `(1, seq_len, seq_len)`
                while len(attention_bias.shape) < 3:
                    attention_bias = attention_bias.unsqueeze(0)
                all_attention_bias.append(attention_bias)

            if label_mask is not None:
                if not isinstance(label_mask, torch.Tensor):
                    label_mask = torch.tensor(label_mask)
                all_label_mask.append(label_mask)

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
            if self.num_evaled % self.SENT_SIZE == self.batch_size or self.batch_size == self.SENT_SIZE:
                # Make sure bias has the same length with kv cache, we must pass pad into GenBias
                self.sent_to_add = torch.LongTensor(self.pad_tokens_until_max(data[0]["input_ids"], max_len=max_input_len))
                batch["add_len"] = data[0]["input_ids"].shape[0]
            if self.num_evaled % self.SENT_SIZE == 0:
                if self.generate_TG_attention_bias is not None:
                    self.generate_TG_attention_bias(self.sent_to_add, True)
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
        getattr(self, self.map_task_dict[task]).append(torch.tensor(result, dtype=torch.bool, device=self.device))

    def compute(self) -> Dict[str, float]:
        acc_dict = {}
        avg_acc = 0.0
        for key in test_suite_dict:
            acc = sum(getattr(self, key))
            if isinstance(acc, torch.Tensor):
                acc = acc.item()
            acc_dict[key] = acc
            if acc_dict[key]>0:
                acc_dict[key] /= len(getattr(self, key))
            if key != 'nn-nv-rpl':
                avg_acc += acc_dict[key]
        acc_dict["avg"] = avg_acc / len(test_suite_dict)
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
        transformer_grammar_type: str = "",
        **kwargs
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
        self.transformer_grammar_type = transformer_grammar_type
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
                        if self.transformer_grammar_type == "pause1/2":
                            sent["tag"][0] = [0] + sent["tag"][0]
                            pause_tag = sent["tag"][0] + sent["tag"][0]
                            pause_tag[0::2] = sent["tag"][0]
                            pause_tag[1::2] = sent["tag"][0]
                            sent["tag"][0] = pause_tag[1:]
                            paused_input = torch.zeros(2 * sent["input_ids"].shape[1], dtype=torch.long)
                            paused_input[::2] = sent["input_ids"][0]
                            paused_input[1::2] = 50260  # Special token
                            sent["input_ids"] = paused_input.unsqueeze(0)
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


BLiMP_TASK_ANAPHOR_AGR = ["anaphor_gender_agreement", "anaphor_number_agreement"]
BLiMP_TASK_ARG_STRUCTURE = ["animate_subject_passive", "animate_subject_trans", "causative",
                            "drop_argument", "inchoative", "intransitive", "passive_1", "passive_2", "transitive"]
BLiMP_TASK_BINDING = ["principle_A_c_command", "principle_A_case_1", "principle_A_case_2",
                      "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3",
                      "principle_A_reconstruction"]
BLiMP_TASK_CONTROL_RAISING = ["existential_there_object_raising", "existential_there_subject_raising",
                              "expletive_it_object_raising", "tough_vs_raising_1", "tough_vs_raising_2"]
BLiMP_TASK_DET_NOUN_AGR = ["determiner_noun_agreement_1", "determiner_noun_agreement_2",
                           "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2",
                           "determiner_noun_agreement_with_adj_2", "determiner_noun_agreement_with_adj_irregular_1",
                           "determiner_noun_agreement_with_adj_irregular_2", "determiner_noun_agreement_with_adjective_1"]
BLiMP_TASK_ELLIPSIS = ["ellipsis_n_bar_1", "ellipsis_n_bar_2"]
BLiMP_TASK_FILLER_GAP = ["wh_questions_object_gap", "wh_questions_subject_gap", "wh_questions_subject_gap_long_distance", 
                         "wh_vs_that_no_gap", "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap", 
                         "wh_vs_that_with_gap_long_distance"]
BLiMP_TASK_IRREGULAR_FORMS = ["irregular_past_participle_adjectives", "irregular_past_participle_verbs"]
BLiMP_TASK_ISLAND_EFFECTS = ["adjunct_island", "complex_NP_island", "coordinate_structure_constraint_complex_left_branch",
                             "coordinate_structure_constraint_object_extraction", "left_branch_island_echo_question",
                             "left_branch_island_simple_question", "sentential_subject_island", "wh_island"]
BLiMP_TASK_NPI_LICENSING = ["matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
                            "only_npi_licensor_present", "only_npi_scope", "sentential_negation_npi_licensor_present",
                            "sentential_negation_npi_scope"]
BLiMP_TASK_QUANTIFIERS = ["existential_there_quantifiers_1", "existential_there_quantifiers_2",
                          "superlative_quantifiers_1", "superlative_quantifiers_2"]
BLiMP_TASK_SUBJECT_VERB_AGR = ["distractor_agreement_relational_noun", "distractor_agreement_relative_clause",
                               "irregular_plural_subject_verb_agreement_1", "irregular_plural_subject_verb_agreement_2",
                               "regular_plural_subject_verb_agreement_1", "regular_plural_subject_verb_agreement_2"]
BLiMP_TASK_DICT = {
    "anaphor_agreement" : BLiMP_TASK_ANAPHOR_AGR,
    "argument_structure" : BLiMP_TASK_ARG_STRUCTURE,
    "binding" : BLiMP_TASK_BINDING,
    "control_raising" : BLiMP_TASK_CONTROL_RAISING,
    "determiner_noun_agreement" : BLiMP_TASK_DET_NOUN_AGR,
    "ellipsis" : BLiMP_TASK_ELLIPSIS,
    "filler_gap_dependency" : BLiMP_TASK_FILLER_GAP,
    "irregular_forms" : BLiMP_TASK_IRREGULAR_FORMS,
    "island_effects" : BLiMP_TASK_ISLAND_EFFECTS,
    "npi_licensing" : BLiMP_TASK_NPI_LICENSING,
    "quantifiers" : BLiMP_TASK_QUANTIFIERS,
    "subject_verb_agreement" : BLiMP_TASK_SUBJECT_VERB_AGR, 
}
BLiMP_TASK_LIST = [x for v in BLiMP_TASK_DICT.values() for x in v]


# TODO: 
class BLiMPMetric(Metric):
    full_state_update: bool = False
    
    def __init__(
            self, 
            metric_type="BLiMP", 
            dataset_name: str = "tree_300", # terminal, tree_300 or tg_300
            vocab_path = None,
            device_eval_batch_size = None, 
            dataset_length = None,
            samples_per_sent = 300, 
            pair_per_task = 1000, 
        ) -> None:
        super().__init__(sync_on_compute=True)

        self.metric_type = metric_type
        self.task_dict = BLiMP_TASK_DICT
        self.task_list = BLiMP_TASK_LIST
        self.pair_per_task = pair_per_task
        self.device_eval_batch_size = device_eval_batch_size 
        self.dataset_length = dataset_length
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)

        if dataset_name in ["terminal", "pause1/2"]:
            self.SENT_SIZE = 1
            self.add_state("loglikelihoods", default=torch.zeros((dataset_length), dtype=torch.float32), dist_reduce_fx="sum")
        else:
            self.SENT_SIZE = samples_per_sent
            self.add_state("loglikelihoods", default=torch.zeros((dataset_length//self.SENT_SIZE, self.SENT_SIZE), dtype=torch.float32), dist_reduce_fx="sum")

    def reset(self):
        if self.SENT_SIZE == 1:
            self.loglikelihoods = torch.zeros((self.dataset_length), dtype=torch.float32, device=self.device)
        else:
            self.loglikelihoods = torch.zeros((self.dataset_length//self.SENT_SIZE, self.SENT_SIZE), dtype=torch.float32, device=self.device)

    def update(self, batch: Dict[str, Any], lm_logits:torch.Tensor):
        logits_for_loss = lm_logits[..., :-1, :].to(dtype=torch.float32).contiguous()
        # tokenizer = Tokenizer.from_file("/home/wangpch/TG-Interpolation/dataset/bbc-news/TG_GPT2_tokenizer.json")
        # print(self.device)
        
        # print_tensor_data(batch["input_ids"])
        # print_tensor_data(batch["attention_bias"])
        # shape: (batch_size * seq_len, vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len)
        labels, label_mask, attention_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, self.vocab.pad)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, self.vocab.pad)
        labels = labels[..., 1:].contiguous()
        # shape: (batch_size * seq_len,)
        labels = labels.view(-1)
        ce_loss = F.cross_entropy(
            logits_for_loss, labels, ignore_index=self.vocab.pad, reduction="none")
        
        # for sent_id, loglikelihood in zip(batch["sent_id"], ce_loss):
        sample_id = batch["sent_id"] % self.SENT_SIZE
        sent_id = batch["sent_id"] // self.SENT_SIZE
        
        
        ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1).sum(dim=1)
        if self.SENT_SIZE==1:
            self.loglikelihoods[sent_id : sent_id + self.device_eval_batch_size] = ce_loss
        else:
            self.loglikelihoods[sent_id, sample_id : sample_id + self.device_eval_batch_size] = ce_loss
        
        # self.loglikelihoods[self.cur_sent, self.cur_batch:self.cur_batch + self.device_eval_batch_size] = ce_loss
        # self.cur_batch += self.device_eval_batch_size
        # if self.cur_batch == self.SENT_SIZE:
        #    self.cur_batch = 0
        #    self.cur_sent += 1


    def compute(self) -> torch.Tensor: 
        cnt_dict = {}
        if self.SENT_SIZE!=1:
            loglikelihoods = torch.logsumexp(-self.loglikelihoods, dim=1)
        else:
            loglikelihoods = -self.loglikelihoods
        
        for task_id, task in enumerate(self.task_list):
            id_bias = task_id * self.pair_per_task * 2
            cnt_dict[task] = 0
            for pair_id in range(self.pair_per_task):
                p_good = loglikelihoods[id_bias + pair_id * 2]
                p_bad = loglikelihoods[id_bias + pair_id * 2 + 1]
                if p_good==p_bad or (not (p_good>p_bad) and not (p_bad>p_good)):
                    print(f"index is {id_bias + pair_id * 2}, prob is {p_good}")

                if p_good > p_bad:
                    cnt_dict[task] += 1

        acc_dict = {}
        total_cnt = 0
        for term, term_task_list in self.task_dict.items():
            term_cnt = 0
            for task in term_task_list: 
                acc_dict[term + '/' + task] = cnt_dict[task] / self.pair_per_task
                term_cnt += cnt_dict[task]
            acc_dict[term + '/overall'] = term_cnt / (self.pair_per_task * len(term_task_list))
            total_cnt += term_cnt

        acc_dict['overall/overall'] = total_cnt / (self.pair_per_task * len(self.task_list))

        return acc_dict


# TODO:
class BLiMPApproximationDataset(metaclass=abc.ABCMeta):
    metric_type: str

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: str = "tree_300", # terminal, tree_300 or tg_300
        model_ctx_len: int = 2048,
        split="test",
        metric_type="BLiMP", 
        generate_TG_attention_bias: Optional[Callable | str] = None, 
        transformer_grammar_type: str = "",
        vocab_path: str = None,
        device_eval_batch_size: int = 60, 
        samples_per_sent: int = 300,
        pair_per_task: int = 1000, 
    ):

        super().__init__()
        self.tokenizer = tokenizer
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.dataset_path = dataset_path
        self.metric_type = metric_type
        self.task_list = BLiMP_TASK_LIST
        self.batch_size = device_eval_batch_size
        self.transformer_grammar_type = transformer_grammar_type

        if transformer_grammar_type in ["terminal", "pause1/2"]:
            self.SENT_SIZE = 1
        else:
            self.SENT_SIZE = samples_per_sent
        self.TASK_SIZE = 2 * pair_per_task * self.SENT_SIZE
        self.length = len(self.task_list) * self.TASK_SIZE 

        self.samples: List[Dict[str, Any]] = []
        if transformer_grammar_type in ["terminal", "pause1/2"]:
            self.dataset_name = "terminal"
        elif transformer_grammar_type[:4] == "tree":
            self.dataset_name = "tree_300"
        else:
            self.dataset_name = "tg_300"
        self.dataset = np.load(os.path.join(self.dataset_path, f"blimp_{self.dataset_name}.npy"), mmap_mode='r')
        self.input_len = self.dataset.shape[1]
        self.generate_TG_attention_bias = generate_TG_attention_bias

        self.prep_examples()
        self.reset()
        log.info(f"Loading Dataset finished")

    def prep_examples(self):
        return

    def __getitem__(self, index):
        return {
            "sent_id" : index, 
            "input_ids": torch.LongTensor(self.dataset[index // self.dataset.shape[1], index % self.dataset.shape[1]].copy()), 
        }

    def __len__(self):
        return self.length

    def reset(self) -> None:
        return

    def collate_fn(self, data):
        sent_ids = []
        input_ids = []
        all_attention_bias = []
        all_label_mask = []
        # pad according to max_lengths
        for sample in data:
            cur_input_id = sample["input_ids"]
            if self.transformer_grammar_type == "pause1/2":
                paused_input = torch.zeros(2 * len(cur_input_id), dtype=cur_input_id.dtype, device=cur_input_id.device)
                paused_input[::2] = cur_input_id
                paused_input[1::2] = torch.where(cur_input_id != self.vocab.pad, 50260, self.vocab.pad)  # Special token
                cur_input_id = paused_input

            attention_bias, label_mask = None, None
            if self.generate_TG_attention_bias is not None:
                attention_bias, label_mask = self.generate_TG_attention_bias(cur_input_id)
            sent_ids.append(sample["sent_id"])
            input_ids.append(cur_input_id)

            if attention_bias is not None:
                if not isinstance(attention_bias, torch.Tensor):
                    attention_bias = torch.tensor(attention_bias)
                # Reshape to `(1, seq_len, seq_len)`
                while len(attention_bias.shape) < 3:
                    attention_bias = attention_bias.unsqueeze(0)
                all_attention_bias.append(attention_bias)

            if label_mask is not None:
                if not isinstance(label_mask, torch.Tensor):
                    label_mask = torch.tensor(label_mask)
                all_label_mask.append(label_mask)

        batch = {
            "sent_id" : min(sent_ids),
            "input_ids": torch.stack(input_ids),
        }
        if all_attention_bias:
            batch["attention_bias"] = torch.stack(all_attention_bias)
        if all_label_mask:
            batch["label_mask"] = torch.stack(all_label_mask)
        return batch

    def token_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def token_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)


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
    BoolQPATH = "./dataset/SuperGLUE/BoolQ/"
    def __init__(
        self,
        tokenizer,
        dataset_path="boolq",
        dataset_name=None,
        model_ctx_len=2048,
        split="val",
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path
        )

    def load_local_datasets(self):
        self.dataset = []
        with open(os.path.join(self.BoolQPATH, f"{self.split}.jsonl"), "r") as file:
            for line in file:
                self.dataset.append(json.loads(line.strip()))
        for key in ["passage", "question"]:
            with open(os.path.join(self.BoolQPATH, f"{self.split}_{key}.txt"), "r") as file:
                for idx, line in enumerate(file):
                    self.dataset[idx][key] = convert_TG_format(line.strip())
        

    def doc_to_text(self, doc):
        return doc["passage"] + "<|SEP|> (SQ (NP Question NP) : " + doc["question"] + " ? SQ) <|SEP|> (NP (NP Answer NP) : (NP"

    def doc_to_continuations(self, doc):
        label = not doc["label"]
        del doc
        # add spaces in front of continuation
        if self.split=="train":
            return [[" yes", " no"][label]]
        else:
            return [" yes", " no"]

    def doc_to_label(self, doc):
        # if doc['answer'] is True, return index of " yes" which is 0
        if doc["label"]:
            return 0
        else:
            return 1

    def doc_to_domain_conditional(self, doc):
        del doc
        return "(NP (NP Answer NP) : (NP"


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

class CB(ICLMultiChoiceTaskDataset):
    """Prompt: "premise\nQuestion:{hypothesis}. True, False or Neither?\nAnswer: {True/False/Neither}"
    continuations: True, False, Neither.

    "cause": "because",
    "effect": "therefore",

    implement PMI_DC
    acc, random at 33%

    {
        'premise': 'It was a complex language. Not written down but handed down. One might say it was peeled down.',
        'hypothesis': 'the language was peeled down',
        'label': 0
    }
    """

    metric_type = "acc"
    CBPATH = "./dataset/SuperGLUE/CB/"
    LABEL_DICT = {"entailment": 0, "contradiction": 1, "neutral": 2}
    def __init__(
        self,
        tokenizer,
        dataset_path="CB",
        dataset_name=None,
        model_ctx_len=2048,
        split="val",
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path
        )

    def load_local_datasets(self):
        self.dataset = []
        with open(os.path.join(self.CBPATH, f"{self.split}.jsonl"), "r") as file:
            for line in file:
                self.dataset.append(json.loads(line.strip()))
        for key in ["premise", "hypothesis"]:
            with open(os.path.join(self.CBPATH, f"{self.split}_{key}.txt"), "r") as file:
                for idx, line in enumerate(file):
                    self.dataset[idx][key] = convert_TG_format(line.strip())
    
    def doc_to_text(self, doc):
        return doc["premise"] + "<|SEP|> (FRAG (NP Question NP) : " + doc["hypothesis"] + " . FRAG) (FRAG True, False or Neither ? FRAG) <|SEP|> (NP (NP Answer NP) : (NP"

    def doc_to_continuations(self, doc):
        label = self.LABEL_DICT[doc["label"]]
        del doc
        # add spaces in front of continuation
        if self.split=="train":
            return [[" True", " False", " Neither"][label]]
        else:
            return [" True", " False", " Neither"]

    def doc_to_label(self, doc):
        return self.LABEL_DICT[doc["label"]]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "(NP (NP Answer NP) : (NP"

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

    metric_type = "acc"
    RTEPATH = "./dataset/SuperGLUE/RTE/"
    def __init__(
        self,
        tokenizer,
        dataset_path="rte",
        dataset_name=None,
        model_ctx_len=2048,
        split="val",
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path
        )
    
    def load_local_datasets(self):
        self.dataset = []
        with open(os.path.join(self.RTEPATH, f"{self.split}.jsonl"), "r") as file:
            for line in file:
                self.dataset.append(json.loads(line.strip()))
        for key in ["premise", "hypothesis"]:
            with open(os.path.join(self.RTEPATH, f"{self.split}_{key}.txt"), "r") as file:
                for idx, line in enumerate(file):
                    self.dataset[idx][key] = convert_TG_format(line.strip())

    def doc_to_text(self, doc):
        return doc["premise"] + "<|SEP|> (S (NP Question NP) : " + doc["hypothesis"] + " S) (ADJP (ADJP True or False ADJP) ? ADJP) <|SEP|> (NP (NP Answer NP) : (NP"

    def doc_to_continuations(self, doc):
        label = doc["label"]=="not_entailment"
        del doc
        # add spaces in front of continuation
        if self.split=="train":
            return [[" True", " False"][label]]
        else:
            return [" True", " False"]

    def doc_to_label(self, doc):
        return doc["label"]=="not_entailment"

    def doc_to_domain_conditional(self, doc):
        del doc
        return "(NP (NP Answer NP) : (NP"


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


TG_path = "./dataset/bbc-news/testppl_tg/"
TXLTREE_path = "./dataset/bbc-news/testppl_tree/"
TESTOR_TG_PATH = "./dataset/bbc-news/testor_tg/"
TESTOR_TREE_PATH = "./dataset/bbc-news/testor_tree/"
BLiMP_PATH = "./dataset/BLiMP/tree300/"
BLiMP_RAW_PATH = "./dataset/BLiMP/raw_data/"

TG_task_map = {
    "tg_approx_sent": (TGPerplexityApproximationDataset, {"dataset_path": TG_path, "dataset_name": "newppl.json", "metric_type": "sent"}),
    "tg_approx_sent_testor": (TGPerplexityApproximationDataset, {"dataset_path": TG_path, "dataset_name": "smallppl.json", "metric_type": "sent"}),
    "txl_approx_sent": (TGPerplexityApproximationDataset, {"dataset_path": TXLTREE_path, "dataset_name": "newppl.json", "metric_type": "sent"}),
    "txl_approx_sent_testor": (TGPerplexityApproximationDataset, {"dataset_path": TXLTREE_path, "dataset_name": "smallppl.json", "metric_type": "sent"}),
    "tg_approx_doc": (TGPerplexityApproximationDataset, {"dataset_path": TG_path, "dataset_name": "tg", "metric_type": "doc"}),
    "tg_approx_doc_testor": (TGPerplexityApproximationDataset, {"dataset_path": TESTOR_TG_PATH, "dataset_name": "CC-MAIN-2022-49", "metric_type": "doc"}),
    "txl_approx_doc": (TGPerplexityApproximationDataset, {"dataset_path": TXLTREE_path, "dataset_name": "tree", "metric_type": "doc"}),
    "txl_approx_doc_testor": (TGPerplexityApproximationDataset, {"dataset_path": TESTOR_TREE_PATH, "dataset_name": "CC-MAIN-2022-49", "metric_type": "doc"}),
    "syntactic_generalization": (SGDataset, {"dataset_path": "./evaluation/SG/tokenized"}), 
    "BLiMP": (BLiMPApproximationDataset, {"dataset_path": BLiMP_PATH}), 
    "xsum": (XsumDataset, {"dataset_path":"./dataset/Xsum", "metric_type": "rouge"}),
    "xsum_valid": (XsumDataset, {"dataset_path":"./dataset/Xsum", "metric_type": "rouge", "split":"validation"})
}

Super_GLUE = {
    "boolq": BoolQ,
    "cb": CB,
    "copa": COPA,
    "rte": RTE,
}

label_to_task_map = {
    "piqa": PIQA,
    "hellaswag": HellaSwag,
    "winogrande": WinoGrande,
    "openbook_qa": OpenBookQA,
    "sciq": SciQ,
    "arc_easy": ArcEasy,
    "arc_easy_ppl": ArcEasyCELoss,
    "arc_challenge": ArcChallenge,
    "basic_arithmetic": BasicArithmetic,
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
    **TG_task_map,
    **label_to_task_map,
    **label_to_task_map_new,
    **Super_GLUE,
}
