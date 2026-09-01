"""Standalone K-best constituency parsing for the test-PPL corpus.

This is a drop-in replacement for the original ``parse_testppl.py`` that
generated ``dataset/bbc-news/test300/*.txt`` (300 parse candidates per
sentence, one tree per line).  The original relied on a *forked* benepar +
torch_struct in the ``berkepar`` conda env (a ``TreeCRFtopk``/``CKY_CRF_TOPK``
K-best decode path monkey-patched into the libraries).

This version needs only the **stock** (unmodified) benepar and torch_struct:
it inlines the K-best decode classes into the script's own namespace and reads
raw span scores via ``ChartParser.parse(..., return_scores=True)``.

It also fixes two data defects at the source:
  1. ``(ADJ ... ADJ)`` leak — the EWT label ``ADJ`` has no NT-bracket token in
     ``TG_GPT2_tokenizer.json``; it is normalized to ``ADJP`` before
     serialization (the only such label; verified against the tokenizer's
     added-tokens and benepar_en3_large's label_vocab).
  2. Bare-token (treeless) candidates — for very short boundary fragments the
     K-best low-ranked positions degenerate to empty charts; those are dropped
     and the sentence is back-filled to exactly 300 candidates using the
     lowest-score surviving tree, so every sentence still has 300 legal,
     terminal-consistent trees (keeps doc-level PPL well-defined: cand 0
     unchanged, K=300 uniform, batch token-alignment holds).

Output format is unchanged: 300 lines per sentence (one TG-bracket tree each),
so ``tokenize_testppl.py`` (which assembles via ``line_num % 300``) is
unaffected.

Run (from the ``datatools/`` dir, as the original did):
  export PYTHONPATH=/home/wangpch/TG-Interpolation
  python parse_testppl.py --input_list CC-MAIN-2013-48,CC-MAIN-2014-10
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# Must be set before benepar/transformers imports their generated protobuf code.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import benepar
import spacy
from datasets import load_dataset
from nltk import Tree
from tokenizers import Tokenizer

# Make the repo importable for convert_TG_format / benepar_parse helpers when
# the script is run from datatools/.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
# helpers moved into datatools/parse_pretrain_data/ on 2026-08-25
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "parse_pretrain_data"))

from convert_TG_and_tokenize import convert_TG_format  # noqa: E402
from benepar_parse import preprocess_text, split_long_sentence, set_custom_boundaries  # noqa: E402

# K-best decode is a fixed 300 to match the existing corpus layout.
K_BEST = 300

# EWT ``ADJ`` is the only benepar_en3_large top-level constituent label absent
# from the tokenizer's NT-bracket set; map it to its PTB equivalent so it
# serializes to a real ``<(ADJP>`` token instead of leaking as literal text.
LABEL_NORMALIZE = {"ADJ": "ADJP"}


# --------------------------------------------------------------------------- #
# Inline K-best CKY decode (ported from the berkepar fork; self-contained).
# --------------------------------------------------------------------------- #
import torch_struct  # noqa: E402
from torch_struct.helpers import _Struct, Chart  # noqa: E402

A, B = 0, 1


class CKY_CRF_TOPK(_Struct):
    """CKY log-partition that supports KMaxSemiring top-k decoding.

    Differs from torch_struct.CKY_CRF only by maintaining a parallel B-chart
    built from ``sum_exclu0`` (label-0 / "no constituent" excluded) so that
    non-root right children must carry a real label — without this, low-ranked
    K-best trees degenerate to empty/degenerate structures.
    """

    def _check_potentials(self, edge, lengths=None):
        batch, N, _, NT = self._get_dimension(edge)
        edge = self.semiring.convert(edge)
        if lengths is None:
            lengths = torch.LongTensor([N] * batch).to(edge.device)
        return edge, batch, N, NT, lengths

    def logpartition(self, scores, lengths=None, force_grad=False):
        semiring = self.semiring
        scores, batch, N, NT, lengths = self._check_potentials(scores, lengths)

        # KMaxSemiring in stock torch_struct lacks sum_exclu0 (the fork added
        # it).  Attach it once per semiring class so the B-chart can exclude
        # the null label (index 0), preventing degenerate K-best trees.
        if not hasattr(semiring, "sum_exclu0"):
            k = semiring.size()

            @staticmethod
            def sum_exclu0(xs, dim=-1):
                if dim == -1:
                    xs = xs[..., 1:]
                    xs = xs.permute(tuple(range(1, xs.dim())) + (0,))
                    xs = xs.contiguous().view(xs.shape[:-2] + (-1,))
                    xs = torch.topk(xs, k, dim=-1)[0]
                    xs = xs.permute((xs.dim() - 1,) + tuple(range(0, xs.dim() - 1)))
                    return xs
                raise ValueError("sum_exclu0 only supports dim=-1")
            semiring.sum_exclu0 = sum_exclu0

        beta = [Chart((batch, N, N), scores, semiring) for _ in range(2)]
        L_DIM, R_DIM = 2, 3

        reduced_scores = semiring.sum(scores)
        reduced_scores_ex0 = semiring.sum_exclu0(scores)
        term = reduced_scores.diagonal(0, L_DIM, R_DIM)
        ns = torch.arange(N)
        beta[A][ns, 0] = term
        beta[B][ns, N - 1] = term

        for w in range(1, N):
            left = slice(None, N - w)
            right = slice(w, None)
            Y = beta[A][left, :w]
            Z = beta[B][right, N - w:]
            tmp = semiring.dot(Y, Z)
            score = reduced_scores.diagonal(w, L_DIM, R_DIM)
            score_ex0 = reduced_scores_ex0.diagonal(w, L_DIM, R_DIM)
            beta[A][left, w] = semiring.times(tmp, score)
            beta[B][right, N - w - 1] = semiring.times(tmp, score_ex0)

        final = beta[A][0, :]
        log_Z = final[:, torch.arange(batch), lengths - 1]
        return log_Z, [scores]


class TreeCRFtopk(torch_struct.StructDistribution):
    struct = CKY_CRF_TOPK
    arg_constraints = {}  # silence torch.distribution validation warning


def kbest_charts(scores: np.ndarray, length: int, k: int = K_BEST) -> np.ndarray:
    """Return the top-k parse charts for one sentence.

    ``scores`` is the benepar span-score chart of shape ``(L, L, NT)`` (label 0
    is the "no constituent" / null label).  Returns ``(k, L, L)`` int arrays
    where each ``[j, i, m]`` is the label index of the j-th best tree's
    constituent spanning ``[i, m+1)`` (0 == none).
    """
    scores_t = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)  # (1,L,L,NT)
    length_t = torch.tensor([length])
    dist = TreeCRFtopk(scores_t, lengths=length_t)
    amax = dist.topk(k)               # (k, 1, L, L, NT) marginals (one-hot over labels)
    amax[..., 0] += 1e-9              # resolve null-label ties toward "none"
    # argmax over the label axis -> (k, 1, L, L) label-index charts
    charts = amax.argmax(-1)[:, 0].detach().cpu().numpy()  # (k, L, L)
    return charts


# --------------------------------------------------------------------------- #
# Tree post-processing
# --------------------------------------------------------------------------- #
def normalize_labels(tree: Tree) -> Tree:
    """Map EWT-only labels (e.g. ADJ) to tokenizer-supported ones (ADJP)."""
    if isinstance(tree, Tree):
        lbl = tree.label()
        new_label = LABEL_NORMALIZE.get(lbl, lbl)
        return Tree(new_label, [normalize_labels(child) for child in tree])
    return tree


def chart_has_root(chart: np.ndarray, length: int) -> bool:
    """A valid full-sentence tree must label the root span [0, length)."""
    return bool(chart[0, length - 1] != 0)


def tree_to_tg(tree: Tree) -> str:
    """nltk Tree -> TG-bracket string (empty string on failure)."""
    parsed_string = tree.pformat(margin=100000)
    if tree.leaves() == ["\n"]:
        parsed_string = "(Ċ Ċ)"
    return convert_TG_format(parsed_string)


# --------------------------------------------------------------------------- #
# Sentence splitting (unchanged from the original script)
# --------------------------------------------------------------------------- #
def process_text(text: str, max_len: int = 256):
    text = preprocess_text(text)
    doc = sentparser(text)
    final_sentences = []
    for sent in doc.sents:
        tokens = [token.text for token in sent]
        if len(sent) <= max_len:
            final_sentences.append(tokens)
        else:
            segments = split_long_sentence(tokens, max_len)
            final_sentences.extend(segments)
    return final_sentences


# Module-level parser handles (set in main()).
sentparser = None
chart_parser = None
decoder = None


def parse_sentence_kbest(words) -> list:
    """Return exactly K_BEST TG-bracket strings for one sentence.

    Uses stock benepar to get span scores, then inline K-best decode.  Drops
    degenerate charts and back-fills with the lowest-score surviving tree so
    the count is always K_BEST and every candidate is a legal tree sharing the
    same terminals.
    """
    sentence = benepar.InputSentence(words=words)
    # benepar's retokenizer raises StopIteration on a bare newline leaf; the
    # original script special-cased this to the constant parse "(Ċ Ċ)".
    if words == ["\n"]:
        return ["(Ċ Ċ)"] * K_BEST
    encoded = [beneparser._with_missing_fields_filled(sentence)]
    # Get raw span scores (stock benepar supports return_scores).
    scores = list(chart_parser.parse(encoded, return_scores=True))[0]
    L = len(words)

    charts = kbest_charts(scores, L, K_BEST)

    # Leaves with predicted POS tags (matching the original output style:
    # "(TAG word)").  Run a 1-best parse to reuse benepar's tag prediction.
    one_best = list(beneparser.parse(sentence))[0]
    leaves = one_best.pos() if one_best.pos() else [(w, "XX") for w in words]

    tg_strings = []
    for j in range(K_BEST):
        chart = charts[j, :L, :L]
        if not chart_has_root(chart, L):
            continue
        comp = decoder.compressed_output_from_chart(chart)
        tree = comp.to_tree(leaves, decoder.label_from_index)
        tree = normalize_labels(tree)
        tg = tree_to_tg(tree)
        if tg:
            tg_strings.append(tg)

    if not tg_strings:
        # No valid tree at all (extreme fragment): emit the 1-best as all 300.
        tg_strings = [tree_to_tg(normalize_labels(one_best))]
    if len(tg_strings) < K_BEST:
        fill = tg_strings[-1]   # lowest-score surviving tree
        tg_strings = tg_strings + [fill] * (K_BEST - len(tg_strings))
    return tg_strings


def main() -> None:
    global sentparser, chart_parser, decoder, beneparser

    parser = argparse.ArgumentParser()
    parser.add_argument("--eos", type=int, default=50256)
    parser.add_argument("--bos", type=int, default=50257)
    parser.add_argument("--input_json", type=str, default="../dataset/bbc-news/test_index.json")
    parser.add_argument("--output_dir", type=str, default="../dataset/bbc-news/test300/")
    parser.add_argument("--input_list", type=str, required=True)
    args = parser.parse_args()

    parse_list = args.input_list.split(",")
    tokenizer = Tokenizer.from_file("TG_GPT2_tokenizer.json")

    sentparser = spacy.load("en_core_web_md")
    sentparser.add_pipe("set_custom_boundaries", before="parser")

    beneparser = benepar.Parser("benepar_en3_large")
    chart_parser = beneparser._parser          # stock ChartParser
    decoder = chart_parser.decoder             # ChartDecoder

    with open(args.input_json, "r") as f:
        test_docs = json.load(f)

    for bbc_split in parse_list:
        indices = test_docs[bbc_split]
        ds = load_dataset("permutans/fineweb-bbc-news", bbc_split)["train"]
        os.makedirs(args.output_dir, exist_ok=True)
        with open(args.output_dir + bbc_split + ".txt", "w+") as file:
            for index in tqdm(indices, desc=bbc_split):
                line = ds[index]["text"]
                split_sents = process_text(line.strip(), max_len=256)
                sents, end = [], []
                for sent in split_sents:
                    if sent == ["\n"]:
                        end[-1] = True
                    else:
                        sents.append(sent)
                        end.append(False)

                cur_cnt = 0
                for sent, if_endline in zip(sents, end):
                    cur_cnt += 1
                    tg_strings = parse_sentence_kbest(sent)
                    for tg_string in tg_strings:
                        input_ids = tokenizer.encode(tg_string).ids
                        if if_endline:
                            input_ids.append(198)  # endline token
                        if cur_cnt == 1:
                            input_ids = [args.bos] + input_ids
                        elif cur_cnt == len(split_sents):
                            input_ids = input_ids + [args.eos]
                        file.write(tg_string + "\n")


if __name__ == "__main__":
    main()
