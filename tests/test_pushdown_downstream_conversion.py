from olmo.eval import downstream
from olmo.data.parse_align import TreeVocab
from olmo.eval.downstream import ICLMultiChoiceTaskDataset


class _Vocab:
    def is_opening_non_terminal(self, token):
        return token == 100

    def is_closing_non_terminal(self, token):
        return token == 101

    def convert_treenpy_to_terminal(self, values):
        return values[(values != 100) & (values != 101)]


class _Dataset(ICLMultiChoiceTaskDataset):
    def doc_to_text(self, doc):
        return ""

    def doc_to_continuations(self, doc):
        return []

    def doc_to_label(self, doc):
        return 0

    def doc_to_domain_conditional(self, doc):
        return ""

    def load_local_datasets(self, split, ret=False):
        return []


def test_pushdown_encoding_returns_terminals_and_terminal_coordinate_spans(monkeypatch):
    dataset = object.__new__(_Dataset)
    dataset.tokenizer = object()
    dataset.vocab = _Vocab()
    dataset.parse_binarize_direction = "right"
    dataset.pushdown_tree_vocab = TreeVocab(
        op_lo=100,
        op_hi=100,
        cl_lo=101,
        cl_hi=101,
        id2tok={100: "<(S>", 101: "<S)>"},
    )
    monkeypatch.setattr(
        downstream,
        "encode_TG_string",
        lambda *args, **kwargs: [7, 100, 11, 12, 101, 8],
    )
    terminals, spans = dataset.encode_pushdown_with_spans("ignored")
    assert terminals == [7, 11, 12, 8]
    # The parsed node is binary and coordinates refer to the terminal stream;
    # singleton/preterminal spans are deliberately omitted for Pushdown SHIFTs.
    assert spans == [(1, 1, 2)]
    assert dataset.convert_grammar_input(
        [7, 100, 11, 12, 101, 8], grammar_type="pushdown"
    ) == terminals
