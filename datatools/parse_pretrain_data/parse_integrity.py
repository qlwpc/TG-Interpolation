"""Parser result and one-row-per-document completion checks."""

from pathlib import Path

from nltk import Tree

from datatools.parse_pretrain_data.pipeline_io import atomic_json, sha256_file


def checked_parse_trees(results, inputs):
    # Stock benepar returns a Tree, whereas historical top-k builds return a
    # list of candidates. Tree is itself list-like: blindly taking [0] loses
    # the sentence root (and often all but its first constituent).
    trees = []
    brackets = {"-LRB-": "(", "-RRB-": ")", "-LCB-": "{", "-RCB-": "}",
                "-LSB-": "[", "-RSB-": "]"}
    def normalize(words):
        result = []
        for word in words:
            for escaped, literal in brackets.items():
                word = word.replace(escaped, literal)
            result.append(word)
        return result
    for result in results:
        if isinstance(result, Tree):
            tree = result
        elif isinstance(result, (list, tuple)) and result and isinstance(result[0], Tree):
            tree = result[0]
        else:
            raise ValueError("benepar returned neither a Tree nor top-k Trees")
        trees.append(tree)
    if len(trees) != len(inputs):
        raise ValueError(f"benepar returned {len(trees)} trees for {len(inputs)} sentences")
    for tree, sentence in zip(trees, inputs):
        if normalize(tree.leaves()) != normalize(sentence.words):
            raise ValueError("benepar changed sentence terminals; refusing document index drift")
    return trees


def parsed_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n") or not line.strip():
                raise ValueError(f"incomplete or empty parsed row in {path}:{count + 1}; repair before resuming")
            count += 1
    return count


def write_parse_receipt(path: Path, source_documents: int, parser_model: str, *, complete: bool) -> None:
    documents = parsed_row_count(path)
    if documents != source_documents:
        raise ValueError(f"parsed/source document count mismatch for {path}: {documents} != {source_documents}")
    atomic_json(path.with_suffix(".parse.json"), {
        "schema_version": 1, "parser_model": parser_model, "documents": documents,
        "complete": complete, "parsed_sha256": sha256_file(path),
    })
