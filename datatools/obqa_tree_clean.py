"""
Normalize parse trees: graft each choice subtree onto the gold tree's prefix structure.

Input:
  - A .txt file where every 4 consecutive lines are the parse trees (A/B/C/D) for one sample.
  - A .jsonl file with matching samples in order, each containing question_stem, choices, answerKey.

Output:
  - A .txt file with the same layout (4 lines per sample), trees normalized to gold prefix.

Usage:
  python normalize_trees.py <trees.txt> <data.jsonl> <output.txt>
"""

import sys
import json
import copy
import argparse
from nltk import Tree


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Penn Treebank bracket escapes -> actual characters
PTB_BRACKET_MAP = {
    "-LRB-": "(", "-RRB-": ")",
    "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}",
}

PUNCT_CHARS = set(".!?,;:")


# ---------------------------------------------------------------------------
# Utility: split a line that may contain multiple concatenated trees
# ---------------------------------------------------------------------------

def split_trees(line: str) -> list[str]:
    """
    Split a line like '(S ...) (S ...) (S ...)' into individual tree strings.
    Tracks parenthesis depth to find boundaries between top-level trees.
    """
    trees = []
    depth = 0
    start = None
    for i, ch in enumerate(line):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                trees.append(line[start : i + 1])
                start = None
    return trees


# ---------------------------------------------------------------------------
# Utility: build a map from treeposition -> (start_leaf, end_leaf) span
# ---------------------------------------------------------------------------

def build_span_map(tree: Tree) -> dict:
    """Return {treeposition: (start, end)} for every node. end is exclusive."""
    spans = {}

    def _rec(node, pos: tuple, offset: int) -> int:
        if isinstance(node, str):  # leaf
            spans[pos] = (offset, offset + 1)
            return offset + 1
        start = offset
        for i, child in enumerate(node):
            offset = _rec(child, pos + (i,), offset)
        spans[pos] = (start, offset)
        return offset

    _rec(tree, (), 0)
    return spans


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def normalize_ptb(token: str) -> str:
    """Convert a PTB bracket escape to its actual character."""
    return PTB_BRACKET_MAP.get(token, token)


def leaves_to_chars(leaves) -> str:
    """Concatenate leaves into a space-free string, resolving PTB escapes."""
    return "".join(normalize_ptb(leaf) for leaf in leaves)


def rstrip_punct(s: str) -> str:
    """Strip trailing punctuation characters."""
    while s and s[-1] in PUNCT_CHARS:
        s = s[:-1]
    return s


def count_trailing_punct(leaves: list[str]) -> int:
    """
    Count how many trailing leaves are pure punctuation tokens.
    A token like 'H2O.' is NOT pure punctuation and stops the scan.
    """
    count = 0
    for leaf in reversed(leaves):
        if len(leaf) > 0 and all(c in PUNCT_CHARS for c in leaf):
            count += 1
        else:
            break
    return count


# ---------------------------------------------------------------------------
# Character-level matching of choice text against a leaf sequence
# ---------------------------------------------------------------------------

def match_choice_in_leaves(leaves: list[str], choice_text: str):
    """
    Find the leaf index where choice_text starts within `leaves`,
    searching backwards from the content end (excluding trailing punct).

    Handles:
      - tokenization splits (e.g. "spot," -> "spot" ",")
      - PTB bracket escapes (e.g. "-LRB-" -> "(")
      - fused trailing punct (e.g. "H2O." leaf vs "H2O" choice)
      - choice text with its own trailing punct (e.g. "the ocean.")

    Returns (choice_start_leaf_idx, content_end_leaf_idx) or None.
    """
    n = len(leaves)
    n_punct = count_trailing_punct(leaves)
    content_end = n - n_punct

    if content_end <= 0:
        return None

    choice_chars = choice_text.replace(" ", "")

    for start in range(content_end - 1, -1, -1):
        segment = leaves_to_chars(leaves[start:content_end])
        if segment == choice_chars:
            return start, content_end
        if rstrip_punct(segment) == rstrip_punct(choice_chars):
            return start, content_end

    # Case-insensitive fallback
    for start in range(content_end - 1, -1, -1):
        segment = leaves_to_chars(leaves[start:content_end])
        if segment.lower() == choice_chars.lower():
            return start, content_end
        if rstrip_punct(segment).lower() == rstrip_punct(choice_chars).lower():
            return start, content_end

    return None


# ---------------------------------------------------------------------------
# Locate choice: first try within last tree, then across all trees
# ---------------------------------------------------------------------------

def find_choice_in_line(parsed_trees: list[Tree], choice_text: str):
    """
    Locate the choice text within a line's trees.

    Returns a dict:
      {
        "mode": "single_tree" | "multi_tree",
        # For single_tree mode (choice is within the last tree):
        "tree_pos": treeposition within the last tree,
        # For multi_tree mode (choice spans the last N trees):
        "choice_tree_start": index of the first tree that is part of the choice,
        "local_leaf_start": leaf index within that tree where choice begins,
      }
    or None if matching fails entirely.
    """
    # --- Attempt 1: match within last tree only ---
    last_tree = parsed_trees[-1]
    last_leaves = last_tree.leaves()
    match = match_choice_in_leaves(last_leaves, choice_text)

    if match is not None:
        choice_start, content_end = match
        target_span = (choice_start, content_end)
        span_map = build_span_map(last_tree)

        # Exact span node
        candidates = [pos for pos, span in span_map.items() if span == target_span]
        if candidates:
            return {"mode": "single_tree", "tree_pos": min(candidates, key=len)}

        # Fallback: tightest containing node
        containing = [
            (pos, span)
            for pos, span in span_map.items()
            if span[0] <= choice_start
            and span[1] >= content_end
            and not isinstance(last_tree[pos], str)
        ]
        if containing:
            best = min(containing, key=lambda x: (x[1][1] - x[1][0], len(x[0])))
            return {"mode": "single_tree", "tree_pos": best[0]}

    # --- Attempt 2: match across all trees (choice spans multiple trees) ---
    if len(parsed_trees) > 1:
        all_leaves = []
        # (global_leaf_idx -> tree_idx) boundary list
        tree_starts = []  # tree_starts[ti] = global index of first leaf of tree ti
        for ti, t in enumerate(parsed_trees):
            tree_starts.append(len(all_leaves))
            all_leaves.extend(t.leaves())

        match = match_choice_in_leaves(all_leaves, choice_text)
        if match is not None:
            choice_start_global, _ = match

            # Find which tree the choice starts in
            choice_tree_idx = 0
            for ti, ts in enumerate(tree_starts):
                if ts <= choice_start_global:
                    choice_tree_idx = ti

            local_leaf_start = choice_start_global - tree_starts[choice_tree_idx]
            return {
                "mode": "multi_tree",
                "choice_tree_start": choice_tree_idx,
                "local_leaf_start": local_leaf_start,
            }

    return None


# ---------------------------------------------------------------------------
# Normalize one sample
# ---------------------------------------------------------------------------

def normalize_sample(
    tree_strings: list[str],
    choice_texts: list[str],
    gold_idx: int,
    sample_id: str = "?",
) -> list[str]:
    """
    Given 4 tree strings, 4 choice texts, and the gold index, return 4
    normalized tree strings where every tree shares the gold tree's prefix.
    """
    # Split each line into individual tree strings and parse them
    split_strs = [split_trees(s) for s in tree_strings]
    parsed = [[Tree.fromstring(t) for t in parts] for parts in split_strs]

    # --- Locate choice in the gold line ---
    gold_info = find_choice_in_line(parsed[gold_idx], choice_texts[gold_idx])
    if gold_info is None:
        print(
            f"  WARNING (id={sample_id}): could not locate choice span "
            f"in gold (idx={gold_idx}), returning originals",
            file=sys.stderr,
        )
        return tree_strings

    results = []
    for i in range(len(tree_strings)):
        src_info = find_choice_in_line(parsed[i], choice_texts[i])
        if src_info is None:
            print(
                f"  WARNING (id={sample_id}): could not locate choice span "
                f"in tree idx={i}, keeping original",
                file=sys.stderr,
            )
            results.append(tree_strings[i])
            continue

        new_line = _graft(
            gold_strs=split_strs[gold_idx],
            gold_parsed=parsed[gold_idx],
            gold_info=gold_info,
            src_strs=split_strs[i],
            src_parsed=parsed[i],
            src_info=src_info,
        )
        results.append(new_line)

    return results


def _graft(
    gold_strs, gold_parsed, gold_info,
    src_strs, src_parsed, src_info,
) -> str:
    """
    Build a normalized line = gold prefix + source choice.

    Handles both single-tree and multi-tree modes for gold and source
    independently.
    """
    # --- Determine gold prefix trees and gold "boundary" tree ---
    if gold_info["mode"] == "single_tree":
        # Prefix = all trees except the last; the last tree contains stem+choice
        gold_prefix_strs = gold_strs[:-1]
        gold_boundary_tree = gold_parsed[-1]
        gold_choice_pos = gold_info["tree_pos"]
        gold_suffix_strs = []  # nothing after the boundary tree
    else:
        # multi_tree: choice starts at gold_info["choice_tree_start"]
        ct_start = gold_info["choice_tree_start"]
        local_start = gold_info["local_leaf_start"]
        if local_start == 0:
            # Choice starts at the beginning of a tree -> all preceding trees are prefix
            gold_prefix_strs = gold_strs[:ct_start]
            gold_boundary_tree = None
            gold_choice_pos = None
        else:
            # Choice starts mid-tree -> preceding trees + part of this tree is prefix
            gold_prefix_strs = gold_strs[:ct_start]
            gold_boundary_tree = gold_parsed[ct_start]
            # Find the tightest node covering (local_start, len(leaves))
            leaves = gold_boundary_tree.leaves()
            n_punct = count_trailing_punct(leaves)
            content_end = len(leaves) - n_punct
            target = (local_start, content_end)
            span_map = build_span_map(gold_boundary_tree)
            candidates = [p for p, s in span_map.items() if s == target]
            if candidates:
                gold_choice_pos = min(candidates, key=len)
            else:
                containing = [
                    (p, s) for p, s in span_map.items()
                    if s[0] <= local_start and s[1] >= content_end
                    and not isinstance(gold_boundary_tree[p], str)
                ]
                if containing:
                    gold_choice_pos = min(
                        containing, key=lambda x: (x[1][1] - x[1][0], len(x[0]))
                    )[0]
                else:
                    gold_choice_pos = None
        gold_suffix_strs = []

    # --- Determine source choice trees ---
    if src_info["mode"] == "single_tree":
        src_boundary_tree = src_parsed[-1]
        src_choice_pos = src_info["tree_pos"]
        src_choice_tail_strs = []  # no extra choice trees
    else:
        ct_start = src_info["choice_tree_start"]
        local_start = src_info["local_leaf_start"]
        if local_start == 0:
            src_boundary_tree = None
            src_choice_pos = None
            src_choice_tail_strs = src_strs[ct_start:]
        else:
            src_boundary_tree = src_parsed[ct_start]
            leaves = src_boundary_tree.leaves()
            n_punct = count_trailing_punct(leaves)
            content_end = len(leaves) - n_punct
            target = (local_start, content_end)
            span_map = build_span_map(src_boundary_tree)
            candidates = [p for p, s in span_map.items() if s == target]
            if candidates:
                src_choice_pos = min(candidates, key=len)
            else:
                containing = [
                    (p, s) for p, s in span_map.items()
                    if s[0] <= local_start and s[1] >= content_end
                    and not isinstance(src_boundary_tree[p], str)
                ]
                if containing:
                    src_choice_pos = min(
                        containing, key=lambda x: (x[1][1] - x[1][0], len(x[0]))
                    )[0]
                else:
                    src_choice_pos = None
            src_choice_tail_strs = src_strs[ct_start + 1 :]

    # --- Assemble the output ---
    parts = list(gold_prefix_strs)  # copy

    if gold_boundary_tree is not None and gold_choice_pos is not None:
        # Graft source choice subtree into the gold boundary tree
        new_bt = copy.deepcopy(gold_boundary_tree)
        if src_boundary_tree is not None and src_choice_pos is not None:
            src_subtree = copy.deepcopy(src_boundary_tree[src_choice_pos])
        elif src_boundary_tree is not None:
            src_subtree = copy.deepcopy(src_boundary_tree)
        else:
            # Source choice is purely multi-tree with no boundary;
            # we cannot graft a sub-tree, so just concatenate source choice trees
            parts.append(" ".join(str(new_bt).split()))
            parts.extend(src_choice_tail_strs)
            return " ".join(parts)

        if gold_choice_pos == ():
            new_bt = src_subtree
        else:
            new_bt[gold_choice_pos] = src_subtree
        parts.append(" ".join(str(new_bt).split()))
        parts.extend(src_choice_tail_strs)
    elif gold_boundary_tree is None:
        # Gold choice starts on a tree boundary -> no boundary tree to graft into
        if src_boundary_tree is not None and src_choice_pos is not None:
            # Source has a boundary tree with mixed stem+choice;
            # extract just the choice subtree, then append remaining choice trees
            src_subtree = src_boundary_tree[src_choice_pos]
            parts.append(" ".join(str(src_subtree).split()))
            parts.extend(src_choice_tail_strs)
        else:
            # Both are purely multi-tree; concatenate source choice trees
            if src_info["mode"] == "single_tree":
                parts.append(src_strs[-1])
            else:
                ct_start = src_info["choice_tree_start"]
                parts.extend(src_strs[ct_start:])
    else:
        # gold_choice_pos is None — could not locate position; keep original
        parts.append(" ".join(str(gold_boundary_tree).split()))

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    split = "train"
    parser = argparse.ArgumentParser(
        description="Normalize parse trees by grafting each choice subtree "
                    "onto the gold tree's prefix structure."
    )
    parser.add_argument("--trees", type=str, default=f"../dataset/openbookqa/4-backup/openbookqa_{split}.txt", 
                        help="Path to the .txt file with parse trees (4 lines per sample)")
    parser.add_argument("--data", type=str, default=f"../dataset/openbookqa/openbookqa_{split}.jsonl", 
                        help="Path to the .jsonl file with dataset records")
    parser.add_argument("--output", type=str, default=f"../dataset/openbookqa/openbookqa_{split}.txt", 
                        help="Path for the output .txt file")
    args = parser.parse_args()

    trees_path = args.trees
    jsonl_path = args.data
    output_path = args.output

    with open(trees_path, "r", encoding="utf-8") as f:
        tree_lines = [line.rstrip("\n") for line in f if line.strip()]

    if len(tree_lines) % 4 != 0:
        print(
            f"ERROR: number of non-empty lines in {trees_path} ({len(tree_lines)}) "
            f"is not a multiple of 4",
            file=sys.stderr,
        )
        sys.exit(1)

    n_samples = len(tree_lines) // 4

    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if len(records) != n_samples:
        print(
            f"ERROR: {jsonl_path} has {len(records)} records but {trees_path} "
            f"has {n_samples} groups of 4 trees",
            file=sys.stderr,
        )
        sys.exit(1)

    output_lines = []
    for idx in range(n_samples):
        rec = records[idx]
        sample_trees = tree_lines[idx * 4 : (idx + 1) * 4]

        choice_texts = rec["choices"]["text"]
        labels = rec["choices"]["label"]
        gold_label = rec["answerKey"]
        sample_id = rec.get("id", str(idx))

        if gold_label not in labels:
            print(
                f"  WARNING (id={sample_id}): "
                f"answerKey '{gold_label}' not in labels {labels}, skipping",
                file=sys.stderr,
            )
            output_lines.extend(sample_trees)
            continue

        gold_idx = labels.index(gold_label)
        normalized = normalize_sample(sample_trees, choice_texts, gold_idx, sample_id)
        output_lines.extend(normalized)

        if (idx + 1) % 500 == 0:
            print(f"  processed {idx + 1}/{n_samples} samples ...", file=sys.stderr)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"Done. {n_samples} samples written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
