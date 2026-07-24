"""Tests for DistributedEvalSampler.

Verifies the core invariants the sampler must guarantee for eval correctness:
  1. No padding duplicates — every sample evaluated at most once across ranks.
  2. No truncation — every sample evaluated exactly once (full coverage).
  3. Counts differ by at most one across ranks.
  4. strided mode: rank r gets indices r, r+W, r+2W, ...
  5. contiguous mode: each rank gets a single contiguous range, and the union
     is [0, N) with no gaps/overlaps.
  6. Works end-to-end with a DataLoader (the actual consumer).
  7. Contrast: torch's DistributedSampler(drop_last=False) DOES pad (the bug
     we fixed), and SequentialDistributedSampler DOES truncate.

Run: python tests/test_distributed_eval_sampler.py
No GPU / no process group needed — num_replicas and rank are passed explicitly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import Dataset, DataLoader
from olmo.data.util import DistributedEvalSampler, SequentialDistributedSampler

try:
    from torch.utils.data.distributed import DistributedSampler as TorchDistributedSampler
    HAVE_TORCH_DS = True
except Exception:
    HAVE_TORCH_DS = False


class ListDataset(Dataset):
    def __init__(self, n):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        return i


def collect(dataset, num_replicas, contiguous):
    """Return list of per-rank index lists."""
    return [list(DistributedEvalSampler(dataset, num_replicas=num_replicas, rank=r, contiguous=contiguous))
            for r in range(num_replicas)]


def assert_full_coverage(dataset, num_replicas, contiguous):
    n = len(dataset)
    per_rank = collect(dataset, num_replicas, contiguous)
    all_idx = []
    for r, idxs in enumerate(per_rank):
        # no duplicates within a rank
        assert len(idxs) == len(set(idxs)), f"rank {r} has intra-rank duplicates: {idxs}"
        all_idx.extend(idxs)
    # (1) no padding duplicates + (2) full coverage
    assert len(all_idx) == len(set(all_idx)) == n, (
        f"N={n} W={num_replicas} contiguous={contiguous}: "
        f"total={len(all_idx)} unique={len(set(all_idx))} (expected {n}, 0 padding, 0 truncation)")
    assert set(all_idx) == set(range(n)), f"missing/extra indices: {set(range(n)) ^ set(all_idx)}"
    # (3) counts differ by at most one
    counts = [len(idxs) for idxs in per_rank]
    assert max(counts) - min(counts) <= 1, f"counts {counts} differ by >1"
    return per_rank


def assert_strided(dataset, num_replicas):
    """(4) rank r gets r, r+W, r+2W, ..."""
    per_rank = assert_full_coverage(dataset, num_replicas, contiguous=False)
    n = len(dataset)
    for r in range(num_replicas):
        expected = list(range(r, n, num_replicas))
        assert per_rank[r] == expected, f"strided rank {r}: {per_rank[r]} != {expected}"


def assert_contiguous_blocks(dataset, num_replicas):
    """(5) each rank is a single contiguous range, union is [0,N) no gaps."""
    per_rank = assert_full_coverage(dataset, num_replicas, contiguous=True)
    for r, idxs in enumerate(per_rank):
        if idxs:
            assert idxs == list(range(idxs[0], idxs[-1] + 1)), (
                f"contiguous rank {r} not a single block: {idxs}")
    # blocks must tile [0, N) in order (block boundaries non-decreasing)
    starts = [idxs[0] for idxs in per_rank if idxs]
    ends = [idxs[-1] + 1 for idxs in per_rank if idxs]
    assert starts == sorted(starts), f"block starts not sorted: {starts}"
    # no gap: each block's start == previous block's end
    for i in range(1, len(starts)):
        assert starts[i] == ends[i - 1], f"gap/overlap between rank {i-1} and {i}: {ends[i-1]} -> {starts[i]}"
    assert starts[0] == 0, f"first block doesn't start at 0: {starts[0]}"
    assert ends[-1] == len(dataset), f"last block doesn't end at N: {ends[-1]} != {len(dataset)}"


def test_invariants_across_configs():
    print("=== test_invariants_across_configs ===")
    # (N, W) pairs: divisible, off-by-one, small, large, N < W
    configs = [(841, 4), (841, 2), (841, 3), (841, 8), (840, 4), (39, 4),
               (23, 4), (1, 4), (7, 4), (100, 1), (5, 5), (3, 8)]
    for n, w in configs:
        ds = ListDataset(n)
        assert_strided(ds, w)
        assert_contiguous_blocks(ds, w)
        print(f"  N={n:4d} W={w}: strided OK, contiguous OK")
    print("  PASS\n")


def test_strided_pattern():
    print("=== test_strided_pattern ===")
    ds = ListDataset(841)
    s0 = list(DistributedEvalSampler(ds, num_replicas=4, rank=0, contiguous=False))
    assert s0[:6] == [0, 4, 8, 12, 16, 20], f"strided rank0 prefix: {s0[:6]}"
    assert s0[-1] == 840, f"strided rank0 last: {s0[-1]}"
    print(f"  rank0[:6]={s0[:6]} ... rank0[-1]={s0[-1]} (strided 0,4,8,...)")
    print("  PASS\n")


def test_contiguous_blocks_pattern():
    print("=== test_contiguous_blocks_pattern ===")
    ds = ListDataset(841)
    blocks = [list(DistributedEvalSampler(ds, num_replicas=4, rank=r, contiguous=True)) for r in range(4)]
    assert blocks[0] == list(range(0, 211)), f"block0: {blocks[0][:3]}..{blocks[0][-3:]}"
    assert blocks[1] == list(range(211, 421))
    assert blocks[2] == list(range(421, 631))
    assert blocks[3] == list(range(631, 841))
    print(f"  rank0=[0..210] rank1=[211..420] rank2=[421..630] rank3=[631..840]")
    print("  PASS\n")


def test_len_matches_iter():
    print("=== test_len_matches_iter ===")
    ds = ListDataset(841)
    for r in range(4):
        for contiguous in (False, True):
            s = DistributedEvalSampler(ds, num_replicas=4, rank=r, contiguous=contiguous)
            assert len(s) == len(list(s)), f"len != iter len: rank{r} contig={contiguous}"
    print("  len(s) == len(list(s)) for all ranks/modes")
    print("  PASS\n")


def test_dataloader_end_to_end():
    print("=== test_dataloader_end_to_end ===")
    """(6) works with DataLoader — simulate 4 ranks, collect all batches, check coverage."""
    ds = ListDataset(841)
    total = []
    for r in range(4):
        sampler = DistributedEvalSampler(ds, num_replicas=4, rank=r, contiguous=False)
        loader = DataLoader(ds, batch_size=1, sampler=sampler, shuffle=False)
        for batch in loader:
            total.append(batch.item())
    assert len(total) == 841, f"DataLoader total {len(total)} != 841"
    assert len(set(total)) == 841, "DataLoader produced duplicates"
    assert set(total) == set(range(841)), "DataLoader missing indices"
    print("  4-rank DataLoader (strided, batch=1): 841 unique indices, full coverage")
    print("  PASS\n")


def test_contrast_old_samplers_buggy():
    print("=== test_contrast_old_samplers_buggy ===")
    """(7) Demonstrate the old samplers were buggy: torch DistributedSampler pads,
    SequentialDistributedSampler truncates."""
    ds = ListDataset(841)

    # SequentialDistributedSampler: truncates (loses tail)
    seq_per_rank = [list(SequentialDistributedSampler(ds, num_replicas=4, rank=r)) for r in range(4)]
    seq_all = []
    for idxs in seq_per_rank:
        seq_all.extend(idxs)
    assert len(seq_all) == 840, f"Sequential should truncate to 840, got {len(seq_all)}"
    assert 840 not in seq_all, "Sequential should miss index 840 (truncation)"
    print(f"  SequentialDistributedSampler(841,4): {len(seq_all)} indices — TRUNCATES index 840 (bug)")

    # torch DistributedSampler(drop_last=False): pads with duplicates
    if HAVE_TORCH_DS:
        torch_per_rank = [list(TorchDistributedSampler(ds, num_replicas=4, rank=r, shuffle=False, drop_last=False))
                          for r in range(4)]
        torch_all = []
        for idxs in torch_per_rank:
            torch_all.extend(idxs)
        assert len(torch_all) == 844, f"torch DistributedSampler should pad to 844, got {len(torch_all)}"
        assert len(set(torch_all)) == 841, "should have 844 total but only 841 unique (3 padding duplicates)"
        dups = len(torch_all) - len(set(torch_all))
        print(f"  torch DistributedSampler(841,4,drop_last=False): {len(torch_all)} indices, "
              f"{dups} padding DUPLICATES (bug)")

    # New sampler: neither
    new_per_rank = [list(DistributedEvalSampler(ds, num_replicas=4, rank=r, contiguous=False)) for r in range(4)]
    new_all = []
    for idxs in new_per_rank:
        new_all.extend(idxs)
    assert len(new_all) == 841 == len(set(new_all)), f"new sampler: {len(new_all)} total, {len(set(new_all))} unique"
    print(f"  DistributedEvalSampler(841,4): {len(new_all)} indices, 0 duplicates, 0 truncation (FIXED)")
    print("  PASS\n")


def test_contiguous_preserves_doc_locality():
    print("=== test_contiguous_preserves_doc_locality ===")
    """tg_doc needs all samples of one doc on the same rank. Plain contiguous blocks
    split a doc that straddles a count-based boundary, so tg_doc passes group_starts
    (per-document sample-index boundaries) and the sampler partitions DOCUMENTS."""
    import torch
    ds = ListDataset(900)  # 3 docs of 300 samples each
    docs = [(0, 300), (300, 600), (600, 900)]
    group_starts = torch.LongTensor([0, 300, 600, 900])
    blocks = [list(DistributedEvalSampler(ds, num_replicas=4, rank=r, group_starts=group_starts)) for r in range(4)]
    # Every sample covered exactly once across ranks.
    all_idx = sorted(i for blk in blocks for i in blk)
    assert all_idx == list(range(900)), f"coverage gap/dup: {all_idx[:5]}...{all_idx[-5:]}"
    for d_start, d_end in docs:
        ranks_for_doc = set()
        for sample in range(d_start, d_end):
            for r, idxs in enumerate(blocks):
                if sample in idxs:
                    ranks_for_doc.add(r)
        assert len(ranks_for_doc) == 1, (
            f"doc [{d_start},{d_end}) split across ranks {ranks_for_doc} (would break KV cache)")
        print(f"  doc [{d_start}:{d_end}) entirely on rank {ranks_for_doc.pop()}")

    # Uneven docs (200, 400, 100, 200) across 3 ranks: 4 docs / 3 ranks -> ranks
    # get [2,1,1] docs. Verify no doc splits and full coverage.
    ds2 = ListDataset(900)
    gs2 = torch.LongTensor([0, 200, 600, 700, 900])
    blocks2 = [list(DistributedEvalSampler(ds2, num_replicas=3, rank=r, group_starts=gs2)) for r in range(3)]
    all2 = sorted(i for blk in blocks2 for i in blk)
    assert all2 == list(range(900)), "uneven-doc coverage mismatch"
    for d_start, d_end in [(0, 200), (200, 600), (600, 700), (700, 900)]:
        ranks_for_doc = {r for r, idxs in enumerate(blocks2) for s in range(d_start, d_end) if s in idxs}
        assert len(ranks_for_doc) == 1, f"uneven doc [{d_start},{d_end}) split: {ranks_for_doc}"
    print("  (uneven docs: no splits, full coverage)")
    print("  PASS\n")


if __name__ == "__main__":
    test_strided_pattern()
    test_contiguous_blocks_pattern()
    test_len_matches_iter()
    test_invariants_across_configs()
    test_dataloader_end_to_end()
    test_contrast_old_samplers_buggy()
    test_contiguous_preserves_doc_locality()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
