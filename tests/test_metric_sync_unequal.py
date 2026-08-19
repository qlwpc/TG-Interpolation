"""Regression test for the unified ``sync_on_compute`` crash fix.

Reproduces the user-reported failure: when the BLiMP dataset size is not
divisible by the number of ranks, ``DistributedEvalSampler`` gives each rank a
count that differs by at most one. Under torchmetrics' built-in
``sync_on_compute=True`` + ``dist_reduce_fx="sum"``, ``compute()`` performs a
collective that assumes every rank called ``update()`` the same number of
times — so it deadlocks/hangs (appearing as a crash before results).

The fix (Part A of the beam-search BLiMP plan): every metric sets
``sync_on_compute=False`` and does an explicit, count-insensitive reduction at
the top of ``compute()``:
  - tensor-scatter metrics (BLiMPMetric, TGPerplexityDocumentLevelMetric):
    ``_all_reduce_tensor`` (SUM over disjoint sent_id slots).
  - general list-append metrics (ICLMetric, ...): ``_gather_list``
    (all_gather_object + concat).
  - SG: one fixed-size all-reduce over per-suite correct/sample counts, so CUDA
    tensors from different source devices are never object-gathered together.

This test runs two gloo processes (world_size=2) on CPU, gives rank 0 and rank 1
*unequal* update counts (3 vs 2), and asserts ``compute()`` returns promptly
with the correct finite result for both metric shapes.

Run:
    PYTHONPATH=. python tests/test_metric_sync_unequal.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from olmo.eval.downstream import (  # noqa: E402
    BLiMPMetric,
    ICLMetric,
    SyntacticGeneralizationMetric,
    _all_reduce_tensor,
    _gather_list,
    BLiMP_TASK_LIST,
)

VOCAB = "./dataset/bbc-news/TG_GPT2_tokenizer.json"


def _worker(rank: int, world_size: int, out_dir: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29512"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)

    # Unequal partition: 5 samples total -> rank 0 gets 3, rank 1 gets 2.
    # This is exactly the DistributedEvalSampler non-divisible case.
    n_rank0, n_rank1 = 3, 2
    my_n = n_rank0 if rank == 0 else n_rank1
    base = 0 if rank == 0 else n_rank0

    results = {}

    # --- (1) tensor-scatter metric: BLiMPMetric (SENT_SIZE==1 terminal path) ---
    # Test via update_beam (the beam-search path): it scatters a precomputed
    # log-likelihood into self.loglikelihoods by sent_id, then compute()
    # SUM all-reduces the fixed-size tensor. Each rank writes its OWN disjoint
    # sent_id slots (non-overlapping) — exactly the count-insensitive case.
    pair_per_task = 2
    metric = BLiMPMetric(
        vocab_path=VOCAB,
        dataset_name="terminal",
        device_eval_batch_size=1,
        dataset_length=len(BLiMP_TASK_LIST) * 2 * pair_per_task,
        samples_per_sent=1,
        pair_per_task=pair_per_task,
        tree_eval_type="default",
    )
    metric.to(torch.device("cpu"))
    for i in range(my_n):
        sid = base + i
        # update_beam needs only sent_id; LL is a log-prob (stored as -LL).
        metric.update_beam({"sent_id": torch.tensor(sid)}, torch.tensor([float(-(sid + 1))]))
    blimp_acc = metric.compute()
    results["blimp_overall_finite"] = bool(
        torch.isfinite(torch.tensor(float(blimp_acc["overall/overall"])))
    )
    # Verify the all-reduce reconstructed the disjoint writes: slot sid should
    # hold -LL = (sid+1) on every rank after compute()'s all_reduce.
    # (loglikelihoods[sid] = stored -LL = sid+1; check a couple of slots.)
    slot0 = float(metric.loglikelihoods[0].item())
    slot3 = float(metric.loglikelihoods[3].item())
    results["blimp_slot0_ok"] = abs(slot0 - 1.0) < 1e-4   # rank0 wrote sid0 -> -LL=1
    results["blimp_slot3_ok"] = abs(slot3 - 4.0) < 1e-4   # rank1 wrote sid3 -> -LL=4

    # --- (2) list-append metric: ICLMetric ---
    icl = ICLMetric(metric_type="acc", vocab_path=VOCAB)
    icl.to(torch.device("cpu"))
    for i in range(my_n):
        doc_id = base + i
        batch = {
            "doc_id": torch.tensor([doc_id]),
            "cont_id": torch.tensor([0]),
            "label_id": torch.tensor([0]),
            "ctx_len": torch.tensor([2]),
            "cont_len": torch.tensor([1]),
            "continuation": torch.tensor([[7]]),
            "cont_str_len": torch.tensor([1]),
            "cont_byte_len": torch.tensor([1]),
        }
        logits = torch.zeros(1, 3, 50320)
        logits[0, 1, 7] = 0.0  # continuation position logit for token 7
        icl.update(batch, logits)
    icl_result = icl.compute()
    results["icl_finite"] = all(
        torch.isfinite(torch.tensor(float(v))).item() for v in icl_result.values()
    )
    # After _gather_list in compute(), every rank sees the full 5-item list.
    results["icl_total_items"] = len(icl.loglikelihoods)

    # --- (3) SG metric: fixed-size correct/count reduction ---
    # Both ranks contribute to the same suite with unequal update counts. Rank
    # 0 contributes three correct results and rank 1 contributes two incorrect
    # results, so the global Gross_Syntactic_Expectation accuracy is 3/5.
    sg = SyntacticGeneralizationMetric()
    sg.to(torch.device("cpu"))
    correct_scores = {
        "sub_no-matrix": 2.0,
        "no-sub_no-matrix": 1.0,
        "sub_matrix": 1.0,
        "no-sub_matrix": 2.0,
    }
    incorrect_scores = {
        "sub_no-matrix": 1.0,
        "no-sub_no-matrix": 2.0,
        "sub_matrix": 2.0,
        "no-sub_matrix": 1.0,
    }
    for _ in range(my_n):
        sg.update("subordination", correct_scores if rank == 0 else incorrect_scores)
    sg_result = sg.compute()
    results["sg_gross_accuracy"] = sg_result["Gross_Syntactic_Expectation"]
    results["sg_avg_accuracy"] = sg_result["avg"]
    # compute() must leave the rank-local tensor list untouched; gathering it
    # would reintroduce mixed cuda:N devices under NCCL.
    results["sg_state_stays_local"] = len(sg.Gross_Syntactic_Expectation) == my_n

    # --- (4) TGPerplexityDocumentLevelMetric: global sent_id scatter ---
    # Multi-rank docppl: each rank processes whole documents, so its sent_ids
    # index DISTINCT rows of the [n_sent, SENT_SIZE] tensor. update() scatters by
    # batch["sent_id"] (1-based -> row = sid//SENT_SIZE - 1); compute() SUM
    # all-reduces. Verify disjoint writes reconstruct correctly with unequal
    # per-rank update counts.
    from olmo.eval.downstream import TGPerplexityDocumentLevelMetric
    SENT_SIZE = 300
    n_sent_total = 6  # 6 sentences -> tensor shape (6, 300); rank1 index 1200 -> row4
    doc_metric = TGPerplexityDocumentLevelMetric(
        vocab_path=VOCAB,
        term_length=[10] * n_sent_total,
        device_eval_batch_size=100,
        dataset_length=n_sent_total * SENT_SIZE,
        samples_per_sent=SENT_SIZE,
    )
    doc_metric.to(torch.device("cpu"))
    # Rank 0 owns flat indices 0..599 (sentences 0,1); rank 1 owns 1200..1799
    # (sentences 4,5). Unequal counts: rank0 writes 6 trees, rank1 writes 6.
    # index i -> row = i//300, col = i%300 (one unique slot per tree).
    my_indices = list(range(0, 600)) if rank == 0 else list(range(1200, 1800))
    for i in my_indices:
        ce = torch.tensor(float(i + 1))  # scalar loss per tree
        doc_metric.update({"index": torch.LongTensor([i])}, ce)
    # Unwritten slots stay 0; SUM all-reduces disjoint rows.
    doc_ppl = doc_metric.compute()
    results["docppl_finite"] = bool(torch.isfinite(doc_ppl))
    # After compute()'s all_reduce, every rank sees the full tensor.
    # index 5 -> row0,col5 = 6; index 1202 -> row4,col2 = 1203.
    results["docppl_row0_ok"] = abs(float(doc_metric.loglikelihoods[0, 5].item()) - 6.0) < 1e-3
    results["docppl_row2_ok"] = abs(float(doc_metric.loglikelihoods[4, 2].item()) - 1203.0) < 1e-3
    # Unwritten slot (row0, col 0 — index 0 written by rank0 -> value 1, not 0).
    # Pick a truly unwritten slot: row1 (indices 300..599 all written by rank0),
    # so use row3 col0 (indices 900..1199, owned by nobody).
    results["docppl_unwritten_zero"] = float(doc_metric.loglikelihoods[3, 0].item()) == 0.0

    # --- (5) helper unit checks ---
    t = torch.arange(4, dtype=torch.float32) * (rank + 1)
    t_red = _all_reduce_tensor(t)
    # rank0 = [0,1,2,3], rank1 = [0,2,4,6] -> SUM = [0,3,6,9]
    results["all_reduce_sum_ok"] = bool(
        torch.allclose(t_red, torch.tensor([0.0, 3.0, 6.0, 9.0]))
    )
    g = _gather_list([rank, rank + 100])
    results["gather_count"] = len(g)
    results["gather_has_both"] = (0 in g) and (1 in g) and (100 in g) and (101 in g)

    # Write results to a per-rank file (reliable across spawn; mp.Queue's feeder
    # thread can drop data if the process exits immediately after put).
    import json as _json
    out_path = os.path.join(out_dir, f"rank{rank}.json")
    with open(out_path, "w") as f:
        _json.dump(results, f)
    dist.barrier()
    dist.destroy_process_group()


def main():
    if not os.path.exists(VOCAB):
        print("SKIP: tokenizer not found (run from repo root).")
        return
    world_size = 2
    ctx = mp.get_context("spawn")
    out_dir = tempfile.mkdtemp(prefix="blimp_sync_")
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=_worker, args=(rank, world_size, out_dir))
        p.start()
        procs.append(p)
    # Wait for both to finish (hard timeout — if the sync regressed into a
    # deadlock, compute() hangs forever and this fires).
    deadline = time.time() + 120
    for p in procs:
        remaining = max(1.0, deadline - time.time())
        p.join(timeout=remaining)
        if p.is_alive():
            p.terminate()
            p.join()

    import json as _json
    all_results = {}
    for rank in range(world_size):
        out_path = os.path.join(out_dir, f"rank{rank}.json")
        if os.path.exists(out_path):
            with open(out_path) as f:
                all_results[rank] = _json.load(f)

    if len(all_results) < world_size:
        print(f"FAIL: only {len(all_results)}/{world_size} ranks wrote results — deadlocked (the bug).")
        sys.exit(1)

    ok = True
    for rank in sorted(all_results):
        r = all_results[rank]
        print(f"[rank {rank}] {r}")
        ok = ok and r.get("blimp_overall_finite")
        ok = ok and r.get("blimp_slot0_ok")
        ok = ok and r.get("blimp_slot3_ok")
        ok = ok and r.get("icl_finite")
        ok = ok and abs(r.get("sg_gross_accuracy", -1.0) - 0.6) < 1e-8
        ok = ok and abs(r.get("sg_avg_accuracy", -1.0) - 0.1) < 1e-8
        ok = ok and r.get("sg_state_stays_local")
        ok = ok and r.get("all_reduce_sum_ok")
        ok = ok and r.get("gather_count") == 4
        ok = ok and r.get("gather_has_both")
        # docppl global sent_id scatter (multi-rank disjoint rows)
        ok = ok and r.get("docppl_finite")
        ok = ok and r.get("docppl_row0_ok")
        ok = ok and r.get("docppl_row2_ok")
        ok = ok and r.get("docppl_unwritten_zero")
    # Each rank's view should hold the full gathered 5 items.
    ok = ok and all(all_results[r]["icl_total_items"] == 5 for r in all_results)

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
