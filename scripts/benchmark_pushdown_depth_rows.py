"""Microbenchmark full versus incremental Pushdown depth tape construction."""

import time

import torch

from olmo.pushdown import compute_depth_matrix_gpu, compute_last_depth_row_gpu


def measure(fn, spans: torch.Tensor, length: int, repeats: int) -> float:
    for _ in range(3):
        fn(spans, length)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn(spans, length)
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start) / repeats


def main() -> None:
    device = torch.device("cuda")
    batch, n_spans, length = 6, 700, 1200
    generator = torch.Generator(device=device).manual_seed(7)
    left = torch.randint(0, length, (batch, n_spans), generator=generator, device=device)
    right = torch.randint(0, length, (batch, n_spans), generator=generator, device=device)
    left, right = torch.minimum(left, right), torch.maximum(left, right)
    spans = torch.stack([left, right, right], dim=-1)
    full = compute_depth_matrix_gpu(spans, length)[:, -1:, :]
    incremental = compute_last_depth_row_gpu(spans, length)
    if not torch.equal(full, incremental):
        raise AssertionError("incremental depth row differs from full matrix")
    full_ms = measure(compute_depth_matrix_gpu, spans, length, repeats=10)
    row_ms = measure(compute_last_depth_row_gpu, spans, length, repeats=100)
    print(f"batch={batch} spans={n_spans} length={length}")
    print(f"full_ms={full_ms:.3f} row_ms={row_ms:.3f} speedup={full_ms / row_ms:.1f}x")

    closed = [
        [(int(left[b, i]), int(right[b, i])) for i in range(n_spans)]
        for b in range(batch)
    ]

    def old_collate():
        result = torch.full((batch, n_spans, 3), -1, dtype=torch.long, device=device)
        for b, beam_spans in enumerate(closed):
            for i, (span_left, span_right) in enumerate(beam_spans):
                result[b, i] = torch.tensor(
                    [span_left, span_right, span_right], dtype=torch.long, device=device
                )
        return result

    def new_collate():
        rows = [
            [[span_left, span_right, span_right] for span_left, span_right in beam_spans]
            for beam_spans in closed
        ]
        return torch.tensor(rows, dtype=torch.long, device=device)

    torch.cuda.synchronize()
    start = time.perf_counter()
    old = old_collate()
    torch.cuda.synchronize()
    old_collate_ms = 1000.0 * (time.perf_counter() - start)
    start = time.perf_counter()
    new = new_collate()
    torch.cuda.synchronize()
    new_collate_ms = 1000.0 * (time.perf_counter() - start)
    if not torch.equal(old, new):
        raise AssertionError("batched span collation differs from indexed collation")
    print(
        f"old_collate_ms={old_collate_ms:.3f} new_collate_ms={new_collate_ms:.3f} "
        f"speedup={old_collate_ms / new_collate_ms:.1f}x"
    )


if __name__ == "__main__":
    main()
