import numpy as np

from scripts.submit_pause_sep_pretrain import (
    choose_microbatch,
    count_token_stream,
    make_batch_plan,
    round_to_multiple,
)


def test_round_to_gpu_multiple_uses_half_up():
    assert round_to_multiple(216.518, 8) == 216
    assert round_to_multiple(272.792, 8) == 272
    assert round_to_multiple(4.0, 8) == 8


def test_microbatch_is_largest_exact_divisor_under_cap():
    assert choose_microbatch(27, 17) == 9
    assert choose_microbatch(34, 17) == 17
    assert choose_microbatch(35, 9) == 7


def test_eight_gpu_pause_batch_plans_are_exact():
    pause1 = make_batch_plan(20_122_051_168, 2048, 8, 17)
    assert (pause1.global_batch_size, pause1.per_device_batch_size) == (216, 27)
    assert (pause1.microbatch_size, pause1.gradient_accumulation_steps) == (9, 3)

    pause2 = make_batch_plan(30_183_076_752, 2049, 8, 17)
    assert (pause2.global_batch_size, pause2.per_device_batch_size) == (272, 34)
    assert (pause2.microbatch_size, pause2.gradient_accumulation_steps) == (17, 2)


def test_microbatch_override_must_divide_per_device_batch():
    try:
        make_batch_plan(30_183_076_752, 2049, 8, 17, microbatch_override=8)
    except ValueError as exc:
        assert "must divide" in str(exc)
    else:
        raise AssertionError("invalid microbatch override was accepted")


def test_token_count_for_raw_uint16(tmp_path):
    path = tmp_path / "tokens.bin"
    path.write_bytes(b"\x00\x00" * 13)
    assert count_token_stream(path, itemsize=2) == 13


def test_token_count_uses_npy_header_shape(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(13, dtype=np.uint16))
    assert path.stat().st_size > 13 * 2
    assert count_token_stream(path, itemsize=2) == 13
