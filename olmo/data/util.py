from typing import Generator, List, NamedTuple, Iterator, Union

import numpy as np
import torch
from nltk import Tree
import re

import torch.distributed as dist
from torch.utils.data import Sampler


def find_end_first_consecutive_true(arr: np.ndarray) -> int:
    """Function to find the end position of the first consecutive sequence of True in an array."""
    if not arr[0]:
        return 0

    prog = np.cumsum(arr)
    if prog[-1] == len(arr):
        return len(arr)

    true_locs = np.where(prog[:-1:] == prog[1::])[0]

    return true_locs[0] + 1


def find_start_last_consecutive_true(arr: np.ndarray) -> int:
    """Function to find the start position of the last consecutive sequence of True in an array."""
    reverse = find_end_first_consecutive_true(arr[::-1])
    return len(arr) - reverse if reverse > 0 else -1


def group_consecutive_values(arr: np.ndarray, stepsize: int = 1) -> List[np.ndarray]:
    """Function to group consecutive values in an array."""
    return np.split(arr, np.where(np.diff(arr) != stepsize)[0] + 1)


class RepetitionTuple(NamedTuple):
    """Tuple to store information about a periodic sequence."""

    start: int
    end: int
    period: int
    times: int


def find_periodic_sequences(
    arr: np.ndarray, max_period: int, min_period: int = 1, mask_value: int = -1
) -> Generator[RepetitionTuple, None, None]:
    """Function to find periodic sequences in an array.

    This function sweeps through the array and checks for sequences of length
    [min_period, max_period] that repeat at least 3 times. To do so, it
    reshape the array into a matrix with `period` columns and checks if each
    row is equal to the previous row. Blocks of repeating rows indicates repeating
    sequences.

    Because there's no guarantee that the sequences start at the beginning of each
    row, it can only detect sequences that repeat at least 3 times. To account
    for the fact that sequences may not start at the beginning of each row (or
    end at the end of each row), we check the end of the previous row and the
    start of the next row to determine the actual start and end positions of the
    sequence.

    Args:
        arr (np.ndarray): The array to search for periodic sequences.
        max_period (int): The maximum period to check for.
        min_period (int, optional): The minimum period to check for. Defaults to 1.
        mask_value (int, optional): The value to use to pad the array. Defaults to -1.
    """
    # make sure the mask_value is not in the array
    if (arr == mask_value).sum() > 0:
        raise ValueError("`mask_value` is in the array")

    # no since we can only detect sequences that repeat at least 3 times,
    # there is no point in checking for periods greater than 1/3 of the length
    max_period = min(max_period, len(arr) // 3)

    for period in range(min_period, max_period + 1):
        # pad the array so that it can be reshaped into a matrix matching the period
        padded_arr = np.pad(arr, (0, period - (len(arr) % period)), constant_values=mask_value)
        shaped_arr = padded_arr.reshape(-1, period)

        # find rows that are equal to the previous  row; these are the possibly-periodic sequences
        is_equal_to_prev_row = shaped_arr == np.roll(shaped_arr, shift=1, axis=0)
        rows_with_period, *_ = np.where(is_equal_to_prev_row.all(axis=1))

        # no sequences found with this period
        if len(rows_with_period) == 0:
            continue

        # this finds the start and end positions of the sequences with period `period`
        where_true_consecutive = group_consecutive_values(rows_with_period)

        for sequence in where_true_consecutive:
            start_row = sequence[0]
            end_row = sequence[-1]

            # we check if any value at the end of the previous row is True, e.g.:
            #     [[False, False, True, True]
            #      [True, True, True, True]]
            # (in the case above, start offset is 2). If so, we subtract that from the
            # period to get the actual start offset.
            start_offset = find_start_last_consecutive_true(is_equal_to_prev_row[start_row - 1])
            start_offset = period - start_offset if start_offset > 0 else 0

            # same idea as above, we want to compute offset. Only difference is that
            # `find_end_first_consecutive_true` already returns the offset, so we don't
            # need to subtract from the period.
            end_offset = find_end_first_consecutive_true(is_equal_to_prev_row[end_row + 1])

            # because we are always comparing with preceding row in
            # `is_equal_to_prev_row`, we need to subtract 1 from the row number
            start_pos = (start_row - 1) * period - start_offset

            # note that the end position is exclusive
            end_pos = ((end_row + 1) * period) + end_offset

            out = RepetitionTuple(
                start=start_pos, end=end_pos, period=period, times=(end_pos - start_pos) // period
            )
            if out.times > 2:
                # cannot accurately determine the period of a sequence that repeats
                # less than 3 times with this algorithm
                yield out


def get_document_lengths(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    doc_boundaries = torch.cat(
        [
            torch.tensor([-1], dtype=torch.int32),
            (input_ids == eos_token_id).nonzero(as_tuple=True)[0].to(dtype=torch.int32),
            torch.tensor([] if input_ids[-1] == eos_token_id else [input_ids.shape[0] - 1], dtype=torch.int32),
        ]
    )
    return doc_boundaries[1:] - doc_boundaries[:-1]


def _get_bracket_mapping_from_tokenizer(tokenizer) -> bool:
    """Determine if bracket mapping should be applied based on tokenizer config.

    Both GPT-2 and Qwen3 tokenizers have '<-LRB->' added as special tokens,
    so auto-detection by vocabulary lookup is unreliable. Instead, read
    the ``use_bracket_mapping`` flag from the tokenizer if available,
    otherwise default to False (safe for GPT-2).
    """
    if hasattr(tokenizer, 'use_bracket_mapping'):
        return bool(tokenizer.use_bracket_mapping)
    return False


def pformat_flat(self, nodesep="", parens="()", quotes=False,
                 use_bracket_mapping=True):
    childstrs = []
    for child in self:
        if isinstance(child, Tree):
            childstrs.append(pformat_flat(child, nodesep, parens, quotes,
                                          use_bracket_mapping=use_bracket_mapping))
        elif isinstance(child, tuple):
            childstrs.append("/".join(child))
        elif isinstance(child, str) and not quotes:
            if use_bracket_mapping:
                mapping = {
                    "-LRB-": "(",
                    "-RRB-": ")",
                    "-LCB-": "{",
                    "-RCB-": "}",
                    "-LSB-": "[",
                    "-RSB-": "]",
                    "Ċ": "\n",
                }
                out = mapping[child] if child in mapping else child
                for old, new in mapping.items():
                    out = out.replace(old, " " + new)
            else:
                out = child
            return " " + out
        else:
            childstrs.append(repr(child))
    if isinstance(self._label, str):
        if self._label == "qlwpcRegen":
            return "".join(childstrs)
        else:
            return "<{}{}{}>{}<{}{}>".format(
                parens[0],
                self._label,
                nodesep,
                "".join(childstrs),
                self._label,
                parens[1],
            )
    else:
        raise NotImplementedError


def convert_TG_format(input: str, use_bracket_mapping: bool = True) -> str:
    line = "(qlwpcRegen " + input.strip() + ")"
    tree = Tree.fromstring(line, remove_empty_top_bracketing=False)
    outputstr = pformat_flat(tree, use_bracket_mapping=use_bracket_mapping)
    return outputstr


def encode_TG_string(tokenizer, input: str, string_with_POS_tags: bool = True,
                     use_bracket_mapping: bool = None) -> np.ndarray:
    if use_bracket_mapping is None:
        use_bracket_mapping = _get_bracket_mapping_from_tokenizer(tokenizer)
    if string_with_POS_tags:
        TG_str = convert_TG_format(input, use_bracket_mapping=use_bracket_mapping)
    else:
        TG_str = input
    ids = np.array(tokenizer.encode(TG_str, add_special_tokens=False))
    return ids

def _parse_pause_spec(pause_num: Union[int, str]) -> "tuple[int, int]":
    """Parse a pause specification into ``(p, q)``.

    ``(p, q)`` means "insert ``p`` pause tokens after every ``q`` real tokens".

    Accepted forms:
        - int ``N``                  -> ``(N, 1)``
        - ``"N"`` / ``"pauseN"``     -> ``(N, 1)``
        - ``"p/q"`` / ``"pausep/q"`` -> ``(p, q)``

    A trailing ``"_label"`` tag (e.g. ``"pause1/2_label"``) is tolerated.

    Args:
        pause_num: int or string pause specification.

    Returns:
        ``(p, q)`` with ``p >= 0`` and ``q >= 1``.

    Raises:
        ValueError: when the value cannot be parsed or ``q < 1``.
    """
    if isinstance(pause_num, (int, np.integer)):
        p, q = int(pause_num), 1
    else:
        s = str(pause_num).strip()
        if s.startswith("pause"):
            s = s[5:]
        if s.endswith("_label"):
            s = s[: -len("_label")]
        if "/" in s:
            num_str, den_str = s.split("/", 1)
            p, q = int(num_str), int(den_str)
        else:
            p, q = int(s), 1
    if q < 1:
        raise ValueError(f"pause denominator must be >= 1, got {q} from {pause_num!r}")
    if p < 0:
        raise ValueError(f"pause numerator must be >= 0, got {p} from {pause_num!r}")
    return p, q


def pause_input_ids(input_ids, pause_token_id: int = None, pause_num: Union[int, str] = 1):
    """Interleave pause tokens into a 1-D token sequence.

    For a spec ``(p, q)``, every block of ``q`` real tokens is followed by
    ``p`` pause tokens. A trailing partial block (fewer than ``q`` real tokens)
    is emitted with no pauses appended, so the expansion factor is uniform only
    when ``len(input_ids)`` is divisible by ``q``.

    Args:
        input_ids: 1-D list / ``np.ndarray`` / ``torch.Tensor`` of real token ids.
        pause_token_id: id placed in pause slots. When ``None``, each real token
            is broadcast to the pause slots of its own group (used for masks)
            instead of inserting a dedicated pause id.
        pause_num: int ``N``, ``"N"``/``"pauseN"`` (``N`` pauses per token), or
            ``"pausep/q"`` (``p`` pauses per ``q`` tokens).

    Returns:
        Interleaved sequence with the same type as ``input_ids``.

    Raises:
        AssertionError: when ``input_ids`` is not 1-D.
        NotImplementedError: when ``input_ids`` type is unsupported.
    """
    p, q = _parse_pause_spec(pause_num)
    n_real = len(input_ids)
    init_token = pause_token_id if pause_token_id is not None else 0
    n_full = n_real // q          # complete (q real + p pause) blocks
    remainder = n_real % q        # trailing real tokens, no pauses after them
    out_len = n_full * (q + p) + remainder

    if isinstance(input_ids, list):
        if pause_token_id is not None:
            out = [init_token] * out_len
            ri = 0
            for b in range(n_full):
                pos = b * (q + p)
                for j in range(q):
                    out[pos + j] = input_ids[ri]
                    ri += 1
            pos = n_full * (q + p)
            for j in range(remainder):
                out[pos + j] = input_ids[ri]
                ri += 1
            return out
        # pause_token_id is None: broadcast each group's last real token to its pauses
        out = []
        for b in range(n_full):
            base = b * q
            out.extend(input_ids[base:base + q - 1])
            last = input_ids[base + q - 1]
            out.append(last)
            out.extend([last] * p)
        out.extend(input_ids[n_full * q:n_full * q + remainder])
        return out

    if isinstance(input_ids, np.ndarray):
        assert len(input_ids.shape) == 1
        if pause_token_id is not None:
            out = np.full(out_len, init_token, dtype=input_ids.dtype)
            if n_full > 0:
                starts = np.arange(n_full) * (q + p)
                real_pos = (starts[:, None] + np.arange(q)).reshape(-1)
                out[real_pos] = input_ids[: n_full * q]
            if remainder:
                out[n_full * (q + p):] = input_ids[n_full * q:]
            return out
        repeats = np.ones(n_real, dtype=np.int64)
        repeats[q - 1::q] += p
        owner = np.repeat(np.arange(n_real), repeats)
        return input_ids[owner]

    if isinstance(input_ids, torch.Tensor):
        assert len(input_ids.shape) == 1
        device = input_ids.device
        if pause_token_id is not None:
            out = torch.full((out_len,), init_token, dtype=input_ids.dtype, device=device)
            if n_full > 0:
                starts = torch.arange(n_full, device=device) * (q + p)
                real_pos = (starts[:, None] + torch.arange(q, device=device)).reshape(-1)
                out[real_pos] = input_ids[: n_full * q]
            if remainder:
                out[n_full * (q + p):] = input_ids[n_full * q:]
            return out
        repeats = torch.ones(n_real, dtype=torch.long, device=device)
        repeats[q - 1::q] += p
        return torch.repeat_interleave(input_ids, repeats)

    raise NotImplementedError(f"Unknown pause input ids type: {type(input_ids)}")


def pause_label_mask(expanded_len: int, p: int, q: int) -> "np.ndarray":
    """Boolean label mask over a ``pause_input_ids`` output of length ``expanded_len``.

    Returns a 1-D ``np.bool_`` array aligned 1:1 with an expanded sequence, where
    ``True`` marks a real-token position (contributes to the loss) and ``False``
    marks a pause-token position (masked out of the loss).

    For spec ``(p, q)`` every complete block of ``q`` real tokens is followed by
    ``p`` pause tokens, and a trailing partial block (fewer than ``q`` real
    tokens, so no pauses) is entirely ``True``. This mirrors
    :func:`pause_input_ids` exactly, so the ``False`` positions line up with the
    pause slots it inserts -- regardless of whether ``pause_input_ids`` was
    called with a dedicated ``pause_token_id`` or with ``None`` (repeat mode).

    After ``Trainer.get_labels`` left-shifts ``input_ids`` by one (``[..., 1:]``),
    masking ``input_ids[j]`` where this mask is ``False`` causes the loss at logit
    position ``i`` (which targets ``input_ids[i+1]``) to be masked exactly when the
    *next* token is a pause token -- i.e. the loss is set only where the next token
    is a real token.

    Args:
        expanded_len: Length of the expanded sequence. Need not be a whole number
            of ``(q + p)`` blocks; the trailing partial block is handled.
        p: Number of pause tokens per block (``p >= 0``; ``0`` yields all-``True``).
        q: Number of real tokens per block (``q >= 1``).

    Returns:
        ``np.ndarray`` of shape ``(expanded_len,)`` and dtype ``np.bool_``.

    Raises:
        ValueError: when ``q < 1`` or ``p < 0``.
    """
    if q < 1:
        raise ValueError(f"pause denominator must be >= 1, got {q}")
    if p < 0:
        raise ValueError(f"pause numerator must be >= 0, got {p}")
    if p == 0:
        return np.ones(expanded_len, dtype=np.bool_)

    # Build the mask for one complete (q real + p pause) block, then tile it
    # across the full blocks and handle the trailing partial block.
    block = np.concatenate([
        np.ones(q, dtype=np.bool_),       # real-token slots
        np.zeros(p, dtype=np.bool_),      # pause slots
    ])
    period = q + p
    n_full = expanded_len // period         # complete blocks fully contained
    remainder = expanded_len - n_full * period
    mask = np.tile(block, n_full)
    if remainder:
        # Trailing partial block: min(remainder, q) real slots then any pause slots.
        n_real_tail = min(remainder, q)
        tail = np.zeros(remainder, dtype=np.bool_)
        tail[:n_real_tail] = True
        mask = np.concatenate([mask, tail])
    return mask


def pause_spec_from_grammar_type(grammar_type: str) -> "tuple[int, int]":
    """Parse a ``transformer_grammar_type`` into a rational pause spec ``(p, q)``.

    ``(p, q)`` means "insert ``p`` pause tokens after every ``q`` real tokens".
    Returns ``(0, 1)`` (no pauses) for non-``pause`` grammar types.

    A bare ``"pause"`` (no number) is treated as ``(1, 1)``.
    """
    if grammar_type[:5] == "pause":
        spec = grammar_type if grammar_type != "pause" else "pause1"
        return _parse_pause_spec(spec)
    return (0, 1)


def is_pause_label(grammar_type: str) -> bool:
    """Whether ``grammar_type`` requests the pause label-mask ("next token is real").

    ``True`` only for ``pause`` grammar types carrying the ``"_label"`` suffix
    (e.g. ``"pause1/2_label"``). For those, :func:`pause_input_ids` still expands
    the sequence identically to the bare spec (the suffix is stripped by
    :func:`_parse_pause_spec`), but the loss is additionally masked at positions
    whose *next* token is a pause token -- i.e. the model only trains a loss where
    the next token is a real token. Plain pause specs (``"pause1/2"`` etc.) and
    non-pause grammar types return ``False`` (no label masking, current behavior).
    """
    return grammar_type[:5] == "pause" and grammar_type.endswith("_label")


def pause_expanded_len(real_len: int, p: int, q: int) -> int:
    """Length of ``pause_input_ids`` output for ``real_len`` real tokens, spec ``(p, q)``.

    Equals ``real_len + (real_len // q) * p``: each complete block of ``q`` real
    tokens is followed by ``p`` pauses; a trailing partial block emits no pauses.
    This is also the expanded position of real token ``real_len`` (i.e. one past
    the last real token's block), so it gives the split point after ``real_len``
    real tokens regardless of divisibility.

    Raises:
        ValueError: when ``q < 1`` (would otherwise silently divide by zero).
    """
    if q < 1:
        raise ValueError(f"pause denominator must be >= 1, got {q}")
    return real_len + (real_len // q) * p


def pause_trailing_trim(real_len: int, p: int, q: int) -> int:
    """Number of trailing pause tokens after the last real token, spec ``(p, q)``.

    ``p`` if the last real token completes a block (``real_len % q == 0``), else
    ``0`` (trailing partial block emits no pauses).
    """
    return p if real_len % q == 0 else 0


def extract_real_tokens(paused, p: int, q: int, skip_first: bool = False):
    """Reverse of :func:`pause_input_ids` for ``pause_token_id=None`` expansions.

    ``pause_input_ids`` with ``pause_token_id=None`` places real token ``j`` at
    expanded position ``j + (j // q) * p`` (the trailing ``p`` slots of each
    complete ``q``-block repeat the block's last real token). This inverts that
    mapping: it walks real-token indices ``j = 0, 1, 2, ...`` and emits
    ``paused[pos_j]`` while ``pos_j < len(paused)``, i.e. it keeps exactly the
    real-token positions and drops the pause slots.

    Works for any ``(p, q)`` (e.g. pause1/2 ``(1,2)``, pause2 ``(2,1)``,
    pause3 ``(3,1)``) and any ``len(paused)`` — the sequence need not be a whole
    number of blocks. ``skip_first`` drops the leading BOS real token (the
    convention in ``summarization_eval_step``).
    """
    if q < 1:
        raise ValueError(f"pause denominator must be >= 1, got {q}")
    n = len(paused)
    start = 1 if skip_first else 0
    out = []
    j = start
    while True:
        pos = j + (j // q) * p
        if pos >= n:
            break
        out.append(paused[pos])
        j += 1
    if isinstance(paused, np.ndarray):
        return np.array(out, dtype=paused.dtype)
    if isinstance(paused, torch.Tensor):
        return torch.tensor(out, dtype=paused.dtype, device=paused.device)
    return out


class SequentialDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, 
                 shuffle=False, drop_last=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = len(self.dataset) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas
        
    def __iter__(self):
        # 确定当前rank的起始位置
        start_idx = self.rank * self.num_samples
        end_idx = start_idx + self.num_samples
        
        # 生成当前rank的索引序列
        indices = list(range(start_idx, end_idx))
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedEvalSampler(Sampler):
    """Distributed sampler for evaluation that neither pads nor drops.

    Unlike ``torch.utils.data.distributed.DistributedSampler`` (which pads the
    dataset to a multiple of ``num_replicas`` by duplicating the first samples,
    corrupting ``sum/len``-style metrics with duplicate results) and unlike
    ``SequentialDistributedSampler`` (which truncates the tail, silently
    skipping unevaluated cases), this sampler partitions the dataset so that
    every sample is evaluated exactly once across all ranks. Counts differ by
    at most one across ranks; no padding, no truncation.

    Two partitioning modes:
      - ``contiguous=False`` (default): strided — rank ``r`` gets indices
        ``r, r+W, r+2W, ...``. Use for metrics with no order dependence
        (SG, Rouge, ICL, beam_search_icl).
      - ``contiguous=True``: contiguous blocks — rank ``r`` gets a single
        contiguous range ``[start_r, start_r + size_r)``. Use for metrics that
        rely on local order (tg_doc KV cache accumulates within a doc; tg_sent
        assumes sequential sent_id arrival). Unlike
        ``SequentialDistributedSampler``, the tail is NOT truncated: the first
        ``N % W`` ranks get one extra sample so every sample is covered.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False,
                 drop_last=False, contiguous=False, group_starts=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.contiguous = contiguous
        self.epoch = 0
        n = len(self.dataset)
        self.group_starts = group_starts
        if group_starts is not None:
            # Doc-aware partitioning: ``group_starts`` is a sorted array of length
            # n_groups+1 where group g spans sample indices [group_starts[g],
            # group_starts[g+1]). We partition the GROUPS (not samples) across
            # ranks so every group lands entirely on one rank — required by tg_doc,
            # whose KV cache accumulates across the whole document and whose
            # metric scatters one [n_sent, SENT_SIZE] row per sentence (a group of
            # SENT_SIZE trees). Splitting a group would break both. The first
            # (n_groups % W) ranks get one extra group; within a rank, groups are
            # emitted in ascending index order so KV accumulation stays causal.
            n_groups = len(group_starts) - 1
            g_base = n_groups // num_replicas
            g_rem = n_groups % num_replicas
            if rank < g_rem:
                self._group_start = rank * (g_base + 1)
                self._group_end = self._group_start + g_base + 1
            else:
                self._group_start = g_rem * (g_base + 1) + (rank - g_rem) * g_base
                self._group_end = self._group_start + g_base
            self.num_samples = int(group_starts[self._group_end] - group_starts[self._group_start])
            self.total_size = n  # exact, every sample covered once
        else:
            # Each rank gets ceil or floor of n/W; first (n % W) ranks get one extra.
            base = n // num_replicas
            rem = n % num_replicas
            if rank < rem:
                self.num_samples = base + 1
            else:
                self.num_samples = base
            self.total_size = n  # exact, no padding

    def __iter__(self):
        n = len(self.dataset)
        if self.group_starts is not None:
            # Emit this rank's groups in order; flatten each group's contiguous
            # sample range. Group boundaries are document boundaries (and thus
            # sentence and SENT_SIZE boundaries), so the per-sentence 300-sync in
            # TG_doc_eval_step and the metric's row scatter stay aligned.
            gs = self.group_starts
            # Groups are adjacent ranges covering the dataset, so their union
            # is one range. Avoid materializing millions of Python integers for
            # large gold-K evaluations.
            return iter(
                range(int(gs[self._group_start]), int(gs[self._group_end]))
            )
        elif self.contiguous:
            base = n // self.num_replicas
            rem = n % self.num_replicas
            # first `rem` ranks get base+1
            start = self.rank * base + min(self.rank, rem)
            end = start + self.num_samples
            return iter(range(start, end))
        else:
            return iter(range(self.rank, n, self.num_replicas))

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
