"""Stable, exact candidate identity after a model's tree projection."""
from __future__ import annotations

import numpy as np


def stable_unique_indices(sequences):
    """Return first original indices and original-to-unique mapping.

    Callers must remove physical padding slots and project *before* calling.
    Equality is over the complete unpadded token sequence, including BOS/EOS.
    Python's bytes dictionary resolves hash collisions with exact equality.
    """
    seen, first, inverse = {}, [], []
    for index, sequence in enumerate(sequences):
        array = np.asarray(sequence)
        if array.ndim != 1 or array.dtype.kind not in 'iu':
            raise ValueError('candidate must be a one-dimensional integer sequence')
        key = array.astype('<i8', copy=False).tobytes()
        if key not in seen:
            seen[key] = len(first)
            first.append(index)
        inverse.append(seen[key])
    if not first:
        raise ValueError('no valid candidates')
    return np.asarray(first, dtype=np.int16), np.asarray(inverse, dtype=np.int16)


def deduplicate_samples(samples):
    first, inverse = stable_unique_indices(s['input_ids'] for s in samples)
    # Equal inputs with different target masks would not define equal events.
    for original, unique in enumerate(inverse):
        representative = samples[int(first[unique])]
        if ('label_mask' in samples[original]) != ('label_mask' in representative):
            raise ValueError('duplicate candidate has inconsistent target mask')
        if 'label_mask' in representative and not np.array_equal(
                samples[original]['label_mask'], representative['label_mask']):
            raise ValueError('duplicate candidate has inconsistent target mask')
    return [samples[int(i)] for i in first], first, inverse
