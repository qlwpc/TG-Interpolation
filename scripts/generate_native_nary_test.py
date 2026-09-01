#!/usr/bin/env python3
"""Compatibility entry point for model-native GPST/Pushdown top-K data."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from datatools.parse_test_docppl_data.generate_native_topk import *  # noqa: E402,F401,F403
from datatools.parse_test_docppl_data.generate_native_topk import (  # noqa: E402
    _decode_and_adapt,
    _prepare_shard_arrays,
    _tree_terminals_and_record,
    _write_result,
)


if __name__ == "__main__":
    main()
