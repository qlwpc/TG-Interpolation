#!/usr/bin/env python3
"""Compatibility entry point for native top-K validation."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from datatools.parse_data.validate_native_topk import main  # noqa: E402


if __name__ == "__main__":
    main()
