#!/usr/bin/env python3
"""Compatibility entry point for :mod:`datatools.parse_data.tree_to_tg`."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from datatools.parse_data.tree_to_tg import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
