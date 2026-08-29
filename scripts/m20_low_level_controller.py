"""Compatibility shim; use :mod:`m20pro_vla.low_level` in new code."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from m20pro_vla.low_level.controller import *  # noqa: F401,F403,E402
