#!/usr/bin/env python3
"""Compatibility entry point; use ``scripts/mujoco/play_m20_mujoco_vla.py``."""
from pathlib import Path
import runpy
import sys

_TARGET = Path(__file__).resolve().parent / "mujoco" / Path(__file__).name
sys.path.insert(0, str(_TARGET.parent))
runpy.run_path(str(_TARGET), run_name="__main__")
