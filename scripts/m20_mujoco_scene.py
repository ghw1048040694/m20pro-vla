#!/usr/bin/env python3
"""Compatibility entry point; use ``scripts/mujoco/m20_mujoco_scene.py``."""
from pathlib import Path
import runpy
import sys

_TARGET = Path(__file__).resolve().parent / "mujoco" / Path(__file__).name
sys.path.insert(0, str(_TARGET.parent))
runpy.run_path(str(_TARGET), run_name="__main__")
