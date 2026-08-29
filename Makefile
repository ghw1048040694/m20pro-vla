.DEFAULT_GOAL := help

PYTHON ?= python
M20_CLI ?= PYTHONPATH=src $(PYTHON) -m m20pro_vla.cli

.PHONY: help test mujoco-smoke low-level-gate rough-terrain-gate ros-build

help:
	@printf '%s\n' \
	  'make test          Run fast Python contract tests' \
	  'make mujoco-smoke  Validate the MuJoCo observation contract' \
	  'make low-level-gate Run the reusable controller regression gate' \
	  'make rough-terrain-gate Run slope, bump, step, and disturbance gates' \
	  'make ros-build     Build the ROS 2 packages (requires sourced ROS)'

test:
	@$(PYTHON) -m unittest discover -s tests -v

mujoco-smoke:
	@$(M20_CLI) smoke

low-level-gate:
	@$(M20_CLI) low-level-gate

rough-terrain-gate:
	@$(PYTHON) scripts/mujoco/evaluate_m20_rough_terrain_controller.py

ros-build:
	@colcon build --symlink-install
