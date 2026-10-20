.DEFAULT_GOAL := help

PYTHON ?= python
M20_CLI ?= PYTHONPATH=src $(PYTHON) -m m20pro_vla.cli

.PHONY: help test mujoco-smoke low-level-gate experiment-plan ros-build

help:
	@printf '%s\n' \
	  'make test          Run fast Python contract tests' \
	  'make mujoco-smoke  Validate the MuJoCo observation contract' \
	  'make low-level-gate Run the reusable controller regression gate' \
	  'make experiment-plan Validate the current one-config experiment plan' \
	  'make ros-build     Build the ROS 2 packages (requires sourced ROS)'

test:
	@$(PYTHON) -m unittest discover -s tests -v

mujoco-smoke:
	@$(M20_CLI) smoke

low-level-gate:
	@$(M20_CLI) low-level-gate

experiment-plan:
	@$(M20_CLI) experiment --stage plan

ros-build:
	@colcon build --symlink-install
