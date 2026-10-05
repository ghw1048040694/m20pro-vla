"""Closed-loop acceptance plumbing for the M20 SmolVLA player."""

from __future__ import annotations

from m20pro_vla.eval.acceptance import (
    CONTACT_TOLERANT_CORE_CRITERIA,
    CRITERION_NAMES,
    DEFAULT_CONTACT_TOLERANCE_LADDER,
    DEFAULT_MAX_FORWARD_DELTA,
    DEFAULT_MAX_YAW_DELTA,
    DEFAULT_MOTION_SMOOTHING,
    DEFAULT_OBSTACLE_CONTACT_TOLERANCE_STEPS,
    DEFAULT_POLICY_STEPS,
    DEFAULT_VISUAL_STOP_MIN_PIXELS,
    EVALUATION_KNOBS,
    FAILURE_MODES,
    apply_evaluation_config,
    build_criteria,
    classify_failure,
    count_successes_by_contact_tolerance,
    resolve_policy_step_budget,
)

__all__ = [
    "CONTACT_TOLERANT_CORE_CRITERIA",
    "CRITERION_NAMES",
    "DEFAULT_CONTACT_TOLERANCE_LADDER",
    "DEFAULT_MAX_FORWARD_DELTA",
    "DEFAULT_MAX_YAW_DELTA",
    "DEFAULT_MOTION_SMOOTHING",
    "DEFAULT_OBSTACLE_CONTACT_TOLERANCE_STEPS",
    "DEFAULT_POLICY_STEPS",
    "DEFAULT_VISUAL_STOP_MIN_PIXELS",
    "EVALUATION_KNOBS",
    "FAILURE_MODES",
    "apply_evaluation_config",
    "build_criteria",
    "classify_failure",
    "count_successes_by_contact_tolerance",
    "resolve_policy_step_budget",
]
