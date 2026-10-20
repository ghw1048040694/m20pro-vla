"""Acceptance-protocol plumbing for the SmolVLA closed-loop player.

Pure functions only. They read a namespace or a metadata ``dict`` and return
values, so the protocol can be reasoned about and tested without importing
lerobot, mujoco or torch - the same split that ``low_level.shield`` uses for the
safety envelope. The player imports these; the tests exercise them directly.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from m20pro_vla.low_level.shield import (
    DEFAULT_SAFETY_STOP_DISTANCE_M,
    DEFAULT_SUCCESS_RADIUS_M,
)

# Fallbacks for the closed-loop knobs that the acceptance config may override.
# They exist only so a config that omits a key still runs; the config wins when
# it states a value, and an explicit CLI flag wins over both.
DEFAULT_POLICY_STEPS = 900
DEFAULT_MOTION_SMOOTHING = 0.20
DEFAULT_MAX_FORWARD_DELTA = 0.02
DEFAULT_MAX_YAW_DELTA = 0.015
DEFAULT_VISUAL_STOP_MIN_PIXELS = 80
# Settle window kept after a latched stop, in policy steps. Short enough that it
# cannot become an unattended tail, long enough to catch "stops, then topples".
DEFAULT_POST_STOP_HOLD_STEPS = 25
# Contact steps tolerated before ``no_obstacle_contact`` fails. Zero keeps the
# criterion bit-for-bit identical to the historical protocol (a single contact
# step fails the episode); raising it is an explicit, recorded policy change.
DEFAULT_OBSTACLE_CONTACT_TOLERANCE_STEPS = 0
# Tolerances reported *alongside* the strict score. The strict judgement never
# moves with them; they exist so a zero-tolerance score is not read as a
# navigation failure when it is really a contact-policy choice.
DEFAULT_CONTACT_TOLERANCE_LADDER: tuple[int, ...] = (0, 1, 5, 10, 20, 50)

# The four criteria a contact-tolerant success must still satisfy. Kept next to
# the ladder because the tolerance report re-runs the conjunction with only the
# contact clause relaxed.
CONTACT_TOLERANT_CORE_CRITERIA: tuple[str, ...] = (
    "discovered",
    "reached_radius",
    "stop_latched",
    "stable_attitude",
)

# (argparse attribute, acceptance-config key, cast, module fallback)
EVALUATION_KNOBS: tuple[tuple[str, str, type, float | int], ...] = (
    ("fresh_stop_confirmation", "fresh_stop_confirmation", bool, False),
    ("reversible_stop", "reversible_stop", bool, False),
    ("policy_steps", "policy_steps", int, DEFAULT_POLICY_STEPS),
    ("success_radius", "success_radius", float, DEFAULT_SUCCESS_RADIUS_M),
    ("safety_stop_distance", "safety_stop_distance", float, DEFAULT_SAFETY_STOP_DISTANCE_M),
    ("motion_smoothing", "motion_smoothing", float, DEFAULT_MOTION_SMOOTHING),
    ("max_forward_delta", "max_forward_delta", float, DEFAULT_MAX_FORWARD_DELTA),
    ("max_yaw_delta", "max_yaw_delta", float, DEFAULT_MAX_YAW_DELTA),
    ("visual_stop_min_pixels", "visual_stop_min_pixels", int, DEFAULT_VISUAL_STOP_MIN_PIXELS),
    ("post_stop_hold_steps", "post_stop_hold_steps", int, DEFAULT_POST_STOP_HOLD_STEPS),
    (
        "obstacle_contact_tolerance_steps",
        "obstacle_contact_tolerance_steps",
        int,
        DEFAULT_OBSTACLE_CONTACT_TOLERANCE_STEPS,
    ),
)

# The five conditions behind ``success``. Reported individually because a bare
# ``success: false`` hides whether an episode failed on geometry, on the stop
# decision, on attitude, or on collisions - four completely different fixes.
CRITERION_NAMES: tuple[str, ...] = (
    "discovered",
    "reached_radius",
    "stop_latched",
    "stable_attitude",
    "no_obstacle_contact",
)

FAILURE_MODES: tuple[str, ...] = (
    "success",
    "target_never_seen",
    "never_reached",
    "premature_stop",
    "arrived_without_stop",
    "unstable_attitude",
    "obstacle_contact",
    "unknown",
)


def apply_evaluation_config(args: Any, config: dict | None) -> Any:
    """Fill every acceptance knob from the experiment config unless the CLI pinned it.

    The experiment config is the single source of truth for the closed-loop
    protocol, and each knob resolves as *CLI > config > module default*. An
    earlier revision honoured only ``success_radius`` and ``safety_stop_distance``,
    so ``smolvla_evaluation.policy_steps = 650`` was silently ignored and every
    episode ran on the argparse default of 900 - short of the teacher's 986-step
    median and a third of its 1800-step maximum, so the learner was cut off before
    its own demonstration had finished on 34 of 47 layouts.
    """
    evaluation = (config or {}).get("smolvla_evaluation", {}) or {}
    for attr, key, cast, fallback in EVALUATION_KNOBS:
        if getattr(args, attr, None) is None:
            setattr(args, attr, cast(evaluation[key]) if key in evaluation else fallback)
    return args


def resolve_policy_step_budget(args: Any, metadata: dict) -> tuple[int, str]:
    """Give the learner the horizon its teacher demonstration actually used.

    Teacher episodes in ``m20_hidden_search_v2`` run 816-1800 policy steps
    (median 986; 34 of 47 above 900), so a flat budget truncates the closed loop
    strictly earlier than the demonstration it is compared against. The collected
    episode JSON carries that length in ``steps``, so the budget is taken from
    there whenever ``--match-teacher-budget`` is on; otherwise the resolved
    ``--policy-steps`` fallback applies.
    """
    teacher_steps = metadata.get("steps")
    if (
        getattr(args, "match_teacher_budget", False)
        and isinstance(teacher_steps, (int, float))
        and not isinstance(teacher_steps, bool)
        and float(teacher_steps) > 0.0
    ):
        return int(teacher_steps), "teacher_episode_steps"
    return int(args.policy_steps), "policy_steps"


def terminate_after_stop(
    *,
    stop_latched: bool,
    stop_step: int,
    step: int,
    hold_steps: int,
) -> bool:
    """Whether the closed loop should end now that a stop has been latched.

    The collector ends a teacher episode on the step the expert declares arrival,
    so every demonstration the learner trains on terminates there. The player,
    however, ignored its own latch and kept commanding ``(0, 0, 0, 1)`` until the
    step budget ran out - leaving a tail of 94 to 1422 unattended steps in the
    2026-10-03 fleet run. A tail can only hurt: ``max_abs_pitch_deg`` and
    ``obstacle_contact_step_count`` are accumulated over the whole episode, and a
    robot holding a zero-velocity command for a thousand steps is being graded on
    wobble the task never asked about. Every episode whose budget reached 1800
    recorded a pitch of 8.9-16.3 degrees, against at most 3.8 for every episode
    that stopped inside a 1000-step budget.

    ``hold_steps`` keeps a short, bounded settle window so "stops and then falls
    over" is still caught. A negative ``hold_steps`` restores the old
    run-to-budget behaviour.
    """
    if not stop_latched or hold_steps < 0:
        return False
    return step >= stop_step + hold_steps


def build_criteria(
    *,
    target_first_visible_step: int,
    target_reached_step: int,
    stop_step: int,
    stable: bool,
    obstacle_contact_step_count: int,
    obstacle_contact_tolerance_steps: int = DEFAULT_OBSTACLE_CONTACT_TOLERANCE_STEPS,
) -> dict[str, bool]:
    """Return the five success conditions as a named mapping.

    ``obstacle_contact_tolerance_steps`` defaults to 0, which reduces the contact
    clause to ``contact_step_count == 0`` - the historical, bit-for-bit identical
    protocol. A positive value is an explicit policy change that only relaxes
    this one criterion; the other four are untouched. It exists because on the S2
    room scene the binding constraint was contact, not navigation: 12 of 39
    episodes reached the goal and failed solely on a doorway graze, several of
    them on a single frame.
    """
    tolerance = max(0, int(obstacle_contact_tolerance_steps))
    return {
        "discovered": bool(target_first_visible_step >= 0),
        "reached_radius": bool(target_reached_step >= 0),
        "stop_latched": bool(stop_step >= 0),
        "stable_attitude": bool(stable),
        "no_obstacle_contact": int(obstacle_contact_step_count) <= tolerance,
    }


def count_successes_by_contact_tolerance(
    episodes: Iterable[Mapping[str, Any]],
    tolerances: Iterable[int] = DEFAULT_CONTACT_TOLERANCE_LADDER,
) -> dict[str, int]:
    """Count episodes that succeed under each contact tolerance.

    The strict judgement never moves: this re-runs the five-way conjunction with
    only the contact clause relaxed to ``<= tolerance`` while holding the other
    four fixed, keyed by the tolerance as a string so it round-trips through JSON.
    Reporting it beside ``success_count`` separates "the learner did not get
    there" from "the learner got there and grazed a doorway", which are two
    different problems with two different fixes.
    """
    rows = list(episodes)
    counts: dict[str, int] = {}
    for tolerance in tolerances:
        limit = max(0, int(tolerance))
        counts[str(limit)] = sum(
            1
            for item in rows
            if all(item.get("criteria", {}).get(name) for name in CONTACT_TOLERANT_CORE_CRITERIA)
            and int(item.get("obstacle_contact_step_count", 0)) <= limit
        )
    return counts


def classify_failure(criteria: dict[str, bool], stop_step: int) -> str:
    """Name the criterion that decided a failed episode.

    ``success`` is a conjunction of five independent conditions, and a bare
    ``success: false`` hides which one bit. On the 2026-10-03 fleet run this split
    showed the binding constraint was almost never geometry: five episodes voted
    to stop 0.7-3.1 m short (``premature_stop``) and eight never voted at all -
    ``arrived_without_stop`` for the two that were inside the ring, upright and
    untouched when the budget ran out, and ``never_reached`` for the rest.
    """
    if all(criteria.get(name, False) for name in CRITERION_NAMES):
        return "success"
    if not criteria.get("discovered", False):
        return "target_never_seen"
    if not criteria.get("reached_radius", False):
        return "premature_stop" if stop_step >= 0 else "never_reached"
    if not criteria.get("stop_latched", False):
        return "arrived_without_stop"
    if not criteria.get("stable_attitude", False):
        return "unstable_attitude"
    if not criteria.get("no_obstacle_contact", False):
        return "obstacle_contact"
    return "unknown"


__all__ = [
    "CONTACT_TOLERANT_CORE_CRITERIA",
    "CRITERION_NAMES",
    "DEFAULT_CONTACT_TOLERANCE_LADDER",
    "DEFAULT_MAX_FORWARD_DELTA",
    "DEFAULT_MAX_YAW_DELTA",
    "DEFAULT_MOTION_SMOOTHING",
    "DEFAULT_OBSTACLE_CONTACT_TOLERANCE_STEPS",
    "DEFAULT_POLICY_STEPS",
    "DEFAULT_POST_STOP_HOLD_STEPS",
    "DEFAULT_VISUAL_STOP_MIN_PIXELS",
    "EVALUATION_KNOBS",
    "FAILURE_MODES",
    "apply_evaluation_config",
    "build_criteria",
    "classify_failure",
    "count_successes_by_contact_tolerance",
    "resolve_policy_step_budget",
    "terminate_after_stop",
]


def terminal_stop_step(*, stop_step: int, stop_active: bool, steps_executed: int,
                       hold_steps: int, reversible: bool) -> int:
    """Credit only the final held stop when movement can resume."""
    if not reversible:
        return stop_step
    if not stop_active or stop_step < 0:
        return -1
    return stop_step if steps_executed - stop_step - 1 >= max(0, hold_steps) else -1
