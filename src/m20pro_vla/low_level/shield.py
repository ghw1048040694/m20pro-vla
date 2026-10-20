"""LiDAR safety envelope and the geometry invariants it depends on.

Both functions here are pure: they take a body command, a scan, and three
ranges, and they return or raise. Keeping them out of the closed-loop player
means the safety contract can be tested without loading a policy stack.
"""

from __future__ import annotations

import math

import numpy as np

# Episode-level evaluation geometry defaults. Kept here rather than in the
# player so the safety invariant below and the players that consume it agree on
# one set of numbers.
#
# 0.95 m, raised from 0.70 m on 2026-10-04. When the expert stopped at 0.70 m the
# policy camera showed *zero* target pixels in 20 of the 29 episodes whose
# teacher reached that ring, so the stop label was not decodable from the
# observation the policy actually gets. The target is best framed at 0.98 m and
# leaves the frame through the bottom edge beyond that; measured on the same 29
# episodes (diag/stop_rule_feasibility.py), a stop at 0.95 m carries a median of
# 812 target pixels (p10 414, min 89) and **no** episode below the 80-pixel gate.
# It also makes the safety geometry consistent: the base/LiDAR frame gap is
# 0.202 m, so 0.70 - (0.50 + 0.202) = -0.002 m of real margin before, and
# 0.95 - (0.50 + 0.202) = +0.248 m after.
DEFAULT_SUCCESS_RADIUS_M = 0.95
DEFAULT_SAFETY_STOP_DISTANCE_M = 0.50

# The planar LiDAR is 72 rays; 32:41 is the forward fan in the M20 frame.
FORWARD_RAY_SLICE = slice(32, 41)
# Angles run from -pi at index0 to pi-2pi/72. Rear is the wrapped fan.
REAR_RAY_INDICES = np.asarray((68, 69, 70, 71, 0, 1, 2, 3, 4), dtype=np.intp)

# The forward fan measures range to the nearest surface, while success is
# measured between the *base frame* and the target centre. Measured on episode
# 6022 (2026-10-03): the shield's last latch left the base 0.716 m from the
# target with the forward fan reading 0.514 m, so the two frames are 0.202 m
# apart. A stop radius therefore has to clear this gap before it clears the
# success radius - see the note in ``validate_stop_geometry``.
LIDAR_TO_BASE_OFFSET_M = 0.202


def lidar_safety_shield(
    command: np.ndarray,
    lidar: np.ndarray,
    *,
    stop_distance: float,
    slow_distance: float,
) -> tuple[np.ndarray, str]:
    """Apply a footprint-aware safety envelope using only onboard LiDAR."""
    scan = np.asarray(lidar, dtype=np.float64)
    if scan.shape != (72,) or not np.isfinite(scan).all():
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64), "invalid_lidar"
    safe = np.asarray(command, dtype=np.float64).copy()
    if safe[3] > 0.5 or safe[0] == 0.0:
        return safe, "none"
    reverse = bool(safe[0] < 0.0)
    clearance = float(np.min(scan[REAR_RAY_INDICES] if reverse else scan[FORWARD_RAY_SLICE]))
    if clearance <= stop_distance:
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64), "emergency_stop"
    if clearance < slow_distance:
        ratio = (clearance - stop_distance) / max(1.0e-6, slow_distance - stop_distance)
        magnitude = min(abs(float(safe[0])), 0.04 + 0.08 * float(np.clip(ratio, 0.0, 1.0)))
        safe[0] = -magnitude if reverse else magnitude
        return safe, "slow"
    return safe, "none"


def validate_stop_geometry(
    *,
    success_radius: float,
    stop_distance: float,
    slow_distance: float,
) -> None:
    """Reject a geometry whose safety stop sits outside the success ring.

    The shield latches a hard stop at ``stop_distance``. If that range is not
    smaller than ``success_radius`` the robot is braked before it can ever enter
    the radius the episode is graded on, so every near miss is recorded as a
    policy failure. That is what a 0.62 m stop against a 0.45 m radius produced
    on 2026-10-03: episodes stopped 0.026-0.051 m short and were scored 0 even
    though the policy had already arrived.

    This covers the frame mismatch only in the loose sense: the shield reads the
    *forward fan*, success is measured from the *base frame*, and the two differ
    by ``LIDAR_TO_BASE_OFFSET_M`` (0.202 m, measured). The shipped pair is now
    (0.50 / 0.95), which leaves +0.248 m of real margin; the previous (0.50 /
    0.70) pair landed at -0.002 m, nominally unreachable, and episodes did in
    fact stop around 0.70-0.72 m.

    The 0.95 m ring was chosen so the stop is also *observable*: at 0.70 m the
    policy camera saw nothing in 20/29 teacher arrivals (M20-STOP-DIAG-02), and
    the target is best framed at 0.98 m. Stopping a little further out also means
    the shield no longer brakes on the very object the task asks the robot to
    reach, which it did whenever the target itself entered the forward fan.
    """
    for name, value in (
        ("success_radius", success_radius),
        ("stop_distance", stop_distance),
        ("slow_distance", slow_distance),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    if not stop_distance < slow_distance:
        raise ValueError(
            f"stop_distance ({stop_distance}) must be smaller than "
            f"slow_distance ({slow_distance})"
        )
    if not stop_distance < success_radius:
        raise ValueError(
            f"stop_distance ({stop_distance}) must be smaller than success_radius "
            f"({success_radius}); otherwise the LiDAR safety shield latches a stop "
            f"outside the radius the episode is graded on"
        )


def approach_margin(success_radius: float, stop_distance: float) -> float:
    """Real margin left for the base frame to enter the success ring.

    ``success_radius - (stop_distance + LIDAR_TO_BASE_OFFSET_M)``. Negative means
    the shield can brake the base short of the ring however well the policy
    drives. Reported rather than raised: the shipped pair (0.50 / 0.95) sits at
    +0.248 m, but moving either number is a decision about the task, not a bug in
    the geometry.
    """
    return float(success_radius) - (float(stop_distance) + LIDAR_TO_BASE_OFFSET_M)


__all__ = [
    "DEFAULT_SAFETY_STOP_DISTANCE_M",
    "DEFAULT_SUCCESS_RADIUS_M",
    "FORWARD_RAY_SLICE",
    "LIDAR_TO_BASE_OFFSET_M",
    "approach_margin",
    "lidar_safety_shield",
    "validate_stop_geometry",
]
