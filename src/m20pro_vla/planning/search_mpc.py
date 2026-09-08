"""Small predictive planner for search/occlusion MuJoCo curricula.

The planner is intentionally lightweight: it uses the real MuJoCo dynamics
and the canonical M20 low-level controller as the rollout model, then scores a
small set of primitive body commands over a short horizon. This is not a
trained world model yet, but it gives the dataset collector an MPC-style
closed loop for search episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import mujoco
import numpy as np

from m20pro_vla.low_level import M20BodyCommand, M20LowLevelController, M20LowLevelControllerState
from m20pro_vla.sim.mujoco import ObstacleSpec, obstacle_blocks_segment


@dataclass(frozen=True)
class SearchMPCConfig:
    horizon_steps: int = 8
    obstacle_padding: float = 0.05
    target_radius: float = 0.45
    subgoal_radius: float = 0.35
    subgoal_margin: float = 0.22
    lane_margin: float = 0.28
    pre_obstacle_margin_x: float = 0.38
    post_obstacle_margin_x: float = 0.42
    waypoint_radius: float = 0.24
    min_forward_speed: float = 0.12
    max_forward_speed: float = 0.35
    yaw_speed: float = 0.15
    stop_distance: float = 0.48
    progress_weight: float = 12.0
    terminal_distance_weight: float = 2.2
    visibility_weight: float = 3.0
    visibility_transition_weight: float = 4.5
    clearance_weight: float = 0.8
    occlusion_escape_weight: float = 8.0
    side_alignment_weight: float = 5.0
    heading_weight: float = 0.4
    turn_penalty: float = 0.0
    stop_bonus: float = 1.2
    occluded_stop_penalty: float = 2.0
    stop_penalty_far: float = 2.4
    collision_penalty: float = 12.0
    use_route_planner_when_obstacles: bool = True


def yaw_from_quaternion_wxyz(quat: np.ndarray | Sequence[float]) -> float:
    values = np.asarray(quat, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("quaternion must be four finite values in wxyz order")
    return float(np.arctan2(
        2.0 * (values[0] * values[3] + values[1] * values[2]),
        1.0 - 2.0 * (values[2] ** 2 + values[3] ** 2),
    ))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clone_data(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjData:
    clone = mujoco.MjData(model)
    clone.qpos[:] = data.qpos
    clone.qvel[:] = data.qvel
    if hasattr(clone, "act") and clone.act.size == data.act.size:
        clone.act[:] = data.act
    clone.ctrl[:] = data.ctrl
    clone.time = float(data.time)
    if hasattr(clone, "qacc_warmstart") and clone.qacc_warmstart.size == data.qacc_warmstart.size:
        clone.qacc_warmstart[:] = data.qacc_warmstart
    mujoco.mj_forward(model, clone)
    return clone


def obstacle_clearance_xy(point_xy: np.ndarray | Sequence[float], obstacle: ObstacleSpec) -> float:
    point = np.asarray(point_xy, dtype=np.float64)
    center = np.asarray(obstacle.position[:2], dtype=np.float64)
    half = np.asarray(obstacle.size[:2], dtype=np.float64)
    delta = np.maximum(np.abs(point - center) - half, 0.0)
    return float(np.hypot(delta[0], delta[1]))


def _is_target_visible(
    point_xy: np.ndarray,
    target_xy: np.ndarray,
    obstacles: Sequence[ObstacleSpec],
    padding: float,
) -> bool:
    return not any(
        obstacle_blocks_segment(point_xy, target_xy, obstacle, padding=padding)
        for obstacle in obstacles
    )


def _blocking_obstacle(
    point_xy: np.ndarray,
    target_xy: np.ndarray,
    obstacles: Sequence[ObstacleSpec],
    padding: float,
) -> ObstacleSpec | None:
    blocking = [
        obstacle
        for obstacle in obstacles
        if obstacle_blocks_segment(point_xy, target_xy, obstacle, padding=padding)
    ]
    if blocking:
        return min(blocking, key=lambda item: obstacle_clearance_xy(point_xy, item))
    if obstacles:
        point = np.asarray(point_xy, dtype=np.float64)
        return min(
            obstacles,
            key=lambda item: float(
                np.linalg.norm(point - np.asarray(item.position[:2], dtype=np.float64))
            ),
        )
    return None


def _candidate_sequences(
    current_distance: float,
    bearing: float,
    config: SearchMPCConfig,
    preferred_yaw_sign: float | None = None,
) -> list[list[M20BodyCommand]]:
    if current_distance <= config.stop_distance:
        return [[M20BodyCommand(0.0, 0.0, 0.0, True)]]
    forward = float(np.clip(0.28 + 0.06 * min(current_distance, 1.5), config.min_forward_speed, config.max_forward_speed))
    yaw_direct = float(np.clip(0.40 * bearing, -config.yaw_speed, config.yaw_speed))
    yaw_preferred = None
    turn_sign = float(np.sign(bearing))
    if preferred_yaw_sign is not None and abs(preferred_yaw_sign) > 1.0e-9:
        yaw_preferred = math.copysign(config.yaw_speed, preferred_yaw_sign)
        if abs(yaw_direct) > 1.0e-4 and math.copysign(1.0, yaw_direct) != math.copysign(1.0, yaw_preferred):
            yaw_direct = yaw_preferred
        if abs(turn_sign) <= 1.0e-9:
            turn_sign = math.copysign(1.0, preferred_yaw_sign)
    if abs(turn_sign) <= 1.0e-9:
        turn_sign = 1.0
    turn_command = M20BodyCommand(0.0, 0.0, float(turn_sign) * config.yaw_speed, False)
    sequences = [
        [M20BodyCommand(forward, 0.0, yaw_direct, False)] * config.horizon_steps,
        [M20BodyCommand(forward * 0.80, 0.0, yaw_direct, False)] * (config.horizon_steps - 2)
        + [M20BodyCommand(0.0, 0.0, 0.0, True)] * 2,
        [M20BodyCommand(forward * 0.55, 0.0, yaw_direct, False)] * (config.horizon_steps - 3)
        + [M20BodyCommand(0.0, 0.0, 0.0, True)] * 3,
        [M20BodyCommand(forward * 0.30, 0.0, yaw_direct, False)] * (config.horizon_steps - 4)
        + [M20BodyCommand(0.0, 0.0, 0.0, True)] * 4,
        [turn_command] * 2 + [M20BodyCommand(forward, 0.0, yaw_direct, False)] * (config.horizon_steps - 2),
        [turn_command] * 3 + [M20BodyCommand(forward * 0.70, 0.0, yaw_direct, False)] * (config.horizon_steps - 3),
        [turn_command] * 4 + [M20BodyCommand(forward * 0.50, 0.0, yaw_direct, False)] * (config.horizon_steps - 4),
        [turn_command] * 2
        + [M20BodyCommand(forward * 0.55, 0.0, yaw_direct, False)] * (config.horizon_steps - 4)
        + [M20BodyCommand(0.0, 0.0, 0.0, True)] * 2,
        [turn_command] * 3
        + [M20BodyCommand(forward * 0.35, 0.0, yaw_direct, False)] * (config.horizon_steps - 5)
        + [M20BodyCommand(0.0, 0.0, 0.0, True)] * 2,
    ]
    if yaw_preferred is not None:
        sequences.extend([
            [M20BodyCommand(forward * 0.92, 0.0, yaw_preferred, False)] * config.horizon_steps,
            [M20BodyCommand(forward * 0.74, 0.0, yaw_preferred, False)] * config.horizon_steps,
            [M20BodyCommand(forward * 0.30, 0.0, yaw_preferred, False)] * config.horizon_steps,
            [M20BodyCommand(forward * 0.18, 0.0, yaw_preferred, False)] * config.horizon_steps,
            [M20BodyCommand(forward * 0.74, 0.0, yaw_preferred, False)] * (config.horizon_steps - 2)
            + [M20BodyCommand(0.0, 0.0, 0.0, True)] * 2,
            [M20BodyCommand(forward * 0.48, 0.0, yaw_preferred, False)] * (config.horizon_steps - 3)
            + [M20BodyCommand(0.0, 0.0, 0.0, True)] * 3,
            [M20BodyCommand(forward * 0.24, 0.0, yaw_preferred, False)] * (config.horizon_steps - 4)
            + [M20BodyCommand(0.0, 0.0, 0.0, True)] * 4,
            [turn_command] * 2 + [M20BodyCommand(forward * 0.92, 0.0, yaw_preferred, False)] * (config.horizon_steps - 2),
            [turn_command] * 3 + [M20BodyCommand(forward * 0.65, 0.0, yaw_preferred, False)] * (config.horizon_steps - 3),
            [turn_command] * 4 + [M20BodyCommand(forward * 0.42, 0.0, yaw_preferred, False)] * (config.horizon_steps - 4),
            [turn_command] * 2
            + [M20BodyCommand(forward * 0.48, 0.0, yaw_preferred, False)] * (config.horizon_steps - 4)
            + [M20BodyCommand(0.0, 0.0, 0.0, True)] * 2,
            [turn_command] * 3
            + [M20BodyCommand(forward * 0.30, 0.0, yaw_preferred, False)] * (config.horizon_steps - 5)
            + [M20BodyCommand(0.0, 0.0, 0.0, True)] * 2,
        ])
        return [
            *sequences,
            [M20BodyCommand(0.0, 0.0, 0.0, True)],
        ]
    return [
        *sequences,
        [M20BodyCommand(forward * 0.78, 0.0, config.yaw_speed, False)] * config.horizon_steps,
        [M20BodyCommand(forward * 0.78, 0.0, -config.yaw_speed, False)] * config.horizon_steps,
        [M20BodyCommand(forward * 0.48, 0.0, config.yaw_speed, False)] * config.horizon_steps,
        [M20BodyCommand(forward * 0.48, 0.0, -config.yaw_speed, False)] * config.horizon_steps,
        [M20BodyCommand(forward * 0.30, 0.0, config.yaw_speed, False)] * config.horizon_steps,
        [M20BodyCommand(forward * 0.30, 0.0, -config.yaw_speed, False)] * config.horizon_steps,
        [M20BodyCommand(forward * 0.18, 0.0, config.yaw_speed, False)] * config.horizon_steps,
        [M20BodyCommand(forward * 0.18, 0.0, -config.yaw_speed, False)] * config.horizon_steps,
        [M20BodyCommand(forward * 0.78, 0.0, config.yaw_speed, False)] * (config.horizon_steps - 2)
        + [M20BodyCommand(0.0, 0.0, 0.0, True)] * 2,
        [M20BodyCommand(forward * 0.78, 0.0, -config.yaw_speed, False)] * (config.horizon_steps - 2)
        + [M20BodyCommand(0.0, 0.0, 0.0, True)] * 2,
        [M20BodyCommand(forward * 0.48, 0.0, config.yaw_speed, False)] * (config.horizon_steps - 3)
        + [M20BodyCommand(0.0, 0.0, 0.0, True)] * 3,
        [M20BodyCommand(forward * 0.48, 0.0, -config.yaw_speed, False)] * (config.horizon_steps - 3)
        + [M20BodyCommand(0.0, 0.0, 0.0, True)] * 3,
        [M20BodyCommand(forward * 0.30, 0.0, config.yaw_speed, False)] * (config.horizon_steps - 4)
        + [M20BodyCommand(0.0, 0.0, 0.0, True)] * 4,
        [M20BodyCommand(forward * 0.30, 0.0, -config.yaw_speed, False)] * (config.horizon_steps - 4)
        + [M20BodyCommand(0.0, 0.0, 0.0, True)] * 4,
        [M20BodyCommand(0.0, 0.0, 0.0, True)],
    ]


def _route_command(
    current_xy: np.ndarray,
    current_yaw: float,
    goal_xy: np.ndarray,
    target_distance: float,
    *,
    final_approach: bool = False,
    config: SearchMPCConfig | None = None,
) -> tuple[M20BodyCommand, float, float]:
    config = config or SearchMPCConfig()
    delta = goal_xy - current_xy
    goal_distance = float(np.linalg.norm(delta))
    bearing = wrap_angle(math.atan2(float(delta[1]), float(delta[0])) - current_yaw)
    if target_distance <= config.stop_distance:
        return M20BodyCommand(0.0, 0.0, 0.0, True), goal_distance, bearing
    if target_distance <= 0.45:
        if abs(bearing) > 0.20:
            forward = 0.0
        else:
            forward = 0.04
    elif target_distance <= 0.75:
        if abs(bearing) > 0.35:
            forward = 0.0
        elif abs(bearing) > 0.18:
            forward = 0.05
        else:
            forward = 0.08
    elif abs(bearing) > 0.65:
        forward = 0.0
    elif abs(bearing) > 0.35:
        forward = 0.06
    elif abs(bearing) > 0.18:
        forward = 0.12
    else:
        forward = 0.22 if target_distance > 0.60 else 0.10
    yaw_gain = 0.75 if target_distance <= 0.75 else 0.60
    if final_approach:
        # Once the target is visible or the route phase has converged, keep the
        # approach slower and slightly more yaw-responsive so the robot can settle
        # instead of orbiting past the object.
        forward = min(forward, 0.14)
        yaw_gain *= 1.20
    yaw = float(np.clip(yaw_gain * bearing, -0.15, 0.15))
    return M20BodyCommand(forward, 0.0, yaw, False), goal_distance, bearing


class SearchMPCPlanner:
    def __init__(
        self,
        model: mujoco.MjModel,
        target_xy: np.ndarray | Sequence[float],
        obstacles: list[ObstacleSpec],
        config: SearchMPCConfig | None = None,
    ) -> None:
        self.model = model
        self.target_xy = np.asarray(target_xy, dtype=np.float64)
        self.obstacles = list(obstacles)
        self.config = config or SearchMPCConfig()
        self.route_phase = 0
        self.route_side_sign = 1.0

    def _planning_goal(self, current_xy: np.ndarray) -> tuple[np.ndarray, bool, float]:
        visible = _is_target_visible(current_xy, self.target_xy, self.obstacles, self.config.obstacle_padding)
        obstacle = _blocking_obstacle(
            current_xy,
            self.target_xy,
            self.obstacles,
            self.config.obstacle_padding,
        )
        if visible or obstacle is None:
            return self.target_xy, True, 0.0
        side_sign = 1.0 if float(self.target_xy[1] - obstacle.position[1]) >= 0.0 else -1.0
        subgoal = np.array(
            (
                float(obstacle.position[0]) + float(obstacle.size[0]) + self.config.subgoal_margin,
                float(obstacle.position[1]) + side_sign * (float(obstacle.size[1]) + self.config.subgoal_margin),
            ),
            dtype=np.float64,
        )
        return subgoal, False, side_sign

    def _route_waypoints(self, obstacle: ObstacleSpec, side_sign: float) -> tuple[np.ndarray, np.ndarray]:
        lane_y = float(obstacle.position[1]) + side_sign * (
            float(obstacle.size[1]) + self.config.lane_margin
        )
        before = np.array(
            (
                float(obstacle.position[0]) - self.config.pre_obstacle_margin_x,
                lane_y,
            ),
            dtype=np.float64,
        )
        after = np.array(
            (
                float(obstacle.position[0]) + float(obstacle.size[0]) + self.config.post_obstacle_margin_x,
                lane_y,
            ),
            dtype=np.float64,
        )
        return before, after

    def _route_recommend(self, data: mujoco.MjData) -> tuple[np.ndarray, dict[str, float | list[float]]]:
        current_xy = np.asarray(data.qpos[:2], dtype=np.float64)
        current_yaw = yaw_from_quaternion_wxyz(np.asarray(data.qpos[3:7], dtype=np.float64))
        target_distance = float(np.linalg.norm(self.target_xy - current_xy))
        obstacle = _blocking_obstacle(
            current_xy,
            self.target_xy,
            self.obstacles,
            self.config.obstacle_padding,
        )
        visible = _is_target_visible(current_xy, self.target_xy, self.obstacles, self.config.obstacle_padding)
        if obstacle is None or visible:
            self.route_phase = 2
            self.route_side_sign = 0.0
            route_goal = self.target_xy
        else:
            self.route_side_sign = 1.0 if float(self.target_xy[1] - obstacle.position[1]) >= 0.0 else -1.0
            before, after = self._route_waypoints(obstacle, self.route_side_sign)
            if self.route_phase == 0 and float(np.linalg.norm(current_xy - before)) <= self.config.waypoint_radius:
                self.route_phase = 1
            if self.route_phase == 1 and float(np.linalg.norm(current_xy - after)) <= self.config.waypoint_radius:
                self.route_phase = 2
            route_goal = (before, after, self.target_xy)[self.route_phase]
        command, goal_distance, bearing = _route_command(
            current_xy,
            current_yaw,
            route_goal,
            target_distance,
            final_approach=bool(self.route_phase == 2),
            config=self.config,
        )
        info = {
            "score": 0.0,
            "current_distance": target_distance,
            "bearing": bearing,
            "planning_goal": route_goal.tolist(),
            "goal_is_target": bool(self.route_phase == 2),
            "visible_steps": float(visible),
            "route_phase": float(self.route_phase),
            "route_side_sign": float(self.route_side_sign),
            "route_goal_distance": goal_distance,
            "best_forward": float(command.forward),
            "best_yaw": float(command.yaw),
            "best_stop": float(command.stop),
        }
        return command.as_array(), info

    def recommend(
        self,
        data: mujoco.MjData,
        controller: M20LowLevelController,
    ) -> tuple[np.ndarray, dict[str, float | list[float]]]:
        if self.obstacles and self.config.use_route_planner_when_obstacles:
            return self._route_recommend(data)
        start_xy = np.asarray(data.qpos[:2], dtype=np.float64)
        current_yaw = yaw_from_quaternion_wxyz(np.asarray(data.qpos[3:7], dtype=np.float64))
        planning_goal, goal_is_target, preferred_side_sign = self._planning_goal(start_xy)
        delta = planning_goal - start_xy
        current_distance = float(np.linalg.norm(delta))
        bearing = wrap_angle(math.atan2(float(delta[1]), float(delta[0])) - current_yaw)
        sequences = _candidate_sequences(current_distance, bearing, self.config, preferred_side_sign if not goal_is_target else None)

        controller_state = controller.snapshot()
        best_score = -float("inf")
        best_command = sequences[-1][0]
        candidate_scores: list[float] = []
        candidate_labels: list[str] = []
        best_visible_steps = 0
        best_visible_transitions = 0

        for sequence in sequences:
            command = sequence[0]
            rollout_data = clone_data(self.model, data)
            rollout_controller = M20LowLevelController(self.model)
            rollout_controller.restore(controller_state)
            valid = True
            rolling_score = 0.0
            previous_distance = current_distance
            previous_visible = _is_target_visible(start_xy, self.target_xy, self.obstacles, self.config.obstacle_padding)
            visible_steps = 0
            visible_transitions = 0
            for command_step in sequence:
                diagnostics = rollout_controller.step(rollout_data, command_step.as_array())
                if not diagnostics.finite:
                    valid = False
                    break
                rollout_xy = np.asarray(rollout_data.qpos[:2], dtype=np.float64)
                current_yaw = yaw_from_quaternion_wxyz(np.asarray(rollout_data.qpos[3:7], dtype=np.float64))
                step_distance = float(np.linalg.norm(planning_goal - rollout_xy))
                progress = previous_distance - step_distance
                step_bearing = wrap_angle(math.atan2(float(planning_goal[1] - rollout_xy[1]), float(planning_goal[0] - rollout_xy[0])) - current_yaw)
                visible = _is_target_visible(rollout_xy, self.target_xy, self.obstacles, self.config.obstacle_padding)
                clearance = min((obstacle_clearance_xy(rollout_xy, obstacle) for obstacle in self.obstacles), default=1.0)
                occlusion_escape = 0.0
                side_alignment = 0.0
                if not visible and self.obstacles:
                    occlusion_escape = max(abs(float(rollout_xy[1] - obstacle.position[1])) for obstacle in self.obstacles)
                    side_alignment = max(
                        preferred_side_sign * float(rollout_xy[1] - obstacle.position[1])
                        for obstacle in self.obstacles
                    )
                if visible and not previous_visible:
                    visible_transitions += 1
                rolling_score += (
                    self.config.progress_weight * progress
                    + self.config.heading_weight * float(np.cos(step_bearing))
                    + self.config.visibility_weight * (1.0 if visible else -0.25)
                    + self.config.visibility_transition_weight * (1.0 if visible and not previous_visible else 0.0)
                    + self.config.clearance_weight * clearance
                    + self.config.occlusion_escape_weight * occlusion_escape
                    + self.config.side_alignment_weight * side_alignment
                )
                visible_steps += int(visible)
                previous_distance = step_distance
                previous_visible = visible

            final_xy = np.asarray(rollout_data.qpos[:2], dtype=np.float64)
            final_distance = float(np.linalg.norm(planning_goal - final_xy))
            visible = _is_target_visible(final_xy, self.target_xy, self.obstacles, self.config.obstacle_padding)
            clearance = min((obstacle_clearance_xy(final_xy, obstacle) for obstacle in self.obstacles), default=1.0)
            tail_stop = bool(sequence[-1].stop)
            finite = bool(np.isfinite(rollout_data.qpos).all() and np.isfinite(rollout_data.qvel).all())
            base_height = float(rollout_data.qpos[2])
            collision_penalty = 0.0
            if not finite:
                collision_penalty += self.config.collision_penalty
            if base_height < 0.40:
                collision_penalty += self.config.collision_penalty
            if clearance < 0.02:
                collision_penalty += self.config.collision_penalty * 0.5
            score = rolling_score
            score -= self.config.terminal_distance_weight * final_distance
            if not goal_is_target:
                score += self.config.visibility_weight * (1.0 if visible else 0.0)
                if final_distance <= self.config.subgoal_radius:
                    score += self.config.stop_bonus
            elif tail_stop and final_distance <= max(self.config.stop_distance * 2.0, 0.35):
                score += self.config.stop_bonus * 1.25
            score += self.config.stop_bonus if command.stop and final_distance <= self.config.stop_distance else 0.0
            if command.stop and not visible:
                score -= self.config.occluded_stop_penalty
            if command.stop and final_distance > self.config.stop_distance:
                score -= self.config.stop_penalty_far
            score -= self.config.turn_penalty * abs(float(command.yaw))
            score -= collision_penalty
            if not valid:
                score -= self.config.collision_penalty
            candidate_scores.append(score)
            candidate_labels.append(
                " -> ".join(
                    f"f={step.forward:+.2f},y={step.yaw:+.2f},stop={int(step.stop)}"
                    for step in sequence
                )
            )
            if score > best_score:
                best_score = score
                best_command = command
                best_visible_steps = visible_steps
                best_visible_transitions = visible_transitions

        info = {
            "score": float(best_score),
            "current_distance": current_distance,
            "bearing": bearing,
            "planning_goal": planning_goal.tolist(),
            "goal_is_target": bool(goal_is_target),
            "visible_steps": best_visible_steps,
            "visible_transitions": best_visible_transitions,
            "candidate_scores": candidate_scores,
            "candidate_labels": candidate_labels,
            "best_forward": float(best_command.forward),
            "best_yaw": float(best_command.yaw),
            "best_stop": float(best_command.stop),
        }
        return best_command.as_array(), info
