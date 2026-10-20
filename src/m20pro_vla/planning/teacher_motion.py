"""Explicit teacher-only responsive motion; never a learned-policy controller.

Inputs are the observation teacher's current local route goal and onboard
lidar. No target identity, target coordinates or object-to-room map is read.
Physical action labels use a new contract and require a matching learner
normalization before they can be admitted to training.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from m20pro_vla.low_level.shield import lidar_safety_shield


MOTION_CONTRACT = 'm20_teacher_responsive_physical_v1'


@dataclass
class ResponsiveTeacherMotion:
    previous: np.ndarray = field(default_factory=lambda: np.zeros(4))
    retreat_start: np.ndarray | None = None
    retreat_ticks: int = 0
    pause_ticks: int = 0
    recovery_count: int = 0
    recovery_armed: bool = True

    def command(self, desired, *, xy, yaw, local_goal, lidar, scan=False):
        raw = np.asarray(desired, dtype=np.float64)
        rays = np.asarray(lidar, dtype=np.float64)
        if raw.shape != (4,) or rays.shape != (72,) or not np.isfinite(raw).all():
            raise ValueError('Expected finite body command and 72 lidar rays')
        valid = bool(np.isfinite(rays).all() and np.all(rays >= 0))
        front = float(rays[32:41].min()) if valid else 0.
        rear = float(rays[[68, 69, 70, 71, 0, 1, 2, 3, 4]].min()) if valid else 0.
        phase = 'route'
        # Success stop is supplied by the already-discovered teacher phase.
        # It has priority over all recovery maneuvers.
        if raw[3] >= .5:
            target = np.array([0., 0., 0., 1.])
            self.retreat_start = None
            self.pause_ticks = 0
            phase = 'terminal_stop'
        elif not valid:
            target = np.array([0., 0., 0., 1.])
            phase = 'invalid_lidar_stop'
        else:
            if front > 1.15:
                self.recovery_armed = True
            if (self.retreat_start is None and not self.pause_ticks
                    and self.recovery_armed and front < .70 and rear > 1.25):
                self.retreat_start = np.asarray(xy, dtype=float).copy()
                self.retreat_ticks = 0
                self.recovery_count += 1
                self.recovery_armed = False
            if self.retreat_start is not None:
                moved = float(np.linalg.norm(np.asarray(xy) - self.retreat_start))
                if rear <= .60 or front >= 1.05 or moved >= .50 or self.retreat_ticks >= 150:
                    self.retreat_start = None
                    self.pause_ticks = 20
                else:
                    self.retreat_ticks += 1
            if self.retreat_start is not None:
                target = np.array([-.20, 0., 0., 0.])
                phase = 'rear_clear_retreat'
            elif self.pause_ticks:
                self.pause_ticks -= 1
                target = np.array([0., 0., 0., 1.])
                phase = 'recovery_settle'
            elif scan:
                target = np.array([0., 0., .40, 0.])
                phase = 'room_scan'
            else:
                delta = np.asarray(local_goal, dtype=float) - np.asarray(xy, dtype=float)
                distance = float(np.linalg.norm(delta))
                bearing = (math.atan2(delta[1], delta[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
                if abs(bearing) > .18:
                    target = np.array([0., 0., np.clip(.8 * bearing, -.4, .4), 0.])
                    phase = 'align_then_drive'
                else:
                    speed = .50 if front >= 2.0 and distance >= 1.4 and abs(bearing) < .08 else min(.35, max(0., raw[0]))
                    target = np.array([speed, 0., 0. if speed > .35 else np.clip(.6 * bearing, -.10, .10), 0.])
                    phase = 'clear_fast_drive' if speed > .35 else 'local_approach'
        if target[3] >= .5:
            smooth = target.copy()
        else:
            smooth = target.copy()
            # 50 Hz labels: same .5 m/s² forward and .375 rad/s² yaw slew as
            # the validated 25 Hz/two-physics-step execution schedule.
            smooth[0] = self.previous[0] + np.clip(.2 * (target[0] - self.previous[0]), -.01, .01)
            smooth[2] = self.previous[2] + np.clip(.2 * (target[2] - self.previous[2]), -.0075, .0075)
            # Finish decelerating the previous axis before accelerating the
            # next: the fast moving arc is not a validated primitive.
            if abs(target[2]) > .10 and abs(smooth[0]) > .02:
                smooth[2] = 0.
            if abs(smooth[2]) > .12:
                smooth[0] = 0.
            elif target[0] > .35 and abs(smooth[2]) > .02:
                smooth[0] = min(smooth[0], .10)
        command, reason = lidar_safety_shield(smooth, rays, stop_distance=.5, slow_distance=1.25)
        self.previous = command.copy()
        return command, dict(phase=phase, shield=reason, front=front, rear=rear,
                             recovery_count=self.recovery_count)
