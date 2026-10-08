"""Observation-based exploration schedule for S3 expert demonstrations only.

The schedule has room geometry but no target coordinates or object-to-room
assignment. It tours rooms in a fixed order until the task colour is observed
in the current onboard RGB. Target routing is enabled only after discovery.
This is a training teacher, not a learned-policy execution gate.
"""
from dataclasses import dataclass
import math


@dataclass
class RoomSearchSchedule:
    room_centers: tuple[tuple[float, float], ...]
    arrival_radius: float = 0.75
    min_pixels: int = 5
    confirmation_observations: int = 3
    sweep_radians: float = math.pi + 0.2
    room_index: int = 0
    discovered: bool = False
    evidence_count: int = 0
    scan_yaw: float | None = None
    swept: float = 0.0
    completed_room_scans: int = 0
    safe_handoff_bounds: tuple[tuple[float, float, float, float], ...] = ()
    handoff_radius: float = 0.25
    handoff_room_index: int | None = None
    handoff_complete: bool = False
    scan_start_radius: float | None = None

    def __post_init__(self):
        if not self.room_centers or not all(math.isfinite(v) for p in self.room_centers for v in p):
            raise ValueError('Finite room centers are required')
        if self.arrival_radius <= 0 or self.min_pixels < 1 or self.confirmation_observations < 1:
            raise ValueError('Positive search settings are required')
        if not math.isfinite(self.handoff_radius) or self.handoff_radius <= 0:
            raise ValueError('Positive finite handoff radius required')
        if self.scan_start_radius is not None and (not math.isfinite(self.scan_start_radius)
                or not 0 < self.scan_start_radius <= self.arrival_radius):
            raise ValueError('Scan start radius must be positive and within the scan area')
        if self.safe_handoff_bounds:
            if len(self.safe_handoff_bounds) != len(self.room_centers):
                raise ValueError('Room bounds must match centers')
            for bounds, center in zip(self.safe_handoff_bounds, self.room_centers):
                if len(bounds) != 4 or not all(math.isfinite(v) for v in bounds):
                    raise ValueError('Finite rectangular room bounds required')
                x0,x1,y0,y1 = bounds
                if not (x0 < center[0] < x1 and y0 < center[1] < y1):
                    raise ValueError('Room center must be inside its bounds')

    def _discovered_decision(self, xy):
        if self.handoff_room_index is not None and not self.handoff_complete:
            center = self.room_centers[self.handoff_room_index]
            if math.dist(xy, center) > self.handoff_radius:
                return dict(mode='handoff', goal_xy=center, room_index=self.handoff_room_index,
                            evidence_count=self.evidence_count)
            self.handoff_complete = True
        return dict(mode='target', room_index=self.room_index, evidence_count=self.evidence_count)

    def advance(self, xy, yaw, target_pixels):
        if not all(math.isfinite(v) for v in (*xy, yaw, target_pixels)) or target_pixels < 0:
            raise ValueError('Search observation is invalid')
        if self.discovered:
            return self._discovered_decision(xy)
        self.evidence_count = self.evidence_count + 1 if target_pixels >= self.min_pixels else 0
        if self.evidence_count >= self.confirmation_observations:
            self.discovered = True
            # Opt-in teacher maneuver: finish entering the room before a large
            # target-routing turn, so a skid turn does not begin at a door frame.
            for index,(x0,x1,y0,y1) in enumerate(self.safe_handoff_bounds):
                if x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1:
                    self.handoff_room_index = index
                    break
            return self._discovered_decision(xy)
        center = self.room_centers[self.room_index]
        if math.dist(xy, center) > self.arrival_radius:
            self.scan_yaw = None
            self.swept = 0.0
            return dict(mode='explore', goal_xy=center, room_index=self.room_index)
        if (self.scan_yaw is None and self.scan_start_radius is not None
                and math.dist(xy, center) > self.scan_start_radius):
            # Enter the interior anchor before starting a skid turn. Once
            # started, retain the original outer scan-area limit so normal
            # turn drift does not repeatedly erase measured camera coverage.
            return dict(mode='explore', goal_xy=center, room_index=self.room_index)
        if self.scan_yaw is not None:
            delta = (yaw - self.scan_yaw + math.pi) % (2 * math.pi) - math.pi
            # Large pose jumps cannot masquerade as camera coverage.
            if abs(delta) <= 0.2:
                self.swept = max(0.0, self.swept + delta)
        self.scan_yaw = yaw
        if self.swept >= self.sweep_radians:
            self.completed_room_scans += 1
            self.room_index = (self.room_index + 1) % len(self.room_centers)
            self.scan_yaw = None
            self.swept = 0.0
            return dict(mode='explore', goal_xy=self.room_centers[self.room_index], room_index=self.room_index)
        return dict(mode='scan', room_index=self.room_index, swept_radians=self.swept)
