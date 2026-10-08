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

    def __post_init__(self):
        if not self.room_centers or not all(math.isfinite(v) for p in self.room_centers for v in p):
            raise ValueError('Finite room centers are required')
        if self.arrival_radius <= 0 or self.min_pixels < 1 or self.confirmation_observations < 1:
            raise ValueError('Positive search settings are required')

    def advance(self, xy, yaw, target_pixels):
        if not all(math.isfinite(v) for v in (*xy, yaw, target_pixels)) or target_pixels < 0:
            raise ValueError('Search observation is invalid')
        if self.discovered:
            return dict(mode='target', room_index=self.room_index, evidence_count=self.evidence_count)
        self.evidence_count = self.evidence_count + 1 if target_pixels >= self.min_pixels else 0
        if self.evidence_count >= self.confirmation_observations:
            self.discovered = True
            return dict(mode='target', room_index=self.room_index, evidence_count=self.evidence_count)
        center = self.room_centers[self.room_index]
        if math.dist(xy, center) > self.arrival_radius:
            self.scan_yaw = None
            self.swept = 0.0
            return dict(mode='explore', goal_xy=center, room_index=self.room_index)
        if self.scan_yaw is not None:
            delta = (yaw - self.scan_yaw + math.pi) % (2 * math.pi) - math.pi
            # Large pose jumps cannot masquerade as camera coverage.
            if abs(delta) <= 0.2:
                self.swept += abs(delta)
        self.scan_yaw = yaw
        if self.swept >= self.sweep_radians:
            self.completed_room_scans += 1
            self.room_index = (self.room_index + 1) % len(self.room_centers)
            self.scan_yaw = None
            self.swept = 0.0
            return dict(mode='explore', goal_xy=self.room_centers[self.room_index], room_index=self.room_index)
        return dict(mode='scan', room_index=self.room_index, swept_radians=self.swept)
