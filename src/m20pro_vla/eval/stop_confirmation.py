"""Stop votes from independent observation-conditioned predictions.

Only queued model actions are invalidated. Observation history, model weights,
normalization, motion smoothing and the downstream LiDAR shield are untouched.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PredictionEvidence:
    fresh: bool
    forced_replan: bool
    generation: int
    queued_before: int


class ActionQueueFreshener:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.generation = 0

    def predict(self, policy: Any, predict: Callable[[], Any], *, confirming: bool):
        if not self.enabled:
            return predict(), PredictionEvidence(False, False, -1, -1)
        queues = getattr(policy, '_queues', None)
        if queues is None or 'action' not in queues:
            raise TypeError('Fresh stop confirmation requires the SmolVLA action queue')
        queue = queues['action']
        if not callable(getattr(queue, 'clear', None)):
            raise TypeError('SmolVLA action queue cannot be invalidated')
        queued = len(queue)
        forced = bool(confirming and queued)
        if confirming:
            queue.clear()
        fresh = len(queue) == 0
        action = predict()
        if fresh:
            self.generation += 1
        return action, PredictionEvidence(fresh, forced, self.generation, queued)


class StopConfirmation:
    def __init__(self, *, required_votes: int, require_fresh: bool):
        if required_votes <= 0:
            raise ValueError('Stop confirmation requires positive vote count')
        self.required_votes = required_votes
        self.require_fresh = require_fresh
        self.votes = 0
        self.pending = False
        self.latched = False

    def resolve(self, desired: np.ndarray, previous: np.ndarray, *,
                visual_evidence: bool, prediction_fresh: bool) -> np.ndarray:
        if self.latched:
            return np.asarray((0., 0., 0., 1.))
        if desired[3] <= .5:
            self.votes = 0
            self.pending = False
            return desired.copy()
        if not visual_evidence:
            self.votes = 0
            self.pending = False
            return previous.copy()
        self.pending = True
        if self.require_fresh and not prediction_fresh:
            # A queued stop schedules a fresh prediction on the next observation.
            # It is not itself a confirmation vote.
            self.votes = 0
            return previous.copy()
        self.votes += 1
        if self.votes < self.required_votes:
            return previous.copy()
        self.latched = True
        self.pending = False
        return np.asarray((0., 0., 0., 1.))
