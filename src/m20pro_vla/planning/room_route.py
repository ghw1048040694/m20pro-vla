"""Preserve completed waypoints when an S3 teacher refreshes the same route."""
import numpy as np

from .search_mpc import SearchMPCPlanner


class RoomSearchRoutePlanner(SearchMPCPlanner):
    """Keep progress only when the refreshed path repeats the existing geometry.

    Unreachable paths, path deviation, and newly introduced intermediate points
    retain the base planner's reset behavior. This teacher adapter leaves the
    base planner, replanning cadence, and learned-policy execution unchanged.
    """

    def _reset_global_plan(self):
        super()._reset_global_plan()
        self._room_previous_plan = None
        self._room_previous_index = 0

    def _global_route_goal(self, current_xy):
        old = getattr(self, '_room_previous_plan', None)
        old_index = getattr(self, '_room_previous_index', 0)
        new = self._global_plan
        if (old is not None and new is not old and old.reachable and new.reachable
                and old_index > 0
                and self._polyline_distance(current_xy, old.waypoints)
                    <= self.config.global_replan_deviation):
            # A refreshed start position is new; every other prefix point must
            # already have been passed, and the entire remaining suffix must
            # match. A changed detour can therefore never be skipped here.
            remaining = np.asarray(old.waypoints[old_index:])
            candidate = len(new.waypoints) - len(remaining)
            passed = old.waypoints[1:old_index]
            if (candidate >= 1
                    and np.array_equal(np.asarray(new.waypoints[candidate:]), remaining)
                    and all(any(np.array_equal(point, previous) for previous in passed)
                            for point in new.waypoints[1:candidate])):
                self._global_waypoint_index = candidate
        goal, index, final = super()._global_route_goal(current_xy)
        self._room_previous_plan = new
        self._room_previous_index = index
        return goal, index, final
