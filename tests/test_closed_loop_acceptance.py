from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path

from m20pro_vla.eval.acceptance import (
    CONTACT_TOLERANT_CORE_CRITERIA,
    CRITERION_NAMES,
    DEFAULT_OBSTACLE_CONTACT_TOLERANCE_STEPS,
    DEFAULT_POLICY_STEPS,
    DEFAULT_POST_STOP_HOLD_STEPS,
    EVALUATION_KNOBS,
    FAILURE_MODES,
    apply_evaluation_config,
    build_criteria,
    classify_failure,
    count_successes_by_contact_tolerance,
    resolve_policy_step_budget,
    terminate_after_stop,
)

EXPERIMENT_CONFIG = Path(__file__).resolve().parents[1] / "configs/experiment.json"


def unpinned_args(**overrides: object) -> argparse.Namespace:
    """A namespace shaped like the player's CLI with every acceptance knob unpinned."""
    values: dict[str, object] = {attr: None for attr, _, _, _ in EVALUATION_KNOBS}
    values["match_teacher_budget"] = True
    values.update(overrides)
    return argparse.Namespace(**values)


class EvaluateConfigResolutionTest(unittest.TestCase):
    """The config is the protocol's source of truth; the CLI may still pin a knob."""

    def test_config_fills_every_knob(self) -> None:
        args = unpinned_args()
        apply_evaluation_config(
            args,
            {
                "smolvla_evaluation": {
                    "policy_steps": 650,
                    "success_radius": 0.7,
                    "safety_stop_distance": 0.5,
                    "motion_smoothing": 0.3,
                    "max_forward_delta": 0.04,
                    "max_yaw_delta": 0.02,
                    "visual_stop_min_pixels": 120,
                }
            },
        )
        self.assertEqual(args.policy_steps, 650)
        self.assertAlmostEqual(args.motion_smoothing, 0.3)
        self.assertAlmostEqual(args.max_forward_delta, 0.04)
        self.assertAlmostEqual(args.max_yaw_delta, 0.02)
        self.assertEqual(args.visual_stop_min_pixels, 120)

    def test_post_stop_hold_falls_back_to_the_module_default(self) -> None:
        args = unpinned_args()
        apply_evaluation_config(args, {"smolvla_evaluation": {"policy_steps": 650}})
        self.assertEqual(args.post_stop_hold_steps, DEFAULT_POST_STOP_HOLD_STEPS)

    def test_explicit_cli_value_beats_the_config(self) -> None:
        args = unpinned_args(policy_steps=1234, motion_smoothing=0.05)
        apply_evaluation_config(args, {"smolvla_evaluation": {"policy_steps": 650}})
        self.assertEqual(args.policy_steps, 1234)
        self.assertAlmostEqual(args.motion_smoothing, 0.05)

    def test_module_default_applies_when_the_config_is_silent(self) -> None:
        args = unpinned_args()
        apply_evaluation_config(args, {"smolvla_evaluation": {}})
        self.assertEqual(args.policy_steps, DEFAULT_POLICY_STEPS)

    def test_missing_config_is_tolerated(self) -> None:
        args = unpinned_args()
        apply_evaluation_config(args, None)
        self.assertEqual(args.policy_steps, DEFAULT_POLICY_STEPS)

    def test_experiment_config_policy_steps_is_not_silently_dropped(self) -> None:
        """Regression: the shipped config's budget used to be ignored entirely.

        Only ``success_radius`` and ``safety_stop_distance`` were read, so the
        player ran on the argparse default and the config's own ``policy_steps``
        never reached the episode loop.
        """
        config = json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
        configured = config["smolvla_evaluation"]["policy_steps"]
        args = unpinned_args()
        apply_evaluation_config(args, config)
        self.assertEqual(args.policy_steps, configured)
        self.assertGreaterEqual(
            configured,
            1800,
            "the fallback budget must not truncate the longest teacher episode",
        )


class TeacherBudgetTest(unittest.TestCase):
    """The learner should get the horizon its demonstration actually used."""

    def test_episode_steps_win_over_the_fallback(self) -> None:
        args = unpinned_args(policy_steps=900)
        budget, source = resolve_policy_step_budget(args, {"steps": 1800})
        self.assertEqual(budget, 1800)
        self.assertEqual(source, "teacher_episode_steps")

    def test_missing_episode_steps_fall_back(self) -> None:
        args = unpinned_args(policy_steps=900)
        budget, source = resolve_policy_step_budget(args, {})
        self.assertEqual(budget, 900)
        self.assertEqual(source, "policy_steps")

    def test_matching_can_be_disabled(self) -> None:
        args = unpinned_args(policy_steps=900, match_teacher_budget=False)
        budget, source = resolve_policy_step_budget(args, {"steps": 1800})
        self.assertEqual(budget, 900)
        self.assertEqual(source, "policy_steps")

    def test_bogus_episode_lengths_are_ignored(self) -> None:
        args = unpinned_args(policy_steps=900)
        for bogus in (0, -1, True, "1800", None):
            with self.subTest(bogus=bogus):
                budget, source = resolve_policy_step_budget(args, {"steps": bogus})
                self.assertEqual(budget, 900)
                self.assertEqual(source, "policy_steps")


class FailureModeTest(unittest.TestCase):
    def test_success_needs_every_criterion(self) -> None:
        criteria = {name: True for name in CRITERION_NAMES}
        self.assertEqual(classify_failure(criteria, stop_step=120), "success")

    def test_labels_cover_each_observed_failure(self) -> None:
        base = {name: True for name in CRITERION_NAMES}
        cases = (
            ("discovered", -1, "target_never_seen"),
            ("reached_radius", 279, "premature_stop"),
            ("reached_radius", -1, "never_reached"),
            ("stop_latched", -1, "arrived_without_stop"),
            ("stable_attitude", -1, "unstable_attitude"),
            ("no_obstacle_contact", -1, "obstacle_contact"),
        )
        for broken, stop_step, expected in cases:
            with self.subTest(broken=broken, stop_step=stop_step):
                criteria = dict(base)
                criteria[broken] = False
                self.assertEqual(classify_failure(criteria, stop_step), expected)

    def test_every_label_is_declared(self) -> None:
        self.assertEqual(set(FAILURE_MODES) - {"success", "unknown"}, {
            "target_never_seen",
            "never_reached",
            "premature_stop",
            "arrived_without_stop",
            "unstable_attitude",
            "obstacle_contact",
        })

    def test_build_criteria_reads_the_episode_counters(self) -> None:
        criteria = build_criteria(
            target_first_visible_step=0,
            target_reached_step=546,
            stop_step=-1,
            stable=True,
            obstacle_contact_step_count=0,
        )
        self.assertEqual(
            criteria,
            {
                "discovered": True,
                "reached_radius": True,
                "stop_latched": False,
                "stable_attitude": True,
                "no_obstacle_contact": True,
            },
        )
        # Episode 6024 on 2026-10-03: inside the ring, upright, untouched, never
        # declared the stop - the label has to point at the stop decision alone.
        self.assertEqual(classify_failure(criteria, stop_step=-1), "arrived_without_stop")


class ObstacleContactToleranceTest(unittest.TestCase):
    """Zero tolerance is the historical protocol; relaxing it must touch only the
    contact clause, and the strict score must stay visible beside the ladder."""

    @staticmethod
    def _criteria(*, contact: int, tolerance: int | None = None, **overrides: object):
        kwargs: dict[str, object] = dict(
            target_first_visible_step=0,
            target_reached_step=546,
            stop_step=550,
            stable=True,
            obstacle_contact_step_count=contact,
        )
        kwargs.update(overrides)
        if tolerance is not None:
            kwargs["obstacle_contact_tolerance_steps"] = tolerance
        return build_criteria(**kwargs)

    def test_default_is_zero_tolerance(self) -> None:
        self.assertEqual(DEFAULT_OBSTACLE_CONTACT_TOLERANCE_STEPS, 0)
        self.assertFalse(self._criteria(contact=1)["no_obstacle_contact"])
        self.assertTrue(self._criteria(contact=0)["no_obstacle_contact"])

    def test_tolerance_only_moves_the_contact_clause(self) -> None:
        self.assertTrue(self._criteria(contact=10, tolerance=10)["no_obstacle_contact"])
        self.assertFalse(self._criteria(contact=11, tolerance=10)["no_obstacle_contact"])
        # A missing stop is still a missing stop at any tolerance.
        self.assertFalse(
            self._criteria(contact=0, tolerance=50, stop_step=-1)["stop_latched"]
        )

    def test_negative_tolerance_clamps_to_zero(self) -> None:
        self.assertFalse(self._criteria(contact=1, tolerance=-5)["no_obstacle_contact"])

    def test_config_fills_the_contact_tolerance(self) -> None:
        args = unpinned_args()
        apply_evaluation_config(
            args, {"smolvla_evaluation": {"obstacle_contact_tolerance_steps": 10}}
        )
        self.assertEqual(args.obstacle_contact_tolerance_steps, 10)

    def test_contact_tolerance_defaults_when_the_config_omits_it(self) -> None:
        args = unpinned_args()
        apply_evaluation_config(args, {"smolvla_evaluation": {"policy_steps": 650}})
        self.assertEqual(args.obstacle_contact_tolerance_steps, 0)

    def test_ladder_holds_the_other_four_criteria_fixed(self) -> None:
        # ep(0), ep(1), ep(7) pass up to 10 steps; ep(11) only at 20; ep(3) never,
        # because its stop criterion failed. The ladder must mirror exactly that.
        def episode(contact: int, core: bool = True) -> dict:
            return {
                "criteria": {name: core for name in CONTACT_TOLERANT_CORE_CRITERIA},
                "obstacle_contact_step_count": contact,
            }

        episodes = [episode(0), episode(1), episode(7), episode(11), episode(3, core=False)]
        self.assertEqual(
            count_successes_by_contact_tolerance(episodes, tolerances=(0, 10, 20)),
            {"0": 1, "10": 3, "20": 4},
        )


class TerminateAfterStopTest(unittest.TestCase):
    """A latched stop ends the episode, exactly as it ends a teacher episode."""

    def test_episode_keeps_running_before_the_hold_expires(self) -> None:
        self.assertFalse(
            terminate_after_stop(stop_latched=True, stop_step=992, step=1000, hold_steps=25)
        )

    def test_episode_ends_once_the_hold_expires(self) -> None:
        self.assertTrue(
            terminate_after_stop(stop_latched=True, stop_step=992, step=1017, hold_steps=25)
        )

    def test_no_latch_means_no_termination(self) -> None:
        # Episode 6024: inside the ring at 0.495 m, never voted to stop, so the
        # episode has to keep running - there is nothing to terminate on.
        self.assertFalse(
            terminate_after_stop(stop_latched=False, stop_step=-1, step=816, hold_steps=25)
        )

    def test_negative_hold_restores_run_to_budget(self) -> None:
        """Regression: the player used to ignore its own latch entirely.

        It kept commanding a zero-velocity stop until the budget ran out, which
        on 2026-10-03 left 94-1422 graded steps of pure standing. Every episode
        whose budget reached 1800 then recorded 8.9-16.3 degrees of pitch, while
        every episode that stopped inside a 1000-step budget stayed under 3.8.
        """
        self.assertFalse(
            terminate_after_stop(stop_latched=True, stop_step=378, step=1799, hold_steps=-1)
        )

    def test_zero_hold_ends_on_the_latch_step(self) -> None:
        self.assertTrue(
            terminate_after_stop(stop_latched=True, stop_step=378, step=378, hold_steps=0)
        )


if __name__ == "__main__":
    unittest.main()
