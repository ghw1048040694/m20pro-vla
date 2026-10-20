from __future__ import annotations

import os
import unittest

from m20pro_vla.low_level import (
    BACKEND_ENV,
    BACKENDS,
    DEFAULT_BACKEND,
    M20LowLevelController,
    M20V5PolicyController,
    backend_of,
    build_low_level_controller,
    resolve_backend,
)


class LowLevelBackendWiringTest(unittest.TestCase):
    def test_rl_expert_is_the_default_layer(self) -> None:
        # The analytic controller peaks at 41.8 deg pitch while turning; the VLA
        # inherited that as a "nodding" gait, so the trained expert is now the
        # default execution layer.
        self.assertEqual(DEFAULT_BACKEND, "v5")
        self.assertIs(BACKENDS[DEFAULT_BACKEND], M20V5PolicyController)

    def test_v5_subclasses_the_analytic_controller(self) -> None:
        # This is why backend_of() compares exact types: isinstance() would
        # report every v5 controller as the analytic one.
        self.assertTrue(issubclass(M20V5PolicyController, M20LowLevelController))

    def test_backend_of_uses_exact_types_not_isinstance(self) -> None:
        analytic = object.__new__(M20LowLevelController)
        expert = object.__new__(M20V5PolicyController)
        self.assertEqual(backend_of(analytic), "analytic")
        self.assertEqual(backend_of(expert), "v5")

    def test_backend_of_rejects_a_foreign_controller(self) -> None:
        with self.assertRaises(TypeError):
            backend_of(object())

    def test_explicit_backend_beats_the_environment(self) -> None:
        previous = os.environ.get(BACKEND_ENV)
        os.environ[BACKEND_ENV] = "analytic"
        try:
            self.assertEqual(resolve_backend("v5"), "v5")
            self.assertEqual(resolve_backend(None), "analytic")
            self.assertEqual(resolve_backend(), "analytic")
        finally:
            if previous is None:
                os.environ.pop(BACKEND_ENV, None)
            else:
                os.environ[BACKEND_ENV] = previous

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown low-level backend"):
            build_low_level_controller(object(), "v99")


if __name__ == "__main__":
    unittest.main()
