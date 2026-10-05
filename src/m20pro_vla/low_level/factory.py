"""Select the M20 execution layer by name.

Call sites keep calling ``reset`` / ``step`` exactly as before; only the class
behind the body-command interface changes.

The RL expert (``v5``) is the default as of 2026-10-03. The analytic controller
walks with a 41.8 deg pitch peak while turning, which is what made the VLA look
like it was "nodding"; the v5 expert holds 3.94 deg on the same command. Any run
can still opt back out - by argument or by setting ``M20_LOW_LEVEL_BACKEND``.
"""

from __future__ import annotations

import os

import mujoco

from .controller import M20LowLevelController
from .policy_v5 import M20V5PolicyController

BACKENDS = {
    "analytic": M20LowLevelController,
    "v5": M20V5PolicyController,
}
DEFAULT_BACKEND = "v5"
BACKEND_ENV = "M20_LOW_LEVEL_BACKEND"


def resolve_backend(backend: str | None = None) -> str:
    if backend:
        return backend
    return os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND


def build_low_level_controller(
    model: mujoco.MjModel,
    backend: str | None = None,
    **kwargs,
) -> M20LowLevelController:
    name = resolve_backend(backend)
    try:
        controller_class = BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown low-level backend {name!r}; choose one of {tuple(BACKENDS)}"
        ) from None
    return controller_class(model, **kwargs)


def backend_of(controller: M20LowLevelController) -> str:
    """Name of the backend that produced this exact controller instance.

    ``M20V5PolicyController`` subclasses ``M20LowLevelController``, so this
    compares exact types rather than using ``isinstance`` - otherwise every v5
    controller would be reported as the analytic one.
    """
    for name, controller_class in BACKENDS.items():
        if type(controller) is controller_class:
            return name
    raise TypeError(
        f"{type(controller).__name__} is not a registered low-level backend; "
        f"choose one of {tuple(BACKENDS)}"
    )


__all__ = [
    "BACKENDS",
    "BACKEND_ENV",
    "DEFAULT_BACKEND",
    "backend_of",
    "build_low_level_controller",
    "resolve_backend",
]
