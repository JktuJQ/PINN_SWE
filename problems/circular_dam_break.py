"""
Circular dam break: cylindrical column of water on a wet bed.

Initial state:
    h = h_in,  u = v = 0   for r = sqrt(x^2 + y^2) <= r_dam
    h = h_out, u = v = 0   for r > r_dam

The solution is radially symmetric: an outward-propagating shock and
an inward-propagating rarefaction that eventually interacts at the
origin. There is no closed-form solution, so the reference for
validation is the HLL solver (or a 1D radial finite-volume scheme).
"""

import math
from typing import Any

import numpy as np

from core import BaseProblem, RectangularDomain


class CircularDamBreak(BaseProblem):
    """Circular dam break over a wet bed (radially symmetric 2D problem)."""

    def __init__(
        self,
        domain: RectangularDomain,
        h_in: float = 2.0,
        h_out: float = 1.0,
        r_dam: float = 1.0,
        g: float = 9.81,
    ) -> None:
        super().__init__(domain)
        if h_in <= 0.0 or h_out <= 0.0:
            raise ValueError(
                f"Water depths must be positive, got h_in={h_in}, h_out={h_out}"
            )
        if r_dam <= 0.0:
            raise ValueError(f"Dam radius must be positive, got r_dam={r_dam}")
        self.h_in = h_in
        self.h_out = h_out
        self.r_dam = r_dam
        self.g = g

    @property
    def state_variables(self) -> tuple[str, ...]:
        return "h", "u", "v"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "g": self.g,
            "h_in": self.h_in,
            "h_out": self.h_out,
            "r_dam": self.r_dam,
        }

    @property
    def max_wave_speed(self) -> float:
        return math.sqrt(self.g * max(self.h_in, self.h_out))

    @property
    def disturbance_center(self) -> float:
        return 0.0

    @property
    def disturbance_is_radial(self) -> bool:
        return True

    def initial_condition(self, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        r = np.sqrt(x * x + y * y)
        h0 = np.where(r <= self.r_dam, self.h_in, self.h_out).astype(float)
        return {
            "h": h0,
            "u": np.zeros_like(x, dtype=float),
            "v": np.zeros_like(x, dtype=float),
        }

    def boundary_condition(
        self,
        wall: str,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Far-field Dirichlet on all four walls.

        As long as the domain is large enough that the outgoing shock
        does not reach the boundary within ``t_max``, the true solution
        on every wall is exactly the undisturbed state
        ``(h_out, 0, 0)``. Not an approximation.

        Sufficient condition:
            r_dam + sqrt(g * h_in) * t_max < min(|x_min|, |y_min|)
        """
        if wall not in ("x_min", "x_max", "y_min", "y_max"):
            raise ValueError(f"Unknown wall: {wall!r}")
        h_out = np.full_like(x, self.h_out, dtype=float)
        zeros = np.zeros_like(x, dtype=float)
        return {"dirichlet": {"h": h_out, "u": zeros, "v": zeros}}

    def exact_solution(
        self, x: np.ndarray, y: np.ndarray, t: float
    ) -> dict[str, np.ndarray] | None:
        return None
