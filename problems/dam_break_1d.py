"""
1D Dam Break problem (Riemann problem) for the 2D Shallow Water Equations.

The problem is 1D in space (invariant along y) but is embedded in a 2D
rectangular domain. This makes it a good testbed for 2D solvers (FVM, PINN)
that must recover the y-invariance of the exact solution.
"""

import math
from typing import Any

import numpy as np
from scipy.optimize import brentq

from core import BaseProblem, RectangularDomain


class DamBreak1D(BaseProblem):
    """Dam break over a wet bed (1D Riemann problem embedded in 2D).

    Initial state:
        h = h_l, u = 0, v = 0  for x <= x_dam
        h = h_r, u = 0, v = 0  for x >  x_dam

    The solution consists of a left-going rarefaction wave, a constant
    star region, and a right-going shock wave.
    """

    def __init__(
        self,
        domain: RectangularDomain,
        h_l: float = 2.0,
        h_r: float = 1.0,
        x_dam: float = 0.0,
        g: float = 9.81,
    ):
        super().__init__(domain)
        if h_l <= 0.0 or h_r <= 0.0:
            raise ValueError(f"Water depths must be positive, got h_l={h_l}, h_r={h_r}")
        self.h_l = h_l
        self.h_r = h_r
        self.x_dam = x_dam
        self.g = g

    @property
    def state_variables(self) -> tuple[str, ...]:
        return "h", "u", "v"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "g": self.g,
            "h_l": self.h_l,
            "h_r": self.h_r,
            "x_dam": self.x_dam,
        }

    @property
    def max_wave_speed(self) -> float:
        """Speed of the head of the rarefaction fan, ``sqrt(g * h_l)``.

        The rarefaction head is the fastest signal; the shock travels
        somewhat slower. Beyond ``max_wave_speed * t`` the solution is
        still exactly the initial state.
        """
        return math.sqrt(self.g * max(self.h_l, self.h_r))

    @property
    def disturbance_center(self) -> float:
        return self.x_dam

    def initial_condition(self, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        """Step function for h, zero velocity everywhere."""
        h0 = np.where(x <= self.x_dam, self.h_l, self.h_r).astype(float)
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
        """Boundary conditions on a single wall.

        - x-walls (``'x_min'``, ``'x_max'``): far-field Dirichlet. Within
          ``t_max`` the fastest wave (the head of the rarefaction fan,
          speed ``sqrt(g*h_l)``) does not reach the boundary, so ``h`` is
          exactly the initial depth on that side and both velocities are
          zero. This is *not* an approximation.
        - y-walls (``'y_min'``, ``'y_max'``): free-slip. Only the normal
          velocity ``v`` is constrained to zero. ``h`` and ``u`` are left
          free because the true solution has ``u != 0`` along these walls
          (it does not depend on ``y``); prescribing ``u = 0`` there would
          contradict the PDE.

        Variables absent from the returned dictionary are left free by the
        caller.
        """
        if wall in ("x_min", "x_max"):
            h_far = np.where(x <= self.x_dam, self.h_l, self.h_r).astype(float)
            return {
                "dirichlet": {
                    "h": h_far,
                    "u": np.zeros_like(x, dtype=float),
                    "v": np.zeros_like(x, dtype=float),
                }
            }

        if wall in ("y_min", "y_max"):
            return {
                "dirichlet": {
                    "v": np.zeros_like(x, dtype=float),
                }
            }

        raise ValueError(f"Unknown wall: {wall!r}")

    def exact_solution(
        self, x: np.ndarray, y: np.ndarray, t: float
    ) -> dict[str, np.ndarray] | None:
        """Exact 1D Riemann solution at time ``t``. ``y`` is ignored.

        Reference: E.F. Toro, "Shock-Capturing Methods for Free-Surface
        Shallow Flows", Chapter 5.
        """
        x = np.asarray(x, dtype=float)
        h = np.empty_like(x)
        u = np.empty_like(x)

        if t <= 0:
            h[:] = np.where(x <= self.x_dam, self.h_l, self.h_r)
            u[:] = 0.0
            return {"h": h, "u": u, "v": np.zeros_like(x)}

        h_star, u_star = self._solve_star_region()
        c_l = math.sqrt(self.g * self.h_l)
        c_star = math.sqrt(self.g * h_star)

        xi = (x - self.x_dam) / t

        s_head = -c_l
        s_tail = u_star - c_star
        s_shock = (h_star * u_star) / (h_star - self.h_r)

        left = xi <= s_head
        fan = (xi > s_head) & (xi < s_tail)
        star = (xi >= s_tail) & (xi < s_shock)
        right = xi >= s_shock

        h[left], u[left] = self.h_l, 0.0
        h[star], u[star] = h_star, u_star
        h[right], u[right] = self.h_r, 0.0

        c_fan = (2.0 * c_l - xi[fan]) / 3.0
        u[fan] = (2.0 * c_l + 2.0 * xi[fan]) / 3.0
        h[fan] = c_fan**2 / self.g

        return {"h": h, "u": u, "v": np.zeros_like(x)}

    def _solve_star_region(self) -> tuple[float, float]:
        """Find ``(h*, u*)`` in the star region using Brent's method.

        The star region is bracketed by the left rarefaction tail and the
        right shock. The matching condition is that the left and right
        wave branches produce equal velocities at the star state.
        """

        def shock_branch(h_star: float, h_k: float) -> float:
            return (h_star - h_k) * math.sqrt(
                0.5 * self.g * (h_star + h_k) / (h_star * h_k)
            )

        def rarefaction_branch(h_star: float, h_k: float) -> float:
            return 2.0 * (math.sqrt(self.g * h_star) - math.sqrt(self.g * h_k))

        def f(h_star: float, h_k: float) -> float:
            return (
                shock_branch(h_star, h_k)
                if h_star > h_k
                else rarefaction_branch(h_star, h_k)
            )

        def total(h_star: float) -> float:
            return f(h_star, self.h_l) + f(h_star, self.h_r)

        h_star = brentq(
            total,
            np.float64(1e-12),
            np.float64(1e6 * max(self.h_l, self.h_r)),
            xtol=1e-14,
            rtol=np.float64(1e-14),
            full_output=False,
        )
        u_star = 0.5 * (f(h_star, self.h_r) - f(h_star, self.h_l))  # type: ignore

        return float(h_star), float(u_star)  # type: ignore
