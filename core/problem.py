"""
Abstract base class for mathematical problem statements.

A Problem defines the 'WHAT': the domain, the state variables,
the initial/boundary conditions, physical sources, and the exact solution (if any).
It is completely agnostic to the 'HOW' (no numerical fluxes, no neural networks, no autograd).
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .domain import RectangularDomain


class BaseProblem(ABC):
    """Pure mathematical formulation of a PDE problem."""

    def __init__(self, domain: RectangularDomain) -> None:
        self.domain = domain

    @property
    @abstractmethod
    def state_variables(self) -> tuple[str, ...]:
        """Ordered names of the state variables (e.g., ('h', 'u', 'v'))."""
        pass

    @property
    def parameters(self) -> dict[str, Any]:
        """Physical parameters (e.g., gravity, dam heights, Manning's n)."""
        return {}

    @abstractmethod
    def initial_condition(self, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate the initial state at t = 0.

        Returns:
            Dictionary mapping state variable names to their initial values.
        """
        pass

    @abstractmethod
    def boundary_condition(
        self,
        wall: str,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Boundary conditions on a single wall of the domain.

        Args:
            wall: One of ``'x_min'``, ``'x_max'``, ``'y_min'``, ``'y_max'``.
            x, y, t: 1D arrays of the same shape giving coordinates of the
                sample points on this wall. ``x`` and ``y`` may be constant
                along the wall (e.g. ``x`` is fixed on ``x_min``).

        Returns:
            Nested dictionary ``{condition_type: {variable: value}}``.

            Supported condition types:
              * ``'dirichlet'``: prescribed value of the variable.

            **A variable that is absent from the inner dictionary is left
            free.** This is how the caller distinguishes "prescribed to
            zero" from "not constrained at all".

        Examples:
            Slip wall (v = 0, h and u free)::

                >>> {"dirichlet": {"v": np.zeros_like(x)}}

            Far-field Dirichlet (all three variables prescribed)::

                >>> {"dirichlet": {
                ...     "h": np.full_like(x, 2.0),
                ...     "u": np.zeros_like(x),
                ...     "v": np.zeros_like(x),
                ... }}

            No condition on this wall::

                >>> {}
        """
        pass

    def exact_solution(
        self, x: np.ndarray, y: np.ndarray, t: float
    ) -> dict[str, np.ndarray] | None:
        """Evaluate the exact solution at time ``t`` (if available).

        Args:
            x, y: Arrays of the same shape with spatial coordinates.
                ``y`` may be ignored for problems that are y-invariant.
            t: Time.

        Returns:
            Dictionary mapping state variable names to exact values, or
            ``None`` if no analytical solution exists.
        """
        return None

    def sources(
        self,
        h: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        params: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute physical source terms (topography, friction, rainfall).

        Returns sources in PRIMITIVE variables, all of the same shape as ``h``:

          * ``'mass'``:    rate of change of water depth (dh/dt), e.g. rainfall.
          * ``'accel_x'``: acceleration in x (du/dt), e.g. ``-g*z_x``, friction.
          * ``'accel_y'``: acceleration in y (dv/dt), e.g. ``-g*z_y``, friction.

        Solvers are responsible for integrating these into their update steps
        (e.g. HLL multiplies accelerations by ``h`` to obtain momentum sources;
        PINN subtracts them from the primitive residual).

        The default implementation returns zero sources.
        """
        zeros = np.zeros_like(h)
        return {"mass": zeros, "accel_x": zeros, "accel_y": zeros}
