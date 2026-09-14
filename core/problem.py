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
        """Physical parameters (e.g., gravity, dam heights, Manning's n).

        Solvers (HLL, PINN) can read these to configure their internal operators.
        """
        return {}

    @abstractmethod
    def initial_condition(self, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate the initial state at t = 0.

        Returns:
            Dictionary mapping state variable names to their initial
            values (e.g., {'h': array, 'u': array, 'v': array}).
        """
        pass

    @abstractmethod
    def boundary_condition(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        normal_x: np.ndarray,
        normal_y: np.ndarray,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Evaluate boundary conditions on the domain walls.

        The normal vectors allow the implementation to distinguish
        wall orientation without hardcoding positions.

        Returns:
            Nested dictionary. Outer keys are condition types
            ('dirichlet', 'neumann'). Inner dictionaries map
            state-variable names to their prescribed values.
        """
        pass

    def exact_solution(self, x: np.ndarray, t: float) -> dict[str, np.ndarray] | None:
        """Evaluate the exact solution at time t (if available).

        Returns:
            Dictionary mapping state variable names to exact values,
            or None if no analytical solution exists (e.g., for
            problems with friction or topography).
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

        Returns sources in PRIMITIVE variables:
          - 'h': mass source (e.g., rainfall intensity R)
          - 'u': acceleration in x-direction (e.g., -g*z_x, Manning friction)
          - 'v': acceleration in y-direction (e.g., -g*z_y, Manning friction)

        Default implementation returns zero sources (inviscid SWE without external forces).
        Concrete problems can override this to add physical effects.

        NOTE: Solvers are responsible for integrating these sources into
        their specific update steps (e.g., HLL multiplies accelerations
        by h to get momentum sources; PINN subtracts them from the
        primitive residual).
        """
        return {var: np.zeros_like(h) for var in self.state_variables}
