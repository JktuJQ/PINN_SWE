"""
Artificial viscosity schedules.

A smooth network cannot represent a discontinuity, but it can represent
the smooth solution of the viscous problem whose front thickness is
``O(nu)``. Annealing ``nu`` down drives the solution toward the
inviscid (entropy) one, with progressively sharper fronts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

from ..config import ViscosityConfig


class ViscositySchedule(ABC):
    """Maps ``(epoch, total_epochs)`` to the current artificial viscosity."""

    @abstractmethod
    def nu(self, epoch: int, total: int) -> float: ...


class NoViscosity(ViscositySchedule):
    def nu(self, epoch: int, total: int) -> float:
        return 0.0


class ConstantViscosity(ViscositySchedule):
    def __init__(self, value: float) -> None:
        self.value = value

    def nu(self, epoch: int, total: int) -> float:
        return self.value


class AnnealedViscosity(ViscositySchedule):
    """Geometric decay from ``start`` to ``end`` over ``frac`` of training.

    After the annealing window the value stays at ``end``. Geometric
    decay (rather than linear) matches the way the front thickness
    scales with ``nu``.
    """

    def __init__(
        self,
        start: float,
        end: float,
        frac: float,
    ) -> None:
        if start <= 0.0 or end <= 0.0:
            raise ValueError("viscosity values must be positive")
        self.start = start
        self.end = end
        self.frac = frac

    def nu(self, epoch: int, total: int) -> float:
        window = max(int(total * self.frac), 1)
        if epoch >= window:
            return self.end
        a = epoch / window
        return float(self.start * (self.end / self.start) ** a)
