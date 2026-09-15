"""
Time-horizon schedules.

The time horizon ``t_max`` grows during training: at the beginning the
network only has to fit the early-time behaviour; later it extends.
This avoids asking the network to guess the late-time state before it
has learned the early-time one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import CurriculumConfig


class Curriculum(ABC):
    """Maps ``(epoch, total_epochs)`` to a time horizon."""

    def __init__(self, t_start: float, t_end: float) -> None:
        self.t_start = t_start
        self.t_end = t_end

    @abstractmethod
    def t_max(self, epoch: int, total: int) -> float: ...


class ConstantCurriculum(Curriculum):
    """Horizon fixed at ``t_end`` from the very first epoch."""

    def t_max(self, epoch: int, total: int) -> float:
        return self.t_end


class TimeMarchingCurriculum(Curriculum):
    """Stepwise growth of the horizon over the first ``frac`` of training.

    The horizon is divided into ``stages`` equal pieces. During the
    first ``frac * total`` epochs the horizon grows by one stage every
    ``frac * total / stages`` epochs. After that it stays at ``t_end``.
    """

    def __init__(
        self,
        t_start: float,
        t_end: float,
        stages: int,
        frac: float,
    ) -> None:
        super().__init__(t_start, t_end)
        self.stages = stages
        self.frac = frac

    def t_max(self, epoch: int, total: int) -> float:
        growth_epochs = max(int(total * self.frac), 1)
        if epoch >= growth_epochs:
            return self.t_end
        stage = int(epoch / growth_epochs * self.stages)
        stage = min(stage, self.stages - 1)
        frac_done = (stage + 1) / self.stages
        return self.t_start + (self.t_end - self.t_start) * frac_done
