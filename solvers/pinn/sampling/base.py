"""
Base types for point samplers.

A sampler produces a ``Batch`` of collocation, initial and boundary
points for one training step. Boundary points are grouped by wall name,
matching the keys accepted by ``BaseProblem.boundary_condition``
(``'x_min'``, ``'x_max'``, ``'y_min'``, ``'y_max'``), so that the loss
term can iterate over walls without hardcoding any geometry.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from core import BaseProblem


Tensor3 = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

WALLS: tuple[str, ...] = ("x_min", "x_max", "y_min", "y_max")


@dataclass
class Batch:
    """Points for a single training step.

    Attributes:
        interior: ``(x, y, t)`` collocation points for the PDE residual.
        ic:       ``(x, y, t)`` points on the initial surface.
        walls:    Mapping ``wall_name -> (x, y, t)`` of points on each
                  boundary wall.
    """

    interior: Tensor3
    ic: Tensor3
    walls: dict[str, Tensor3]


class BaseSampler(ABC):
    """Protocol for samplers producing a Batch for one training step."""

    def __init__(self, problem: BaseProblem, device: torch.device) -> None:
        self.problem = problem
        self.device = device

    @abstractmethod
    def sample(self, t_max: float) -> Batch:
        """Return a fresh batch of points for the current time horizon."""
        pass

    def refine(
        self,
        model: nn.Module,
        residual_fn: Callable[..., torch.Tensor],
        t_max: float,
        nu: float,
    ) -> None:
        """Optional hook, called by the trainer at regular intervals.

        The default implementation does nothing. Samplers that adapt the
        collocation set to the current residual (RAR) override this.
        """
        return None
