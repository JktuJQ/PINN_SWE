"""
Loss terms and the context they need to compute themselves.

A loss term is a small object with a ``compute`` method. It receives the
model, the current batch of points, and a ``LossContext`` carrying
everything else that is fixed during training: the problem, the PDE
residual function, the current artificial viscosity, and the current
time horizon.

The design keeps loss terms orthogonal to the training loop: adding a
new term (mass conservation, entropy, symmetry, ...) means writing a
class and listing it in the config, without touching the trainer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, ClassVar

import torch
import torch.nn as nn

from core import BaseProblem
from ..sampling.base import Batch


@dataclass
class LossContext:
    """Everything a loss term needs beyond the model and the batch.

    Attributes:
        problem:      Mathematical problem (IC, BC, parameters).
        residual_fn:  Callable ``(model, x, y, t, nu) -> (N, 3)`` that
                      returns the PDE residual. Passed in rather than
                      built here so the trainer controls viscosity.
        nu:           Current artificial viscosity.
        t_max:        Current time horizon (for curriculum).
        device:       Device on which everything lives.
    """

    problem: BaseProblem
    residual_fn: Callable[..., torch.Tensor]
    nu: float
    t_max: float
    device: torch.device


class LossTerm(ABC):
    """One additive contribution to the training objective.

    Subclasses must set the class-level ``name`` attribute (used as a
    key in the weights dictionary and in the history log) and implement
    ``compute``.
    """

    name: ClassVar[str]

    @abstractmethod
    def compute(
        self,
        model: nn.Module,
        batch: Batch,
        ctx: LossContext,
    ) -> torch.Tensor:
        """Return a scalar tensor (mean squared error over the points)."""
