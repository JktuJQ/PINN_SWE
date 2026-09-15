"""
Cone-focused sampling and residual-adaptive refinement.

The characteristic cone of the initial disturbance is where all the
non-trivial dynamics happen. Outside it the solution is exactly the
initial state, so collocation points there are wasted. Cone sampling
places a fraction of the interior points inside the cone, the rest
uniformly for coverage of the far field.

RAR (residual-adaptive refinement) goes one step further: at regular
intervals it samples a large pool of candidate points, evaluates the
PDE residual on each, and keeps the ones with the largest residual.
The network tells us where it is struggling; we listen.
"""

import math
from typing import Callable

import torch
import torch.nn as nn

from core import BaseProblem
from ..config import SamplingConfig
from .base import Batch, Tensor3
from .uniform import UniformSampler


class ConeSampler(UniformSampler):
    """Uniform + cone-focused sampling of interior points.

    ``cone_frac`` of the interior points are drawn inside the
    characteristic cone of the disturbance; the rest are uniform.
    Initial and boundary sampling is unchanged from the base class.
    """

    def sample(self, t_max: float) -> Batch:
        batch = super().sample(t_max)

        d = self.problem.domain
        n = self.cfg.n_interior
        n_cone = int(n * self.cfg.cone_frac)
        n_uni = n - n_cone

        x_u = self._uniform(n_uni, d.x_min, d.x_max)
        y_u = self._uniform(n_uni, d.y_min, d.y_max)
        t_u = self._uniform(n_uni, 0.0, t_max)

        t_c = self._uniform(n_cone, 0.0, t_max)
        x_lo, x_hi = self._cone_x_bounds(t_c)
        width = (x_hi - x_lo).clamp_min(0.0)
        x_c = x_lo + torch.rand_like(t_c) * width
        y_c = self._uniform(n_cone, d.y_min, d.y_max)

        interior: Tensor3 = (
            torch.cat([x_u, x_c]),
            torch.cat([y_u, y_c]),
            torch.cat([t_u, t_c]),
        )
        return Batch(interior=interior, ic=batch.ic, walls=batch.walls)

    def _cone_x_bounds(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Left and right x-limits of the cone at times ``t``.

        Falls back to the full domain if the problem does not report a
        finite wave speed (the default). The margin accounts for the
        fact that a PINN learns a slightly smeared front.
        """
        d = self.problem.domain
        speed = self.problem.max_wave_speed
        if math.isinf(speed):
            return torch.full_like(t, d.x_min), torch.full_like(t, d.x_max)

        margin = self.cfg.cone_margin
        half = speed * t + margin
        center = self.problem.disturbance_center
        x_lo = (center - half).clamp(d.x_min, d.x_max)
        x_hi = (center + half).clamp(d.x_min, d.x_max)
        return x_lo, x_hi


class ConeRARSampler(ConeSampler):
    """Cone sampling plus residual-adaptive refinement (RAR).

    Every ``rar_every`` epochs the trainer calls ``refine``; the sampler
    builds a large pool of candidate points, evaluates the PDE residual
    on each, and keeps the ``rar_keep`` with the largest residual. Those
    points are concatenated to the interior set for all subsequent
    epochs, until the next refinement replaces them.
    """

    def __init__(
        self,
        problem: BaseProblem,
        cfg: SamplingConfig,
        device: torch.device,
    ) -> None:
        super().__init__(problem, cfg, device)
        self._rar_points: Tensor3 | None = None

    def sample(self, t_max: float) -> Batch:
        batch = super().sample(t_max)
        if self._rar_points is not None:
            x, y, t = batch.interior
            xr, yr, tr = self._rar_points
            batch.interior = (
                torch.cat([x, xr]),
                torch.cat([y, yr]),
                torch.cat([t, tr]),
            )
        return batch

    def refine(
        self,
        model: nn.Module,
        residual_fn: Callable[..., torch.Tensor],
        t_max: float,
        nu: float,
    ) -> None:
        """Sample a pool, score by residual, keep the worst points.

        Note: ``torch.no_grad()`` is deliberately *not* used here. The
        residual needs autograd through the model to form spatial and
        temporal derivatives, and no_grad would break graph creation.
        The short-lived graph is released when local tensors go out of
        scope.
        """
        d = self.problem.domain
        n = self.cfg.rar_pool

        x = self._uniform(n, d.x_min, d.x_max)
        y = self._uniform(n, d.y_min, d.y_max)
        t = self._uniform(n, 0.0, t_max)

        R = residual_fn(model, x, y, t, nu=nu)
        score = (R**2).mean(dim=1)
        keep = min(self.cfg.rar_keep, n)
        idx = torch.topk(score, keep).indices

        self._rar_points = (
            x[idx].detach(),
            y[idx].detach(),
            t[idx].detach(),
        )
