"""Uniform sampling over the whole domain."""

import torch

from core import BaseProblem
from ..config import SamplingConfig
from .base import Batch, BaseSampler, Tensor3, WALLS


class UniformSampler(BaseSampler):
    """Uniform sampling of interior, initial and wall points.

    All points are drawn from a uniform distribution over the domain.
    This is the simplest strategy and the baseline against which
    cone-focused sampling is compared.
    """

    def __init__(
        self,
        problem: BaseProblem,
        cfg: SamplingConfig,
        device: torch.device,
    ) -> None:
        super().__init__(problem, device)
        self.cfg = cfg

    def _uniform(self, n: int, lo: float, hi: float) -> torch.Tensor:
        """Column tensor of shape (n, 1), uniform in [lo, hi)."""
        return torch.rand(n, 1, device=self.device) * (hi - lo) + lo

    def _wall_points(self, wall: str, n: int, t_max: float) -> Tensor3:
        """Points on a single wall. The fixed coordinate is exact."""
        d = self.problem.domain
        t = self._uniform(n, 0.0, t_max)

        if wall == "x_min":
            x = torch.full((n, 1), d.x_min, device=self.device)
            y = self._uniform(n, d.y_min, d.y_max)
        elif wall == "x_max":
            x = torch.full((n, 1), d.x_max, device=self.device)
            y = self._uniform(n, d.y_min, d.y_max)
        elif wall == "y_min":
            x = self._uniform(n, d.x_min, d.x_max)
            y = torch.full((n, 1), d.y_min, device=self.device)
        elif wall == "y_max":
            x = self._uniform(n, d.x_min, d.x_max)
            y = torch.full((n, 1), d.y_max, device=self.device)
        else:
            raise ValueError(f"Unknown wall: {wall!r}")

        return x, y, t

    def sample(self, t_max: float) -> Batch:
        d = self.problem.domain

        n = self.cfg.n_interior
        interior: Tensor3 = (
            self._uniform(n, d.x_min, d.x_max),
            self._uniform(n, d.y_min, d.y_max),
            self._uniform(n, 0.0, t_max),
        )

        n_ic = self.cfg.n_ic
        ic: Tensor3 = (
            self._uniform(n_ic, d.x_min, d.x_max),
            self._uniform(n_ic, d.y_min, d.y_max),
            torch.zeros(n_ic, 1, device=self.device),
        )

        walls = {w: self._wall_points(w, self.cfg.n_bc, t_max) for w in WALLS}

        return Batch(interior=interior, ic=ic, walls=walls)
