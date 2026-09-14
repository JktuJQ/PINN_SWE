"""
Structured grid for classical finite-volume / finite-difference solvers.
"""

from typing import Tuple

import numpy as np


class StructuredGrid2D:
    """Cell-centered uniform rectangular grid."""

    def __init__(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        Nx: int,
        Ny: int,
    ):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.Nx, self.Ny = Nx, Ny

        self.x = np.linspace(self.x_min, self.x_max, Nx + 1)
        self.y = np.linspace(self.y_min, self.y_max, Ny + 1)

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.xc = 0.5 * (self.x[:-1] + self.x[1:])
        self.yc = 0.5 * (self.y[:-1] + self.y[1:])

        self.X, self.Y = np.meshgrid(self.xc, self.yc, indexing="ij")

    @property
    def shape(self) -> Tuple[int, int]:
        return self.Nx, self.Ny
