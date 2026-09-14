"""
Geometric bounding box for the computational domain.
"""

from typing import Tuple


class RectangularDomain:
    """
    Axis-aligned rectangular domain in (x, y, t).
    """

    def __init__(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        t_range: Tuple[float, float],
    ):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.t_min, self.t_max = t_range

    @property
    def spatial_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (self.x_min, self.x_max), (self.y_min, self.y_max)

    @property
    def time_bounds(self) -> Tuple[float, float]:
        return self.t_min, self.t_max
