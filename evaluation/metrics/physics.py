"""
Physics-specific metrics for shallow water equations.

These metrics measure physical properties like shock sharpness, y-invariance and mass conservation.
"""

import numpy as np

from core import BaseProblem, StructuredGrid2D


def shock_sharpness(x: np.ndarray, h: np.ndarray) -> float:
    """Width of the shock front at 10–90 % of the jump magnitude.

    Finds the shock location by the maximum gradient of ``h(x)``, then
    measures the width of the front between 10 % and 90 % of the local
    jump. For an exact (discontinuous) solution the width is 0.

    Args:
        x: 1D array of x-coordinates (sorted).
        h: 1D array of water depth values.

    Returns:
        Front width ``|x_10 − x_90|``, or NaN if the front cannot be
        localized (fewer than 5 points in the search window, or the
        jump is numerically flat).
    """
    if len(x) < 5:
        return float("nan")

    dh_dx = np.gradient(h, x)
    shock_idx = int(np.argmax(np.abs(dh_dx)))
    x_shock = x[shock_idx]

    window = (x > x_shock - 2.0) & (x < x_shock + 2.0)
    if window.sum() < 5:
        return float("nan")

    order = np.argsort(x[window])
    xs, hs = x[window][order], h[window][order]

    h_left = hs[0]
    h_right = hs[-1]
    jump = abs(h_right - h_left)
    if jump < 1e-6:
        return float("nan")

    if h_left > h_right:
        h_90 = h_left - 0.90 * jump
        h_10 = h_left - 0.10 * jump
    else:
        h_10 = h_left + 0.10 * jump
        h_90 = h_left + 0.90 * jump

    def crossing(level: float) -> float | None:
        d = hs - level
        sign_change = np.where(np.diff(np.sign(d)) != 0)[0]
        if len(sign_change) == 0:
            return None
        i = int(sign_change[0])
        y0, y1 = d[i], d[i + 1]
        if y1 == y0:
            return float(xs[i])
        return float(xs[i] - y0 * (xs[i + 1] - xs[i]) / (y1 - y0))

    x90 = crossing(h_90)
    x10 = crossing(h_10)
    if x90 is None or x10 is None:
        return float("nan")

    return abs(x10 - x90)


def y_invariance_error(field_2d: np.ndarray) -> float:
    """Maximum deviation from y-invariance: max|h(x,y) - <h>_y|.

    For the 1D dam break problem embedded in 2D, the exact solution does not
    depend on y. This metric measures how much the numerical solution "wobbles"
    in the y-direction, which is a sign of 2D artifacts or poor boundary
    conditions.

    Args:
        field_2d: 2D array of shape (Nx, Ny) representing h(x, y).

    Returns:
        Maximum absolute deviation from the y-averaged profile.
    """
    h_mean_y = field_2d.mean(axis=1, keepdims=True)
    deviation = field_2d - h_mean_y
    return float(np.max(np.abs(deviation)))


def mass_conservation(
    field_2d: np.ndarray,
    grid: StructuredGrid2D,
    initial_mass: float | None = None,
) -> float:
    """Compute the total mass (integral of h over the domain).

    For the shallow water equations without sources, mass should be conserved:
    integral of h(x,y,t) dx dy = const.

    Args:
        field_2d: 2D array of shape (Nx, Ny) representing h(x, y, t).
        grid: Structured grid (provides dx, dy).
        initial_mass: If provided, returns the relative change in mass:
                      |M(t) - M(0)| / M(0). Otherwise returns M(t).

    Returns:
        Total mass M(t) = ∫∫ h dx dy, or relative mass error if initial_mass
        is provided.
    """
    mass = np.sum(field_2d) * grid.dx * grid.dy

    if initial_mass is not None:
        if initial_mass < 1e-14:
            return 0.0
        return float(abs(mass - initial_mass) / initial_mass)

    return float(mass)


def compute_initial_mass(
    problem: BaseProblem,
    grid: StructuredGrid2D,
) -> float:
    """Compute the initial mass M(0) = integral of h(x,y,0) dx dy.

    Args:
        problem: Mathematical problem (provides initial_condition).
        grid: Structured grid.

    Returns:
        Initial mass.
    """
    ic = problem.initial_condition(grid.X, grid.Y)
    h0 = ic["h"]
    return float(np.sum(h0) * grid.dx * grid.dy)
