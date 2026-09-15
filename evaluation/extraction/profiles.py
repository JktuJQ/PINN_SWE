"""
Extract 1D profiles and 2D fields from FVM snapshots and PINN models.

This module bridges the gap between solvers (which return different formats)
and evaluation tools (which expect unified NumPy arrays).
"""

import numpy as np
import torch

from core import StructuredGrid2D


def extract_1d_from_fvm(
    snapshot: dict[str, np.ndarray],
    grid: StructuredGrid2D,
    y_val: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract 1D profile ``(x, h, u, v)`` from an FVM snapshot at ``y = y_val``.

    Picks the nearest row of cells to ``y_val``.
    """
    j = int(np.argmin(np.abs(grid.yc - y_val)))

    x = grid.xc
    h = snapshot["h"][:, j]
    u = snapshot["u"][:, j]
    v = snapshot["v"][:, j]

    return x, h, u, v


def evaluate_pinn_1d(
    model: torch.nn.Module,
    t: float,
    x_range: tuple[float, float],
    y_val: float = 0.0,
    n_points: int = 500,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate PINN along ``y = y_val`` at time ``t``.

    If ``device`` is None, uses the device of the model's first parameter.
    """
    if device is None:
        device = next(model.parameters()).device

    x = torch.linspace(x_range[0], x_range[1], n_points, device=device).view(-1, 1)
    y = torch.full_like(x, y_val)
    tt = torch.full_like(x, float(t))

    with torch.no_grad():
        h, u, v = model(x, y, tt)

    return (
        x.cpu().numpy().ravel(),
        h.cpu().numpy().ravel(),
        u.cpu().numpy().ravel(),
        v.cpu().numpy().ravel(),
    )


def evaluate_pinn_2d(
    model: torch.nn.Module,
    t: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_points: int = 200,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate PINN on an ``n_points × n_points`` grid at time ``t``.

    If ``device`` is None, uses the device of the model's first parameter.
    """
    if device is None:
        device = next(model.parameters()).device

    xs = torch.linspace(x_range[0], x_range[1], n_points, device=device)
    ys = torch.linspace(y_range[0], y_range[1], n_points, device=device)
    XX, YY = torch.meshgrid(xs, ys, indexing="ij")

    flat_x = XX.reshape(-1, 1)
    flat_y = YY.reshape(-1, 1)
    flat_t = torch.full_like(flat_x, float(t))

    with torch.no_grad():
        h, u, v = model(flat_x, flat_y, flat_t)

    return {
        "h": h.reshape(n_points, n_points).cpu().numpy(),
        "u": u.reshape(n_points, n_points).cpu().numpy(),
        "v": v.reshape(n_points, n_points).cpu().numpy(),
    }
