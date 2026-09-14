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
    """Extract 1D profile (x, h, u, v) from FVM snapshot at y = y_val.

    Args:
        snapshot: Dictionary {'h': (Nx,Ny), 'u': (Nx,Ny), 'v': (Nx,Ny)}.
        grid: Structured grid (provides X, Y, xc, yc).
        y_val: Y-coordinate for the slice. Defaults to 0.0.

    Returns:
        (x, h, u, v) as 1D NumPy arrays of length Nx.
    """
    j = np.argmin(np.abs(grid.yc - y_val))

    x = grid.xc
    h = snapshot["h"][:, j]
    u = snapshot["u"][:, j]
    v = snapshot["v"][:, j]

    return x, h, u, v


def extract_2d_from_fvm(
    snapshot: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Extract 2D fields from FVM snapshot (identity, for API consistency).

    Args:
        snapshot: Dictionary {'h': (Nx,Ny), 'u': (Nx,Ny), 'v': (Nx,Ny)}.

    Returns:
        The same dictionary (no transformation needed for FVM).
    """
    return snapshot


def evaluate_pinn_1d(
    model: torch.nn.Module,
    t: float,
    x_range: tuple[float, float],
    y_val: float = 0.0,
    n_points: int = 500,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate PINN model along y = y_val at time t.

    Args:
        model: PyTorch model mapping (x, y, t) -> (h, u, v).
        t: Time at which to evaluate.
        x_range: (x_min, x_max) for the profile.
        y_val: Y-coordinate for the slice. Defaults to 0.0.
        n_points: Number of points along x. Defaults to 500.
        device: Device for evaluation. Defaults to CPU.

    Returns:
        (x, h, u, v) as 1D NumPy arrays of length n_points.
    """
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
    device: torch.device = torch.device("cpu"),
) -> dict[str, np.ndarray]:
    """Evaluate PINN model on a 2D grid at time t.

    Args:
        model: PyTorch model mapping (x, y, t) -> (h, u, v).
        t: Time at which to evaluate.
        x_range: (x_min, x_max).
        y_range: (y_min, y_max).
        n_points: Number of points in each direction. Defaults to 200.
        device: Device for evaluation. Defaults to CPU.

    Returns:
        Dictionary {'h': (n_points, n_points), 'u': ..., 'v': ...} as NumPy arrays.
    """
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
