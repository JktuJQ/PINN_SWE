"""Vectorized HLL solver for the 2D Shallow Water Equations."""

from typing import Iterator, Optional, Any

import numpy as np

from core import BaseProblem, StructuredGrid2D
from . import fvm_swe


def _hll_flux_x(
    hL: np.ndarray,
    huL: np.ndarray,
    hvL: np.ndarray,
    hR: np.ndarray,
    huR: np.ndarray,
    hvR: np.ndarray,
    g: float,
) -> list[np.ndarray]:
    """HLL flux through vertical faces (normal = x)."""
    _, uL, vL = fvm_swe.conservative_to_primitive(hL, huL, hvL)
    _, uR, vR = fvm_swe.conservative_to_primitive(hR, huR, hvR)
    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))

    sL = np.minimum(uL - cL, uR - cR)
    sR = np.maximum(uL + cL, uR + cR)

    FL = fvm_swe.flux_x(hL, huL, hvL, g)
    FR = fvm_swe.flux_x(hR, huR, hvR, g)
    UL = (hL, huL, hvL)
    UR = (hR, huR, hvR)

    denom = np.where(np.abs(sR - sL) > 1e-14, sR - sL, 1.0)
    out = []
    for fL, fR, uL_i, uR_i in zip(FL, FR, UL, UR):
        f_hll = (sR * fL - sL * fR + sL * sR * (uR_i - uL_i)) / denom
        f = np.where(sL >= 0.0, fL, np.where(sR <= 0.0, fR, f_hll))
        out.append(f)
    return out


def _hll_flux_y(
    hL: np.ndarray,
    huL: np.ndarray,
    hvL: np.ndarray,
    hR: np.ndarray,
    huR: np.ndarray,
    hvR: np.ndarray,
    g: float,
) -> list[np.ndarray]:
    """HLL flux through horizontal faces (normal = y)."""
    _, uL, vL = fvm_swe.conservative_to_primitive(hL, huL, hvL)
    _, uR, vR = fvm_swe.conservative_to_primitive(hR, huR, hvR)
    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))

    sL = np.minimum(vL - cL, vR - cR)
    sR = np.maximum(vL + cL, vR + cR)

    FL = fvm_swe.flux_y(hL, huL, hvL, g)
    FR = fvm_swe.flux_y(hR, huR, hvR, g)
    UL = (hL, huL, hvL)
    UR = (hR, huR, hvR)

    denom = np.where(np.abs(sR - sL) > 1e-14, sR - sL, 1.0)
    out = []
    for fL, fR, uL_i, uR_i in zip(FL, FR, UL, UR):
        f_hll = (sR * fL - sL * fR + sL * sR * (uR_i - uL_i)) / denom
        f = np.where(sL >= 0.0, fL, np.where(sR <= 0.0, fR, f_hll))
        out.append(f)
    return out


def _faces_x(
    H: np.ndarray, HU: np.ndarray, HV: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Left/right states at each vertical face."""
    return (
        np.concatenate([H[:1, :], H], axis=0),
        np.concatenate([HU[:1, :], HU], axis=0),
        np.concatenate([HV[:1, :], HV], axis=0),
        np.concatenate([H, H[-1:, :]], axis=0),
        np.concatenate([HU, HU[-1:, :]], axis=0),
        np.concatenate([HV, HV[-1:, :]], axis=0),
    )


def _faces_y(
    H: np.ndarray, HU: np.ndarray, HV: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bottom/top states at each horizontal face."""
    return (
        np.concatenate([H[:, :1], H], axis=1),
        np.concatenate([HU[:, :1], HU], axis=1),
        np.concatenate([HV[:, :1], HV], axis=1),
        np.concatenate([H, H[:, -1:]], axis=1),
        np.concatenate([HU, HU[:, -1:]], axis=1),
        np.concatenate([HV, HV[:, -1:]], axis=1),
    )


def _step(
    H: np.ndarray,
    HU: np.ndarray,
    HV: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    g: float,
    problem: BaseProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One HLL update: U^{n+1} = U^n - dt/dx * dF - dt/dy * dG + dt * S."""
    Fh, Fhu, Fhv = _hll_flux_x(*_faces_x(H, HU, HV), g)
    Gh, Ghu, Ghv = _hll_flux_y(*_faces_y(H, HU, HV), g)

    H_new = H - dt / dx * (Fh[1:, :] - Fh[:-1, :]) - dt / dy * (Gh[:, 1:] - Gh[:, :-1])
    HU_new = (
        HU - dt / dx * (Fhu[1:, :] - Fhu[:-1, :]) - dt / dy * (Ghu[:, 1:] - Ghu[:, :-1])
    )
    HV_new = (
        HV - dt / dx * (Fhv[1:, :] - Fhv[:-1, :]) - dt / dy * (Ghv[:, 1:] - Ghv[:, :-1])
    )

    _, U, V = fvm_swe.conservative_to_primitive(H, HU, HV)
    sources = problem.sources(H, U, V, problem.parameters)

    H_new += dt * sources.get("h", 0.0)
    HU_new += dt * H * sources.get("u", 0.0)
    HV_new += dt * H * sources.get("v", 0.0)

    H_new = np.maximum(H_new, fvm_swe.H_MIN)
    return H_new, HU_new, HV_new


def solve(
    problem: BaseProblem,
    grid: StructuredGrid2D,
    t_end: float,
    cfl: float = 0.45,
    save_times: Optional[list[float]] = None,
    verbose: bool = False,
) -> Iterator[dict[str, Any]]:
    """Run the HLL solver as a generator, yielding snapshots at save_times.

    This is memory-efficient: only one snapshot is held in memory at a time.
    For large grids or many save_times, this prevents OOM errors.

    Args:
        problem:    Mathematical problem (provides IC and parameters like g).
        grid:       Structured 2D grid (provides dx, dy, X, Y).
        t_end:      Final simulation time.
        cfl:        CFL number (default 0.45 for 2D stability).
        save_times: Moments at which to yield snapshots. Defaults to [t_end].
        verbose:    Print progress every 50 steps.

    Yields:
        Dictionary with keys 'h', 'u', 'v', 't' at each save_time.
        Arrays have shape (Nx, Ny).

    Example:
        >>> for snap in solve(problem, grid, t_end=1.0, save_times=[0.5, 1.0]):
        ...     print(f"t={snap['t']:.2f}, max(h)={snap['h'].max():.3f}")

        >>> snapshots = list(solve(problem, grid, t_end=1.0))
    """
    g = problem.parameters.get("g", 9.81)

    ic = problem.initial_condition(grid.X, grid.Y)
    h0, u0, v0 = ic["h"], ic["u"], ic["v"]

    H, HU, HV = fvm_swe.primitive_to_conservative(h0, u0, v0)

    if save_times is None:
        save_times = [t_end]
    save_times = sorted(float(s) for s in save_times)

    next_save = 0
    t = 0.0
    dx, dy = grid.dx, grid.dy
    steps = 0

    while next_save < len(save_times) and save_times[next_save] <= 0.0:
        h_snap, u_snap, v_snap = fvm_swe.conservative_to_primitive(H, HU, HV)
        yield {
            "t": 0.0,
            "h": h_snap.copy(),
            "u": u_snap.copy(),
            "v": v_snap.copy(),
        }
        next_save += 1

    while t < t_end - 1e-15:
        _, U, V = fvm_swe.conservative_to_primitive(H, HU, HV)
        max_sx, max_sy = fvm_swe.max_wave_speeds(H, U, V, g)
        dt = cfl / (max_sx / dx + max_sy / dy)

        target = save_times[next_save] if next_save < len(save_times) else t_end
        dt = min(dt, target - t, t_end - t)

        H, HU, HV = _step(H, HU, HV, dx, dy, dt, g, problem)
        t += dt
        steps += 1

        while next_save < len(save_times) and t >= save_times[next_save] - 1e-12:
            h_snap, u_snap, v_snap = fvm_swe.conservative_to_primitive(H, HU, HV)
            yield {
                "h": h_snap.copy(),
                "u": u_snap.copy(),
                "v": v_snap.copy(),
                "t": t,
            }
            next_save += 1

        if verbose and steps % 50 == 0:
            print(f"  HLL t={t:.4f}/{t_end}  ({steps} steps)")

    if verbose:
        print(f"  HLL done: {steps} steps, grid {grid.Nx}x{grid.Ny}")


def solve_all(
    problem: BaseProblem,
    grid: StructuredGrid2D,
    t_end: float,
    cfl: float = 0.45,
    save_times: Optional[list[float]] = None,
    verbose: bool = False,
) -> tuple[list[dict[str, np.ndarray]], list[float]]:
    """Materialize all snapshots into a list (for metrics/plotting).

    This is a convenience wrapper around `solve()` for cases where you
    need all snapshots in memory (e.g., computing errors at multiple times).

    Returns:
        (snapshots, saved_times) where each snapshot is a dict
        {'h': array, 'u': array, 'v': array} of shape (Nx, Ny).
    """
    snapshots = []
    saved_times = []
    for snap in solve(problem, grid, t_end, cfl, save_times, verbose):
        saved_times.append(snap["t"])
        snapshots.append(
            {
                "h": snap["h"],
                "u": snap["u"],
                "v": snap["v"],
            }
        )
    return snapshots, saved_times
