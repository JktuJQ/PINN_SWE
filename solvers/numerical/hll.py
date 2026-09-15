"""
Vectorized HLL solver for the 2D Shallow Water Equations.
"""

from typing import Any, Iterator, Optional

import numpy as np

from core import BaseProblem, StructuredGrid2D
from . import fvm_swe


def _hll_flux(
    hL: np.ndarray,
    huL: np.ndarray,
    hvL: np.ndarray,
    hR: np.ndarray,
    huR: np.ndarray,
    hvR: np.ndarray,
    g: float,
    axis: str,
) -> list[np.ndarray]:
    """HLL flux through a set of faces. ``axis`` is ``'x'`` or ``'y'``.

    Inputs are arrays of the same shape giving left/right states at each
    face. Returns a list ``[F_h, F_hu, F_hv]`` of the same shape.
    """
    _, uL, vL = fvm_swe.conservative_to_primitive(hL, huL, hvL)
    _, uR, vR = fvm_swe.conservative_to_primitive(hR, huR, hvR)
    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))

    if axis == "x":
        wL, wR = uL, uR
        FL = fvm_swe.flux_x(hL, huL, hvL, g)
        FR = fvm_swe.flux_x(hR, huR, hvR, g)
    elif axis == "y":
        wL, wR = vL, vR
        FL = fvm_swe.flux_y(hL, huL, hvL, g)
        FR = fvm_swe.flux_y(hR, huR, hvR, g)
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    sL = np.minimum(wL - cL, wR - cR)
    sR = np.maximum(wL + cL, wR + cR)

    UL = (hL, huL, hvL)
    UR = (hR, huR, hvR)

    denom = np.where(np.abs(sR - sL) > 1e-14, sR - sL, 1.0)
    out = []
    for fL, fR, uL_i, uR_i in zip(FL, FR, UL, UR):
        f_hll = (sR * fL - sL * fR + sL * sR * (uR_i - uL_i)) / denom
        f = np.where(sL >= 0.0, fL, np.where(sR <= 0.0, fR, f_hll))
        out.append(f)
    return out


def _wall_ghost(
    interior: np.ndarray,
    wall_value: np.ndarray | None,
) -> np.ndarray:
    """Ghost value from linear extrapolation through a wall.

    ``wall_value`` is the prescribed value on the wall (or None if free).
    """
    if wall_value is None:
        return interior.copy()
    return 2.0 * wall_value - interior


def _ghost_state(
    h_edge: np.ndarray,
    hu_edge: np.ndarray,
    hv_edge: np.ndarray,
    bc: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ghost (h, hu, hv) from interior edge values and BC on one wall."""
    dirichlet = bc.get("dirichlet", {})

    _, u_int, v_int = fvm_swe.conservative_to_primitive(h_edge, hu_edge, hv_edge)

    ghost_h = _wall_ghost(h_edge, dirichlet.get("h"))
    ghost_u = _wall_ghost(u_int, dirichlet.get("u"))
    ghost_v = _wall_ghost(v_int, dirichlet.get("v"))

    return ghost_h, ghost_h * ghost_u, ghost_h * ghost_v


def _extend_x(
    H: np.ndarray,
    HU: np.ndarray,
    HV: np.ndarray,
    problem: BaseProblem,
    grid: StructuredGrid2D,
    t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extend state arrays by one ghost cell on each side in x."""
    yc = grid.yc
    t_arr = np.full(grid.Ny, t)

    bc_min = problem.boundary_condition(
        "x_min", np.full(grid.Ny, grid.x_min), yc, t_arr
    )
    gh_min, ghu_min, ghv_min = _ghost_state(H[0, :], HU[0, :], HV[0, :], bc_min)

    bc_max = problem.boundary_condition(
        "x_max", np.full(grid.Ny, grid.x_max), yc, t_arr
    )
    gh_max, ghu_max, ghv_max = _ghost_state(H[-1, :], HU[-1, :], HV[-1, :], bc_max)

    H_ext = np.concatenate([gh_min[None, :], H, gh_max[None, :]], axis=0)
    HU_ext = np.concatenate([ghu_min[None, :], HU, ghu_max[None, :]], axis=0)
    HV_ext = np.concatenate([ghv_min[None, :], HV, ghv_max[None, :]], axis=0)
    return H_ext, HU_ext, HV_ext


def _extend_y(
    H: np.ndarray,
    HU: np.ndarray,
    HV: np.ndarray,
    problem: BaseProblem,
    grid: StructuredGrid2D,
    t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extend state arrays by one ghost cell on each side in y."""
    xc = grid.xc
    t_arr = np.full(grid.Nx, t)

    bc_min = problem.boundary_condition(
        "y_min", xc, np.full(grid.Nx, grid.y_min), t_arr
    )
    gh_min, ghu_min, ghv_min = _ghost_state(H[:, 0], HU[:, 0], HV[:, 0], bc_min)

    bc_max = problem.boundary_condition(
        "y_max", xc, np.full(grid.Nx, grid.y_max), t_arr
    )
    gh_max, ghu_max, ghv_max = _ghost_state(H[:, -1], HU[:, -1], HV[:, -1], bc_max)

    H_ext = np.concatenate([gh_min[:, None], H, gh_max[:, None]], axis=1)
    HU_ext = np.concatenate([ghu_min[:, None], HU, ghu_max[:, None]], axis=1)
    HV_ext = np.concatenate([ghv_min[:, None], HV, ghv_max[:, None]], axis=1)
    return H_ext, HU_ext, HV_ext


def _step(
    H: np.ndarray,
    HU: np.ndarray,
    HV: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    g: float,
    problem: BaseProblem,
    grid: StructuredGrid2D,
    t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One HLL update: U^{n+1} = U^n - dt/dx * dF - dt/dy * dG + dt * S."""
    Hx, HUx, HVx = _extend_x(H, HU, HV, problem, grid, t)
    Hy, HUy, HVy = _extend_y(H, HU, HV, problem, grid, t)

    Fh, Fhu, Fhv = _hll_flux(
        Hx[:-1, :],
        HUx[:-1, :],
        HVx[:-1, :],
        Hx[1:, :],
        HUx[1:, :],
        HVx[1:, :],
        g,
        axis="x",
    )
    Gh, Ghu, Ghv = _hll_flux(
        Hy[:, :-1],
        HUy[:, :-1],
        HVy[:, :-1],
        Hy[:, 1:],
        HUy[:, 1:],
        HVy[:, 1:],
        g,
        axis="y",
    )

    H_new = H - dt / dx * (Fh[1:, :] - Fh[:-1, :]) - dt / dy * (Gh[:, 1:] - Gh[:, :-1])
    HU_new = (
        HU - dt / dx * (Fhu[1:, :] - Fhu[:-1, :]) - dt / dy * (Ghu[:, 1:] - Ghu[:, :-1])
    )
    HV_new = (
        HV - dt / dx * (Fhv[1:, :] - Fhv[:-1, :]) - dt / dy * (Ghv[:, 1:] - Ghv[:, :-1])
    )

    _, U, V = fvm_swe.conservative_to_primitive(H, HU, HV)
    sources = problem.sources(H, U, V, problem.parameters)

    H_new = H_new + dt * sources["mass"]
    HU_new = HU_new + dt * H * sources["accel_x"]
    HV_new = HV_new + dt * H * sources["accel_y"]

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

    Yields:
        Dictionary with keys ``'t'``, ``'h'``, ``'u'``, ``'v'`` at each
        save time. Arrays have shape ``(Nx, Ny)``.

    Example:
        >>> for snap in solve(problem, grid, t_end=1.0, save_times=[0.5, 1.0]):
        ...     print(f"t={snap['t']:.2f}, max(h)={snap['h'].max():.3f}")
    """
    g = problem.parameters.get("g", 9.81)

    ic = problem.initial_condition(grid.X, grid.Y)
    H, HU, HV = fvm_swe.primitive_to_conservative(ic["h"], ic["u"], ic["v"])

    if save_times is None:
        save_times = [t_end]
    save_times = sorted(float(s) for s in save_times)

    next_save = 0
    t = 0.0
    dx, dy = grid.dx, grid.dy
    steps = 0

    while next_save < len(save_times) and save_times[next_save] <= 0.0:
        h_snap, u_snap, v_snap = fvm_swe.conservative_to_primitive(H, HU, HV)
        yield {"t": 0.0, "h": h_snap.copy(), "u": u_snap.copy(), "v": v_snap.copy()}
        next_save += 1

    while t < t_end - 1e-15:
        _, U, V = fvm_swe.conservative_to_primitive(H, HU, HV)
        max_sx, max_sy = fvm_swe.max_wave_speeds(H, U, V, g)
        dt = cfl / (max_sx / dx + max_sy / dy)

        target = save_times[next_save] if next_save < len(save_times) else t_end
        dt = min(dt, target - t, t_end - t)

        H, HU, HV = _step(H, HU, HV, dx, dy, dt, g, problem, grid, t)
        t += dt
        steps += 1

        while next_save < len(save_times) and t >= save_times[next_save] - 1e-12:
            h_snap, u_snap, v_snap = fvm_swe.conservative_to_primitive(H, HU, HV)
            yield {
                "t": t,
                "h": h_snap.copy(),
                "u": u_snap.copy(),
                "v": v_snap.copy(),
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
    """Materialize all snapshots into a list.

    Returns:
        ``(snapshots, times)`` where each snapshot is a dict ``{'h', 'u', 'v'}``
        of shape ``(Nx, Ny)`` and ``times`` is the parallel list of ``t``.
    """
    snapshots: list[dict[str, np.ndarray]] = []
    times: list[float] = []
    for snap in solve(problem, grid, t_end, cfl, save_times, verbose):
        times.append(snap["t"])
        snapshots.append({"h": snap["h"], "u": snap["u"], "v": snap["v"]})
    return snapshots, times
