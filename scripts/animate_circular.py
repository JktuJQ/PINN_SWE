"""
3D animation of the HLL solution for the circular dam break.

Renders a 3D surface animation and prints diagnostics that reveal what
happens at the center of the domain:

* h at the grid center as a function of time;
* global min/max of h over all frames;
* radial profile h(r) at several times;
* mass conservation (integral of h over the domain);
* rotational symmetry error (comparing h(x,y) with h(y,x)).

Usage:
    python -m scripts.animate_circular
    python -m scripts.animate_circular --nx 150 --n-frames 60 --fps 15
    python -m scripts.animate_circular --t-end 0.4 --out data/anim.mp4
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import StructuredGrid2D
from problems import make_problem, default_domain
from solvers.numerical import hll
from evaluation.animation.evolution import animate_3d_fvm
from evaluation.plotting.styles import apply_paper_style, get_colors


def center_value(grid: StructuredGrid2D, h: np.ndarray) -> float:
    """h at the grid cell nearest to the origin."""
    i = int(np.argmin(np.abs(grid.xc)))
    j = int(np.argmin(np.abs(grid.yc)))
    return float(h[i, j])


def mass(grid: StructuredGrid2D, h: np.ndarray) -> float:
    """Integral of h over the domain."""
    return float(np.sum(h) * grid.dx * grid.dy)


def rotational_symmetry_error(
    grid: StructuredGrid2D, h: np.ndarray
) -> float:
    """max|h(x,y) - h(y,x)| on the square domain.

    For a radially symmetric solution the field must be invariant under
    the swap x <-> y, so this error is a direct measure of how well the
    solver preserves symmetry.
    """
    if h.shape[0] != h.shape[1]:
        return float("nan")
    return float(np.max(np.abs(h - h.T)))


def radial_profile(
    grid: StructuredGrid2D, h: np.ndarray, n_bins: int = 80
) -> tuple[np.ndarray, np.ndarray]:
    """Angle-averaged h(r) with linear binning.

    Returns (r_centers, h_mean) where h_mean is the mean of h over all
    cells whose distance from the origin falls into the same bin.
    """
    X, Y = grid.X, grid.Y
    r = np.sqrt(X * X + Y * Y).ravel()
    h_flat = h.ravel()

    r_max = float(r.max())
    edges = np.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    idx = np.clip(np.digitize(r, edges) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=h_flat, minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)

    h_mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return centers, h_mean


def print_diagnostics(
    problem,
    grid: StructuredGrid2D,
    snapshots: list[dict[str, np.ndarray]],
    times: list[float],
) -> None:
    """Print a compact diagnostic table."""
    h0 = snapshots[0]["h"]
    m0 = mass(grid, h0)

    print()
    print("Diagnostics")
    print("-" * 78)
    print(f"{'t':>6}  {'h_center':>9}  {'h_min':>8}  {'h_max':>8}  "
          f"{'mass_err %':>11}  {'sym_err':>9}")
    print("-" * 78)

    for snap, t in zip(snapshots, times):
        h = snap["h"]
        hc = center_value(grid, h)
        hmin = float(h.min())
        hmax = float(h.max())
        mass_err = 100.0 * abs(mass(grid, h) - m0) / max(m0, 1e-14)
        sym = rotational_symmetry_error(grid, h)
        print(f"{t:>6.3f}  {hc:>9.4f}  {hmin:>8.4f}  {hmax:>8.4f}  "
              f"{mass_err:>11.3f}  {sym:>9.4f}")

    print("-" * 78)
    print(f"initial mass M0 = {m0:.4f}")
    print(f"expected h_out = {problem.h_out:.4f}, h_in = {problem.h_in:.4f}")


def plot_radial_profiles(
    problem,
    grid: StructuredGrid2D,
    snapshots: list[dict[str, np.ndarray]],
    times: list[float],
    save_path: Path,
    n_show: int = 6,
) -> None:
    """Plot h(r) at a few selected times."""
    apply_paper_style()
    colors = get_colors()

    idx_all = np.linspace(0, len(snapshots) - 1, n_show).astype(int)

    fig, ax = plt.subplots(figsize=(9, 6))

    for k, i in enumerate(idx_all):
        r, h_mean = radial_profile(grid, snapshots[i]["h"])
        ax.plot(r, h_mean, lw=1.6, label=f"t = {times[i]:.2f}")

    ax.axhline(problem.h_out, color="k", lw=0.8, ls=":", alpha=0.6)
    ax.axhline(problem.h_in, color="k", lw=0.8, ls=":", alpha=0.6)
    ax.axvline(problem.r_dam, color="k", lw=0.8, ls="--", alpha=0.6,
               label=f"r_dam = {problem.r_dam}")

    ax.set_xlabel("r")
    ax.set_ylabel("h(r), angle-averaged")
    ax.set_title("Circular dam break: radial profiles")
    ax.set_xlim(0, min(grid.x_max, 8.0))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {save_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=100)
    ap.add_argument("--t-end", type=float, default=0.5)
    ap.add_argument("--n-frames", type=int, default=50)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default="data/figures/circular_3d.mp4")
    ap.add_argument("--elev", type=float, default=30.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--no-animation", action="store_true",
                    help="only compute diagnostics and radial plot")
    args = ap.parse_args()

    domain = default_domain("circular")
    problem = make_problem("circular", domain)
    grid = StructuredGrid2D(
        (domain.x_min, domain.x_max),
        (domain.y_min, domain.y_max),
        args.nx, args.nx,
    )

    reach = problem.r_dam + problem.max_wave_speed * args.t_end
    print(f"wave reach at t={args.t_end}: r = {reach:.2f}  "
          f"(domain half-width {domain.x_max:.2f})")
    if reach > domain.x_max - 0.5:
        print("WARNING: the wave reaches the boundary; far-field BC "
              "is no longer exact")

    save_times = [
        args.t_end * i / max(args.n_frames - 1, 1) for i in range(args.n_frames)
    ]
    print(f"running HLL on {args.nx}x{args.nx}, {args.n_frames} frames...")
    snapshots, times = hll.solve_all(
        problem, grid, t_end=args.t_end, save_times=save_times,
    )

    print_diagnostics(problem, grid, snapshots, times)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_radial_profiles(
        problem, grid, snapshots, times,
        save_path=out_dir / "circular_radial.png",
    )

    if not args.no_animation:
        print(f"saving 3D animation to {args.out}...")
        animate_3d_fvm(
            snapshots,
            domain,
            times,
            var="h",
            elevation=args.elev,
            azimuth=args.azim,
            fps=args.fps,
            save_path=str(args.out),
        )

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
