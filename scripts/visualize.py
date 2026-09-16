"""
Visualization script: PINN vs HLL vs exact for the dam-break problem.

Produces four figures in the output directory:

    comparison.png   h(x) and u(x) at selected times (PINN / HLL / exact)
    field_2d.png     PINN h(x,y) heatmap and deviation from y-invariance
    evolution.png    PINN h(x) profiles at several times
    errors.png       L2(h) error vs time for PINN and HLL

Usage:
    python -m scripts.visualize
    python -m scripts.visualize --checkpoint PATH --outdir DIR
    python -m scripts.visualize --times 0.0 0.25 0.5 0.75 1.0
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from core import RectangularDomain, StructuredGrid2D
from problems.dam_break_1d import DamBreak1D
from solvers.pinn import load_pinn
from solvers.numerical import hll
from evaluation.plotting.styles import apply_paper_style, get_colors
from evaluation.plotting.fields_2d import plot_heatmap, plot_y_deviation
from evaluation.extraction.profiles import extract_1d_from_fvm


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_pinn_1d(
    model,
    domain: RectangularDomain,
    t: float,
    y_val: float = 0.0,
    n: int = 800,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, h, u) along y = y_val at time t."""
    device = next(model.parameters()).device
    x = torch.linspace(domain.x_min, domain.x_max, n, device=device).view(-1, 1)
    y = torch.full_like(x, y_val)
    tt = torch.full_like(x, float(t))
    with torch.no_grad():
        h, u, _ = model(x, y, tt)
    return (
        x.cpu().numpy().ravel(),
        h.cpu().numpy().ravel(),
        u.cpu().numpy().ravel(),
    )


def evaluate_pinn_2d(
    model,
    domain: RectangularDomain,
    t: float,
    n: int = 200,
) -> dict[str, np.ndarray]:
    """Return {'h', 'u', 'v'} on an n x n lattice over the domain."""
    device = next(model.parameters()).device
    xs = torch.linspace(domain.x_min, domain.x_max, n, device=device)
    ys = torch.linspace(domain.y_min, domain.y_max, n, device=device)
    XX, YY = torch.meshgrid(xs, ys, indexing="ij")
    flat_x = XX.reshape(-1, 1)
    flat_y = YY.reshape(-1, 1)
    flat_t = torch.full_like(flat_x, float(t))
    with torch.no_grad():
        h, u, v = model(flat_x, flat_y, flat_t)
    return {
        "h": h.reshape(n, n).cpu().numpy(),
        "u": u.reshape(n, n).cpu().numpy(),
        "v": v.reshape(n, n).cpu().numpy(),
    }


def exact_1d(
    problem: DamBreak1D,
    t: float,
    n: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, h, u) of the exact solution along y = 0 at time t."""
    x = np.linspace(problem.domain.x_min, problem.domain.x_max, n)
    exact = problem.exact_solution(x, np.zeros_like(x), t)
    if exact is None:
        raise RuntimeError("problem has no exact solution")
    return x, exact["h"], exact["u"]


def l2_error(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.linalg.norm(ref))
    if denom < 1e-14:
        return 0.0
    return float(np.linalg.norm(pred - ref) / denom)


def figure_comparison(
    problem,
    grid,
    times,
    snapshots,
    model,
    colors,
    save_path: Path,
) -> None:
    """Two rows (h, u) by several time columns: PINN, HLL, exact."""
    n = len(times)
    fig, axes = plt.subplots(2, n, figsize=(3.8 * n, 7.5), squeeze=False)

    for k, t in enumerate(times):
        xe, he, ue = exact_1d(problem, t)

        xh, hh, uh, _ = extract_1d_from_fvm(snapshots[k], grid, y_val=0.0)

        xp = hp = up = None
        if model is not None:
            xp, hp, up = evaluate_pinn_1d(model, problem.domain, t)

        ax = axes[0][k]
        ax.plot(xe, he, "k-", lw=2.2, label="exact")
        ax.plot(xh, hh, "--", color=colors["fvm"], lw=1.6, label="HLL")
        if xp is not None:
            ax.plot(xp, hp, "-", color=colors["pinn"], lw=1.6, label="PINN")
        ax.set_title(f"h(x, y=0),  t = {t:.2f}")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel("h")
            ax.legend(fontsize=9)

        ax = axes[1][k]
        ax.plot(xe, ue, "k-", lw=2.2, label="exact")
        ax.plot(xh, uh, "--", color=colors["fvm"], lw=1.6, label="HLL")
        if xp is not None:
            ax.plot(xp, up, "-", color=colors["pinn"], lw=1.6, label="PINN")
        ax.set_title(f"u(x, y=0),  t = {t:.2f}")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel("u")
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {save_path}")


def figure_field_2d(
    problem,
    model,
    t,
    save_path: Path,
) -> None:
    """PINN h(x,y) heatmap and deviation from y-invariance."""
    field = evaluate_pinn_2d(model, problem.domain, t=t)

    plot_heatmap(
        field,
        problem.domain,
        var="h",
        t=t,
        save_path=str(save_path),
    )
    print(f"  wrote {save_path}")

    dev_path = save_path.with_name(save_path.stem + "_ydev" + save_path.suffix)
    plot_y_deviation(
        field,
        problem.domain,
        var="h",
        save_path=str(dev_path),
    )
    print(f"  wrote {dev_path}")


def figure_evolution(
    problem,
    model,
    times,
    colors,
    save_path: Path,
) -> None:
    """PINN h(x) profiles at several times, with the exact final curve."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, t in enumerate(times):
        xp, hp, _ = evaluate_pinn_1d(model, problem.domain, t)
        ax.plot(xp, hp, color=cycle[i % len(cycle)], lw=1.6, label=f"t = {t:.2f}")

    xe, he, _ = exact_1d(problem, times[-1])
    ax.plot(xe, he, "k--", lw=2.0, label=f"exact, t = {times[-1]:.2f}")

    ax.set_xlabel("x")
    ax.set_ylabel("h (depth)")
    ax.set_title("PINN: h(x, y=0) at multiple times")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {save_path}")


def figure_errors(
    problem,
    grid,
    times,
    snapshots,
    model,
    colors,
    save_path: Path,
) -> None:
    """L2(h) error vs time for PINN and HLL."""
    t_grid = np.linspace(0.05, max(times), 20)

    hll_snaps, hll_times = hll.solve_all(
        problem, grid, t_end=max(times), save_times=list(t_grid)
    )
    hll_l2 = []
    for snap, tt in zip(hll_snaps, hll_times):
        xh, hh, _, _ = extract_1d_from_fvm(snap, grid, y_val=0.0)
        xe, he, _ = exact_1d(problem, float(tt), n=len(xh))
        he_interp = np.interp(xh, xe, he)
        hll_l2.append(l2_error(hh, he_interp))

    pinn_l2 = []
    if model is not None:
        for tt in t_grid:
            xp, hp, _ = evaluate_pinn_1d(model, problem.domain, float(tt))
            _, he, _ = exact_1d(problem, float(tt), n=len(xp))
            pinn_l2.append(l2_error(hp, he))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_grid, [100 * v for v in hll_l2], "s--", color=colors["fvm"], label="HLL")
    if model is not None:
        ax.plot(
            t_grid, [100 * v for v in pinn_l2], "o-", color=colors["pinn"], label="PINN"
        )
    ax.set_xlabel("t")
    ax.set_ylabel("отн. ошибка $L_2(h)$, %")
    ax.set_title("Относительная ошибка $L_2$ по глубине")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {save_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="solvers/pinn/weights/pinn_swe.model",
        help="PINN checkpoint path (legacy or new format)",
    )
    ap.add_argument(
        "--outdir",
        default="data/figures",
        help="directory for the produced figures",
    )
    ap.add_argument(
        "--hll-nx",
        type=int,
        default=200,
        help="HLL grid resolution (Nx = Ny)",
    )
    ap.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0],
        help="times for the comparison figure",
    )
    ap.add_argument(
        "--no-pinn",
        action="store_true",
        help="skip PINN curves (HLL vs exact only)",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    apply_paper_style()
    colors = get_colors()

    domain = RectangularDomain((-6.0, 6.0), (-6.0, 6.0), (0.0, 1.0))
    problem = DamBreak1D(domain)
    grid = StructuredGrid2D(
        (domain.x_min, domain.x_max),
        (domain.y_min, domain.y_max),
        Nx=args.hll_nx,
        Ny=args.hll_nx,
    )

    model = None
    if not args.no_pinn:
        ckpt = Path(args.checkpoint)
        if ckpt.exists():
            device = _pick_device()
            model = load_pinn(domain, str(ckpt)).to(device)
            model.eval()
            print(f"loaded PINN from {ckpt} on {device}")
        else:
            print(f"no checkpoint at {ckpt}, producing HLL-vs-exact only")

    print(f"running HLL on {args.hll_nx}x{args.hll_nx} grid...")
    snapshots, saved_times = hll.solve_all(
        problem,
        grid,
        t_end=max(args.times),
        save_times=args.times,
    )
    times = [float(t) for t in saved_times]

    figure_comparison(
        problem,
        grid,
        times,
        snapshots,
        model,
        colors,
        outdir / "comparison.png",
    )

    if model is not None:
        figure_field_2d(
            problem,
            model,
            t=max(times),
            save_path=outdir / "field_2d.png",
        )
        figure_evolution(
            problem,
            model,
            times=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            colors=colors,
            save_path=outdir / "evolution.png",
        )

    figure_errors(
        problem,
        grid,
        times,
        snapshots,
        model,
        colors,
        outdir / "errors.png",
    )

    print(f"\nall figures saved to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
