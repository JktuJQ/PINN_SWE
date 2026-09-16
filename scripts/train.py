"""
Train a PINN for the dam-break problem.

The default configuration reproduces the legacy ``swe_constants.py``
setup exactly (128x6 network, 8000 Adam epochs + 400 L-BFGS iterations,
cone-focused sampling with RAR, vanishing viscosity, time-marching
curriculum). Passing no arguments is equivalent to a full legacy run.

Usage:
    python -m scripts.train                     # full run (~30-80 min)
    python -m scripts.train --quick             # 300-epoch sanity check
    python -m scripts.train --adam-epochs 4000  # custom length
    python -m scripts.train --device cpu
    python -m scripts.train --out my_model.pt
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

from core import RectangularDomain
from problems.dam_break_1d import DamBreak1D
from solvers.pinn import PINN, save_pinn, train_pinn
from evaluation.plotting.styles import apply_paper_style
from evaluation.plotting.training import plot_training_history


def evaluate_against_exact(
    problem: DamBreak1D,
    model: PINN,
    times: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    n_x: int = 800,
) -> dict[float, dict[str, float]]:
    """Return per-time L2/Linf errors of h and u along y = 0."""
    model.eval()
    device = next(model.parameters()).device
    results: dict[float, dict[str, float]] = {}

    for t in times:
        x = torch.linspace(
            problem.domain.x_min, problem.domain.x_max, n_x, device=device
        ).view(-1, 1)
        y = torch.zeros_like(x)
        tt = torch.full_like(x, float(t))
        with torch.no_grad():
            h, u, _ = model(x, y, tt)

        x_np = x.cpu().numpy().ravel()
        h_np = h.cpu().numpy().ravel()
        u_np = u.cpu().numpy().ravel()

        exact = problem.exact_solution(x_np, np.zeros_like(x_np), float(t))
        if exact is None:
            continue

        h_ex, u_ex = exact["h"], exact["u"]
        results[float(t)] = {
            "L2_h": float(np.linalg.norm(h_np - h_ex) / np.linalg.norm(h_ex)),
            "Linf_h": float(np.max(np.abs(h_np - h_ex))),
            "L2_u": float(
                np.linalg.norm(u_np - u_ex) / max(np.linalg.norm(u_ex), 1e-12)
            ),
        }
    return results


def print_metrics_table(results: dict[float, dict[str, float]]) -> None:
    print(f"\n{'t':>6}  {'L2(h)':>8}  {'Linf(h)':>9}  {'L2(u)':>8}")
    print("-" * 38)
    for t in sorted(results):
        r = results[t]
        print(
            f"{t:>6.2f}  {100 * r['L2_h']:>7.2f}%  "
            f"{r['Linf_h']:>9.4f}  {100 * r['L2_u']:>7.2f}%"
        )
    print("(legacy reference at t=1.0: L2(h) ≈ 0.53%)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train a PINN for the 2D dam-break problem."
    )

    ap.add_argument(
        "--quick",
        action="store_true",
        help="run a short 300 + 20 training (for testing the pipeline)",
    )

    ap.add_argument(
        "--out",
        default="solvers/pinn/weights/pinn_swe.model",
        help="checkpoint output path",
    )
    ap.add_argument(
        "--figdir",
        default="data/figures",
        help="directory for the training-history figure",
    )

    ap.add_argument("--x-max", type=float, default=6.0)
    ap.add_argument("--y-max", type=float, default=6.0)
    ap.add_argument("--t-max", type=float, default=1.0)

    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=6)

    ap.add_argument("--adam-epochs", type=int, default=8000)
    ap.add_argument("--lbfgs-steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-final", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--n-interior", type=int, default=8192)
    ap.add_argument("--n-ic", type=int, default=2048)
    ap.add_argument("--n-bc", type=int, default=1024)
    ap.add_argument("--rar-every", type=int, default=500)
    ap.add_argument("--rar-pool", type=int, default=40000)
    ap.add_argument("--rar-keep", type=int, default=2048)

    ap.add_argument("--nu-start", type=float, default=0.10)
    ap.add_argument("--nu-end", type=float, default=0.005)
    ap.add_argument("--nu-anneal-frac", type=float, default=0.6)
    ap.add_argument("--curriculum-stages", type=int, default=4)
    ap.add_argument("--curriculum-frac", type=float, default=0.4)

    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="if > 0, print L2(h) at t=1.0 every N epochs "
        "(useful to watch progress on long runs)",
    )

    args = ap.parse_args()

    if args.quick:
        args.adam_epochs = 300
        args.lbfgs_steps = 20
        args.n_interior = 1024
        args.n_ic = 256
        args.n_bc = 128
        args.rar_pool = 2048
        args.rar_keep = 256
        args.rar_every = 50
        print("quick mode: 300 Adam + 20 L-BFGS, small networks and batches")

    apply_paper_style()

    domain = RectangularDomain(
        (-args.x_max, args.x_max),
        (-args.y_max, args.y_max),
        (0.0, args.t_max),
    )
    problem = DamBreak1D(domain)
    model = PINN(domain, hidden=args.hidden, depth=args.depth)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print("PINN training")
    print("=" * 60)
    print(
        f"  domain       : x∈[{-args.x_max}, {args.x_max}], "
        f"y∈[{-args.y_max}, {args.y_max}], t∈[0, {args.t_max}]"
    )
    print(
        f"  network      : hidden={args.hidden}, depth={args.depth}, "
        f"params={n_params}"
    )
    print(f"  optimizer    : Adam({args.adam_epochs}) + L-BFGS({args.lbfgs_steps})")
    print(f"  lr           : {args.lr} -> {args.lr_final}")
    print(
        f"  sampling     : {args.n_interior} interior, "
        f"{args.n_ic} ic, {args.n_bc}/wall, "
        f"RAR every {args.rar_every}"
    )
    print(
        f"  viscosity    : {args.nu_start} -> {args.nu_end} "
        f"over {args.nu_anneal_frac:.0%} of Adam"
    )
    print(
        f"  curriculum   : {args.curriculum_stages} stages, "
        f"growth over {args.curriculum_frac:.0%} of Adam"
    )
    print(f"  seed         : {args.seed}")
    print(f"  device       : {args.device}")
    print(f"  output       : {args.out}")
    print()

    t0 = time.time()
    history = train_pinn(
        problem,
        model,
        n_interior=args.n_interior,
        n_ic=args.n_ic,
        n_bc=args.n_bc,
        rar_every=args.rar_every,
        rar_pool=args.rar_pool,
        rar_keep=args.rar_keep,
        nu_start=args.nu_start,
        nu_end=args.nu_end,
        nu_anneal_frac=args.nu_anneal_frac,
        curriculum_stages=args.curriculum_stages,
        curriculum_frac=args.curriculum_frac,
        adam_epochs=args.adam_epochs,
        lbfgs_steps=args.lbfgs_steps,
        lr=args.lr,
        lr_final=args.lr_final,
        seed=args.seed,
        device=args.device,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\ntraining took {elapsed / 60:.1f} min")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu")
    save_pinn(model_cpu, str(out_path))
    print(f"saved checkpoint to {out_path}")

    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    plot_training_history(
        history,
        save_path=str(figdir / "training_history.png"),
    )
    print(f"saved training history to {figdir / 'training_history.png'}")

    print("\nevaluating against exact solution...")
    results = evaluate_against_exact(problem, model_cpu)
    print_metrics_table(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
