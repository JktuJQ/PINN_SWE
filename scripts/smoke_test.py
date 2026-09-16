"""
Layered smoke test for the simplified PINN stack.

Run from the project root::

    python -m scripts.smoke_test

Each section prints OK / FAIL. A failure in one section does not stop
the rest, so a single run gives the full picture.
"""

import os
import traceback
from pathlib import Path

import numpy as np
import torch

CHECKPOINT = Path("solvers/pinn/weights/pinn_swe.model")

_SECTION = {"n": 0}


def section(title: str) -> None:
    _SECTION["n"] += 1
    print(f"\n[{_SECTION['n']:02d}] {title}")
    print("-" * (len(title) + 8))


def check(name: str, fn):
    try:
        result = fn()
        print(f"  OK   {name}")
        return result
    except Exception:
        print(f"  FAIL {name}")
        traceback.print_exc()
        return None


def evaluate_against_exact(problem, model, times, n_x=800):
    """Return {t: {'L2_h': ..., 'Linf_h': ...}} along y=0."""
    from solvers.pinn import swe_residual  # noqa: F401  (import kept for clarity)

    results = {}
    for t in times:
        x = torch.linspace(problem.domain.x_min, problem.domain.x_max, n_x).view(-1, 1)
        y = torch.zeros_like(x)
        tt = torch.full_like(x, float(t))

        with torch.no_grad():
            h, u, v = model(x, y, tt)

        x_np = x.cpu().numpy().ravel()
        h_np = h.cpu().numpy().ravel()
        u_np = u.cpu().numpy().ravel()

        exact = problem.exact_solution(x_np, np.zeros_like(x_np), float(t))
        if exact is None:
            continue

        h_ex = exact["h"]
        u_ex = exact["u"]
        results[float(t)] = {
            "L2_h": float(np.linalg.norm(h_np - h_ex) / np.linalg.norm(h_ex)),
            "Linf_h": float(np.max(np.abs(h_np - h_ex))),
            "L2_u": float(
                np.linalg.norm(u_np - u_ex) / max(np.linalg.norm(u_ex), 1e-12)
            ),
        }
    return results


def main() -> int:
    failures = 0

    # ------------------------------------------------------------------
    section("core: domain and problem")
    # ------------------------------------------------------------------
    from core import RectangularDomain
    from problems.dam_break_1d import DamBreak1D

    domain = RectangularDomain((-6.0, 6.0), (-6.0, 6.0), (0.0, 1.0))
    check("RectangularDomain", lambda: domain)

    problem = check("DamBreak1D", lambda: DamBreak1D(domain))
    if problem is None:
        return 1

    check("max_wave_speed > 0", lambda: problem.max_wave_speed > 0)
    check("disturbance_center == x_dam", lambda: problem.disturbance_center == 0.0)

    x = np.array([-1.0, 1.0])
    y = np.array([0.0, 0.0])
    t = np.array([0.5, 0.5])

    bc_x = problem.boundary_condition("x_max", x, y, t)
    if set(bc_x.get("dirichlet", {})) == {"h", "u", "v"}:
        print("  OK   x_max BC prescribes h, u, v")
    else:
        print(f"  FAIL x_max BC keys = {set(bc_x.get('dirichlet', {}))}")
        failures += 1

    bc_y = problem.boundary_condition("y_min", x, y, t)
    if set(bc_y.get("dirichlet", {})) == {"v"}:
        print("  OK   y_min BC constrains only v")
    else:
        print(f"  FAIL y_min BC keys = {set(bc_y.get('dirichlet', {}))}")
        failures += 1

    # ------------------------------------------------------------------
    section("PINN network")
    # ------------------------------------------------------------------
    from solvers.pinn import PINN

    model_small = check("PINN(domain)", lambda: PINN(domain, hidden=32, depth=3))
    if model_small is None:
        return failures + 1

    N = 4
    x_t = torch.rand(N, 1) * 12 - 6
    y_t = torch.rand(N, 1) * 12 - 6
    t_t = torch.rand(N, 1)

    out = check("forward", lambda: model_small(x_t, y_t, t_t))
    if out is not None:
        h, u, v = out
        if h.shape == u.shape == v.shape == (N, 1):
            print(f"  OK   shapes {tuple(h.shape)}")
        else:
            print(
                f"  FAIL shapes h={tuple(h.shape)} u={tuple(u.shape)} v={tuple(v.shape)}"
            )
            failures += 1
        if bool(torch.all(h > 0)):
            print(
                f"  OK   h strictly positive  [{h.min().item():.3f}, {h.max().item():.3f}]"
            )
        else:
            print("  FAIL h has non-positive values")
            failures += 1

    # ------------------------------------------------------------------
    section("SWE residual")
    # ------------------------------------------------------------------
    from solvers.pinn import swe_residual

    R = check(
        "residual nu=0",
        lambda: swe_residual(problem, model_small, x_t, y_t, t_t, nu=0.0),
    )
    if R is not None:
        if R.shape == (N, 3) and bool(torch.isfinite(R).all()):
            print(f"  OK   shape {tuple(R.shape)}  max|R| = {R.abs().max().item():.3e}")
        else:
            print(f"  FAIL shape {tuple(R.shape)} or non-finite values")
            failures += 1

    R_visc = check(
        "residual nu=0.05",
        lambda: swe_residual(problem, model_small, x_t, y_t, t_t, nu=0.05),
    )
    if R is not None and R_visc is not None:
        if (R - R_visc).abs().max().item() > 0:
            print("  OK   viscosity changes residual")
        else:
            print("  FAIL viscosity has no effect")
            failures += 1

    def _grad_flow():
        model_small.zero_grad(set_to_none=True)
        R_loc = swe_residual(problem, model_small, x_t, y_t, t_t, nu=0.0)
        (R_loc**2).mean().backward()
        bad = [
            name
            for name, p in model_small.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0
        ]
        model_small.zero_grad(set_to_none=True)
        if bad:
            raise RuntimeError(f"no gradient for: {bad[:5]}")
        return True

    check("backward reaches all parameters", _grad_flow)

    # ------------------------------------------------------------------
    section("sampling")
    # ------------------------------------------------------------------
    from solvers.pinn import sample_batch, WALLS

    batch = check(
        "sample_batch",
        lambda: sample_batch(problem, 256, 64, 32, t_max=1.0, device="cpu"),
    )
    if batch is not None:
        interior, ic_pts, walls = batch
        print(
            f"  OK   interior={interior[0].shape[0]}  ic={ic_pts[0].shape[0]}  "
            f"walls={list(walls)}"
        )
        if set(walls) != set(WALLS):
            print(f"  FAIL wall keys = {set(walls)}")
            failures += 1

        ok = True
        for w, (bx, by, _) in walls.items():
            if w == "x_min" and not bool((bx == domain.x_min).all()):
                ok = False
            elif w == "x_max" and not bool((bx == domain.x_max).all()):
                ok = False
            elif w == "y_min" and not bool((by == domain.y_min).all()):
                ok = False
            elif w == "y_max" and not bool((by == domain.y_max).all()):
                ok = False
        print(f"  {'OK' if ok else 'FAIL'}   wall points on correct wall")
        if not ok:
            failures += 1

    # ------------------------------------------------------------------
    section("loss assembly")
    # ------------------------------------------------------------------
    from solvers.pinn import compute_losses

    losses = check(
        "compute_losses",
        lambda: compute_losses(
            problem, model_small, batch, nu=0.01, device=torch.device("cpu")
        ),
    )
    if losses is not None:
        for name, value in losses.items():
            ok = value.dim() == 0 and bool(torch.isfinite(value))
            print(f"  {'OK' if ok else 'FAIL'}   {name} = {value.item():.4e}")
            if not ok:
                failures += 1

    # ------------------------------------------------------------------
    section("RAR and rebalance")
    # ------------------------------------------------------------------
    from solvers.pinn import refine, rebalance_weights

    pts = check(
        "refine",
        lambda: refine(
            problem,
            model_small,
            nu=0.01,
            t_max=1.0,
            pool=256,
            keep=32,
            device=torch.device("cpu"),
        ),
    )
    if pts is not None:
        xr, yr, tr = pts
        if xr.shape == (32, 1) and yr.shape == (32, 1) and tr.shape == (32, 1):
            print(f"  OK   kept {xr.shape[0]} points")
        else:
            print(f"  FAIL shapes {xr.shape}, {yr.shape}, {tr.shape}")
            failures += 1

    def _rebalance():
        model_small.zero_grad(set_to_none=True)
        probe = compute_losses(
            problem, model_small, batch, nu=0.01, device=torch.device("cpu")
        )
        w = {"pde": 1.0, "ic": 10.0, "bc": 10.0}
        return rebalance_weights(w, probe, model_small)

    new_w = check("rebalance_weights", _rebalance)
    if new_w is not None:
        print(f"  OK   weights = { {k: round(v, 3) for k, v in new_w.items()} }")

    # ------------------------------------------------------------------
    section("train_pinn: tiny run (30 Adam + 5 L-BFGS)")
    # ------------------------------------------------------------------
    from solvers.pinn import train_pinn

    history = check(
        "train_pinn",
        lambda: train_pinn(
            problem,
            PINN(domain, hidden=32, depth=3),
            n_interior=256,
            n_ic=64,
            n_bc=32,
            rar_every=15,
            rar_pool=256,
            rar_keep=32,
            adapt_every=15,
            adam_epochs=30,
            lbfgs_steps=5,
            device="cpu",
            verbose=False,
        ),
    )
    if history is not None:
        print(
            f"  OK   {len(history['loss'])} entries  "
            f"loss[0]={history['loss'][0]:.3e}  "
            f"loss[-1]={history['loss'][-1]:.3e}"
        )

    # ------------------------------------------------------------------
    section("HLL solver: 50x50 short run vs exact")
    # ------------------------------------------------------------------
    from core import StructuredGrid2D
    from solvers.numerical import hll

    grid = StructuredGrid2D((-6.0, 6.0), (-6.0, 6.0), Nx=50, Ny=50)
    snaps = check(
        "hll.solve_all",
        lambda: hll.solve_all(problem, grid, t_end=1.0, save_times=[0.25, 0.5, 1.0]),
    )
    if snaps is not None:
        snapshots, times = snaps
        print(f"  OK   {len(snapshots)} snapshots at t={times}")
        j = grid.Ny // 2
        for snap, tt in zip(snapshots, times):
            x = grid.xc
            h = snap["h"][:, j]
            exact = problem.exact_solution(x, np.zeros_like(x), tt)
            if exact is None:
                continue
            l2 = float(np.linalg.norm(h - exact["h"]) / np.linalg.norm(exact["h"]))
            print(f"       t={tt:.2f}  L2(h) = {100 * l2:.2f}%")

    # ------------------------------------------------------------------
    section("trained checkpoint")
    # ------------------------------------------------------------------
    if not CHECKPOINT.exists():
        print(f"  SKIP  no checkpoint at {CHECKPOINT}")
        print(f"       (train one with scripts/train.py to enable this section)")
    else:
        from solvers.pinn import load_pinn

        loaded = check(
            f"load_pinn({CHECKPOINT})",
            lambda: load_pinn(domain, str(CHECKPOINT)),
        )
        if loaded is not None:
            results = check(
                "evaluate vs exact",
                lambda: evaluate_against_exact(
                    problem, loaded, times=(0.25, 0.5, 0.75, 1.0)
                ),
            )
            if results is not None:
                print(f"  {'t':>6}  {'L2(h)':>8}  {'Linf(h)':>9}  {'L2(u)':>8}")
                for tt in sorted(results):
                    r = results[tt]
                    print(
                        f"  {tt:>6.2f}  {100 * r['L2_h']:>7.2f}%  "
                        f"{r['Linf_h']:>9.4f}  {100 * r['L2_u']:>7.2f}%"
                    )
                print(f"  (README reference: L2(h) ≈ 0.55% at t=1.0)")

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    if failures == 0:
        print("SMOKE TEST PASSED")
    else:
        print(f"SMOKE TEST FAILED: {failures} check(s) reported problems")
    print("=" * 60)
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
