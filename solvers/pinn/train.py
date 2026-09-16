"""
Training loop for the PINN.

All in one file: sampling, loss assembly, Adam + L-BFGS.

Parameters are plain keyword arguments with sensible defaults. To
configure a run, just pass the ones you want to change::

    history = train_pinn(problem, model, adam_epochs=2000)

The loop follows three ideas that made the difference between a
plateau at ~8% error and the 0.5% reported in the README:

* vanishing viscosity: nu is annealed from a large value down to a
  small one, sharpening the shock front;
* time-marching curriculum: t_max grows during training, so the
  network learns early time before being asked about late time;
* cone-focused sampling with RAR: most collocation points are placed
  where the dynamics actually happen, and are periodically refreshed
  based on the current residual.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from core import BaseProblem, RectangularDomain
from .residual import swe_residual


WALLS = ("x_min", "x_max", "y_min", "y_max")


def _u(n: int, lo: float, hi: float, device) -> torch.Tensor:
    return torch.rand(n, 1, device=device) * (hi - lo) + lo


def _wall_points(wall: str, n: int, t_max: float, domain: RectangularDomain, device):
    t = _u(n, 0.0, t_max, device)
    if wall == "x_min":
        x = torch.full((n, 1), domain.x_min, device=device)
        y = _u(n, domain.y_min, domain.y_max, device)
    elif wall == "x_max":
        x = torch.full((n, 1), domain.x_max, device=device)
        y = _u(n, domain.y_min, domain.y_max, device)
    elif wall == "y_min":
        x = _u(n, domain.x_min, domain.x_max, device)
        y = torch.full((n, 1), domain.y_min, device=device)
    elif wall == "y_max":
        x = _u(n, domain.x_min, domain.x_max, device)
        y = torch.full((n, 1), domain.y_max, device=device)
    else:
        raise ValueError(wall)
    return x, y, t


def sample_batch(
    problem: BaseProblem,
    n_interior: int,
    n_ic: int,
    n_bc: int,
    t_max: float,
    cone_frac: float = 0.5,
    cone_margin: float = 0.5,
    device: torch.device | str = "cpu",
):
    """Return (interior, ic, walls) tuples of column tensors."""
    d = problem.domain
    device = torch.device(device)

    n_cone = int(n_interior * cone_frac)
    n_uni = n_interior - n_cone

    x_u = _u(n_uni, d.x_min, d.x_max, device)
    y_u = _u(n_uni, d.y_min, d.y_max, device)
    t_u = _u(n_uni, 0.0, t_max, device)

    speed = problem.max_wave_speed
    center = problem.disturbance_center
    t_c = _u(n_cone, 0.0, t_max, device)
    half = speed * t_c + cone_margin
    x_c = (center - half + torch.rand_like(t_c) * (2.0 * half)).clamp(d.x_min, d.x_max)
    y_c = _u(n_cone, d.y_min, d.y_max, device)

    interior = (
        torch.cat([x_u, x_c]),
        torch.cat([y_u, y_c]),
        torch.cat([t_u, t_c]),
    )

    ic = (
        _u(n_ic, d.x_min, d.x_max, device),
        _u(n_ic, d.y_min, d.y_max, device),
        torch.zeros(n_ic, 1, device=device),
    )

    walls = {w: _wall_points(w, n_bc, t_max, d, device) for w in WALLS}

    return interior, ic, walls


def _mse(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return ((p - t) ** 2).mean()


def compute_losses(
    problem: BaseProblem,
    model: nn.Module,
    batch,
    nu: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Return per-term losses: pde, ic, bc."""
    interior, ic_pts, walls = batch

    R = swe_residual(problem, model, *interior, nu=nu)
    loss_pde = (R**2).mean()

    x_ic, y_ic, t_ic = ic_pts
    ic_target = problem.initial_condition(
        x_ic.detach().cpu().numpy(), y_ic.detach().cpu().numpy()
    )
    h_p, u_p, v_p = model(x_ic, y_ic, t_ic)
    preds = {"h": h_p, "u": u_p, "v": v_p}
    loss_ic = torch.zeros((), device=device)
    n = 0
    for name, pred in preds.items():
        if name in ic_target:
            target = torch.as_tensor(
                ic_target[name], dtype=pred.dtype, device=pred.device
            )
            loss_ic = loss_ic + _mse(pred, target)
            n += 1
    loss_ic = loss_ic / max(n, 1)

    loss_bc = torch.zeros((), device=device)
    n_bc_terms = 0
    for wall, (x_b, y_b, t_b) in walls.items():
        bc = problem.boundary_condition(
            wall,
            x_b.detach().cpu().numpy(),
            y_b.detach().cpu().numpy(),
            t_b.detach().cpu().numpy(),
        )
        dirichlet = bc.get("dirichlet", {})
        if not dirichlet:
            continue
        h_b, u_b, v_b = model(x_b, y_b, t_b)
        bcp = {"h": h_b, "u": u_b, "v": v_b}
        for name, target_np in dirichlet.items():
            if name not in bcp:
                continue
            target = torch.as_tensor(
                target_np, dtype=bcp[name].dtype, device=bcp[name].device
            )
            loss_bc = loss_bc + _mse(bcp[name], target)
            n_bc_terms += 1
    loss_bc = loss_bc / max(n_bc_terms, 1)

    return {"pde": loss_pde, "ic": loss_ic, "bc": loss_bc}


def _grad_norm(loss: torch.Tensor, model: nn.Module) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = 0.0
    for gi in grads:
        if gi is not None:
            total += float((gi**2).sum().item())
    return total**0.5


def rebalance_weights(
    weights: dict[str, float],
    losses: dict[str, torch.Tensor],
    model: nn.Module,
    alpha: float = 0.9,
    w_min: float = 1e-2,
    w_max: float = 1e4,
    reference: str = "pde",
) -> dict[str, float]:
    """Adjust weights so all terms contribute comparably to the gradient."""
    if reference not in losses:
        return dict(weights)
    ref_norm = _grad_norm(losses[reference], model)
    if ref_norm < 1e-20:
        return dict(weights)

    out = dict(weights)
    for name, loss in losses.items():
        if name == reference:
            continue
        g = _grad_norm(loss, model)
        if g < 1e-20:
            continue
        target = ref_norm / g
        old = out.get(name, 1.0)
        out[name] = float(min(max(alpha * old + (1 - alpha) * target, w_min), w_max))
    return out


def refine(
    problem: BaseProblem,
    model: nn.Module,
    nu: float,
    t_max: float,
    pool: int,
    keep: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a pool, keep the ``keep`` points with the largest residual."""
    d = problem.domain
    x = _u(pool, d.x_min, d.x_max, device)
    y = _u(pool, d.y_min, d.y_max, device)
    t = _u(pool, 0.0, t_max, device)
    R = swe_residual(problem, model, x, y, t, nu=nu)
    score = (R**2).mean(dim=1)
    idx = torch.topk(score, min(keep, pool)).indices
    return x[idx].detach(), y[idx].detach(), t[idx].detach()


def train_pinn(
    problem: BaseProblem,
    model: nn.Module,
    *,
    n_interior: int = 8192,
    n_ic: int = 2048,
    n_bc: int = 1024,
    cone_frac: float = 0.5,
    rar_every: int = 500,
    rar_pool: int = 40000,
    rar_keep: int = 2048,
    w_pde: float = 1.0,
    w_ic: float = 10.0,
    w_bc: float = 10.0,
    adaptive_weights: bool = True,
    adapt_every: int = 250,
    nu_start: float = 0.10,
    nu_end: float = 0.005,
    nu_anneal_frac: float = 0.6,
    curriculum_stages: int = 4,
    curriculum_frac: float = 0.4,
    adam_epochs: int = 8000,
    lbfgs_steps: int = 400,
    lr: float = 2e-3,
    lr_final: float = 1e-5,
    seed: int = 0,
    device: torch.device | str = "auto",
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Train a PINN and return a dict of per-epoch histories.

    Returns a dict with keys ``loss``, ``pde``, ``ic``, ``bc``, ``nu``,
    ``t_max``. Adam contributes ``adam_epochs`` entries; L-BFGS adds one
    more if ``lbfgs_steps > 0``.
    """
    if isinstance(device, str):
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)
    model.to(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    history: dict[str, list[float]] = {
        "loss": [],
        "pde": [],
        "ic": [],
        "bc": [],
        "nu": [],
        "t_max": [],
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gamma = (lr_final / lr) ** (1.0 / max(adam_epochs, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    weights = {"pde": w_pde, "ic": w_ic, "bc": w_bc}
    rar_points: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    batch = None
    nu = 0.0
    last: dict[str, torch.Tensor] = {}

    def loss_fn() -> torch.Tensor:
        """Recompute all loss terms from the current ``batch`` and ``nu``.

        Defined once, reused by Adam and L-BFGS. Reads ``batch``,
        ``nu``, ``weights`` and ``last`` from the enclosing scope, so
        reassigning any of them before a call takes effect immediately.
        """
        assert batch is not None
        losses = compute_losses(problem, model, batch, nu, device)
        last.clear()
        last.update(losses)
        total_loss = torch.zeros((), device=device)
        for name, value in losses.items():
            total_loss = total_loss + weights[name] * value
        return total_loss

    total = max(adam_epochs, 1)
    for epoch in range(total):
        frac = epoch / max(total - 1, 1)

        a = min(frac / nu_anneal_frac, 1.0)
        nu = nu_start * (nu_end / nu_start) ** a

        if frac < curriculum_frac:
            stage = int(frac / curriculum_frac * curriculum_stages)
            t_max = (
                problem.domain.t_min
                + (problem.domain.t_max - problem.domain.t_min)
                * (stage + 1)
                / curriculum_stages
            )
        else:
            t_max = problem.domain.t_max

        if rar_every > 0 and epoch > 0 and epoch % rar_every == 0:
            rar_points = refine(problem, model, nu, t_max, rar_pool, rar_keep, device)

        interior, ic_pts, walls = sample_batch(
            problem,
            n_interior,
            n_ic,
            n_bc,
            t_max,
            cone_frac=cone_frac,
            device=device,
        )
        if rar_points is not None:
            interior = (
                torch.cat([interior[0], rar_points[0]]),
                torch.cat([interior[1], rar_points[1]]),
                torch.cat([interior[2], rar_points[2]]),
            )
        batch = (interior, ic_pts, walls)

        if adaptive_weights and epoch > 0 and epoch % adapt_every == 0:
            with torch.enable_grad():
                probe = compute_losses(problem, model, batch, nu, device)
            weights = rebalance_weights(weights, probe, model)
            del probe

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history["loss"].append(float(loss.detach().item()))
        for k in ("pde", "ic", "bc"):
            history[k].append(float(last[k].detach().item()))
        history["nu"].append(nu)
        history["t_max"].append(t_max)

        if verbose and epoch % 200 == 0:
            print(
                f"  epoch {epoch:5d}  L={history['loss'][-1]:.2e}  "
                f"PDE={history['pde'][-1]:.2e}  "
                f"IC={history['ic'][-1]:.2e}  "
                f"BC={history['bc'][-1]:.2e}  "
                f"nu={nu:.4f}  t_max={t_max:.2f}"
            )

    if lbfgs_steps > 0:
        if verbose:
            print("  L-BFGS polish...")
        t_max = problem.domain.t_max
        nu = nu_end
        interior, ic_pts, walls = sample_batch(
            problem,
            n_interior * 2,
            n_ic * 2,
            n_bc * 2,
            t_max,
            cone_frac=cone_frac,
            device=device,
        )
        batch = (interior, ic_pts, walls)

        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_steps,
            history_size=50,
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            return loss

        lbfgs.step(closure)
        _ = loss_fn()

        history["loss"].append(float(last["pde"].detach().item()))
        for k in ("pde", "ic", "bc"):
            history[k].append(float(last[k].detach().item()))
        history["nu"].append(nu)
        history["t_max"].append(t_max)

    return history
