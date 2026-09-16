"""
SWE residual for the PINN, computed via autograd.

Conservative form; artificial viscosity ``nu`` implements the
vanishing-viscosity method (smooth networks cannot represent shocks,
but they can represent the viscous solution with front thickness O(nu)).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from core import BaseProblem


def _d(out: torch.Tensor, wrt: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        out, wrt, grad_outputs=torch.ones_like(out), create_graph=True
    )[0]


def _lap(q: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _d(_d(q, x), x) + _d(_d(q, y), y)


def _scales(problem: BaseProblem) -> tuple[float, float]:
    """Characteristic (mass, momentum) scales for residual normalization."""
    p = problem.parameters
    g = float(p.get("g", 9.81))
    h = float(p.get("h_l", p.get("h_left", 1.0)))
    u = math.sqrt(g * h)
    t = max(problem.domain.t_max - problem.domain.t_min, 1e-6)
    return h / t, h * u / t


def swe_residual(
    problem: BaseProblem,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    nu: float = 0.0,
) -> torch.Tensor:
    """Return (N, 3) residual: continuity, x-momentum, y-momentum.

    Sources from ``problem.sources`` are added in primitive variables
    (mass rate for h, accelerations for u and v); accelerations are
    multiplied by h to enter the conservative equations.
    """
    g = float(problem.parameters.get("g", 9.81))
    s_mass, s_mom = _scales(problem)

    x = x.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)

    h, u, v = model(x, y, t)
    hu, hv = h * u, h * v

    h_t = _d(h, t)
    hu_t = _d(hu, t)
    hv_t = _d(hv, t)

    F1, F2, F3 = hu, hu * u + 0.5 * g * h * h, hu * v
    G1, G2, G3 = hv, hu * v, hv * v + 0.5 * g * h * h

    R1 = h_t + _d(F1, x) + _d(G1, y)
    R2 = hu_t + _d(F2, x) + _d(G2, y)
    R3 = hv_t + _d(F3, x) + _d(G3, y)

    src = problem.sources(
        h.detach().cpu().numpy(),
        u.detach().cpu().numpy(),
        v.detach().cpu().numpy(),
        problem.parameters,
    )

    def _t(arr):
        if isinstance(arr, (int, float)):
            return torch.full_like(h, float(arr))
        return torch.as_tensor(arr, dtype=h.dtype, device=h.device)

    R1 = R1 - _t(src["mass"])
    R2 = R2 - h * _t(src["accel_x"])
    R3 = R3 - h * _t(src["accel_y"])

    if nu > 0.0:
        R1 = R1 - nu * _lap(h, x, y)
        R2 = R2 - nu * _lap(hu, x, y)
        R3 = R3 - nu * _lap(hv, x, y)

    return torch.cat([R1 / s_mass, R2 / s_mom, R3 / s_mom], dim=1)
