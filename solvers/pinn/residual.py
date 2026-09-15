"""
PDE residual for the 2D shallow water equations, evaluated via autograd.

Conservative form, with primitive-variable sources:

    h_t    + (hu)_x               + (hv)_y               =  S_mass     + nu*lap(h)
    (hu)_t + (hu^2 + g h^2/2)_x   + (huv)_y              =  h*S_accel_x + nu*lap(hu)
    (hv)_t + (huv)_x              + (hv^2 + g h^2/2)_y   =  h*S_accel_y + nu*lap(hv)

``S_mass`` and ``S_accel_x/y`` are the primitive-variable sources
returned by ``problem.sources``. Note the factor of ``h`` on the
momentum sources: they come in as accelerations (du/dt), and must be
multiplied by ``h`` to enter the conservative equation.

Artificial viscosity ``nu`` implements the vanishing-viscosity method:
a smooth network cannot represent a discontinuity, but it can
approximate the smooth solution of the viscous problem whose front
thickness is O(nu). Annealing ``nu`` down produces sharper and sharper
fronts converging to the entropy solution.

Each residual component is divided by a characteristic scale so that the
three equations contribute comparably to the MSE. Without normalization
the momentum equations (O(g h^2) ~ 20) dominate the continuity equation
(O(1)).

Known limitation
----------------
``problem.sources`` currently takes numpy arrays, so we bridge through
numpy and lose gradient flow through the source term. This is fine for
dam-break-type problems where sources are identically zero, and is
acceptable when sources depend on given fields (topography) rather than
on the network output. If a future problem needs differentiable sources,
``BaseProblem.sources`` should be extended to accept torch tensors.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from core import BaseProblem


__all__ = ["SWEResidual", "compute_scales"]


def _partial(out: torch.Tensor, wrt: torch.Tensor) -> torch.Tensor:
    """``d(out)/d(wrt)`` with the graph retained for higher derivatives."""
    return torch.autograd.grad(
        out, wrt, grad_outputs=torch.ones_like(out), create_graph=True
    )[0]


def _laplacian(q: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``d^2 q/dx^2 + d^2 q/dy^2``."""
    q_xx = _partial(_partial(q, x), x)
    q_yy = _partial(_partial(q, y), y)
    return q_xx + q_yy


def compute_scales(problem: BaseProblem) -> dict[str, float]:
    """Characteristic scales for residual normalization.

    The three components of the SWE residual have different physical
    dimensions and magnitudes. Without normalization, the MSE is
    dominated by the momentum equations and the continuity equation
    barely contributes to the gradient.

    Uses ``h_l`` / ``h_left`` / ``g`` from ``problem.parameters`` when
    available, falling back to safe defaults. For non-dam-break problems
    the caller may want to pass explicit scales to ``SWEResidual``.
    """
    params = problem.parameters
    g = float(params.get("g", 9.81))
    h_ref = float(params.get("h_l", params.get("h_left", 1.0)))
    u_ref = math.sqrt(g * h_ref)

    domain = problem.domain
    t_ref = max(domain.t_max - domain.t_min, 1e-6)

    return {
        "mass": h_ref / t_ref,
        "momentum": h_ref * u_ref / t_ref,
    }


class SWEResidual:
    """Callable computing the 2D SWE residual at collocation points.

    Signature::

        residual(model, x, y, t, nu) -> Tensor of shape (N, 3)

    Columns are (continuity, x-momentum, y-momentum). ``x``, ``y``, ``t``
    are column tensors of shape ``(N, 1)``; the model maps them to
    ``(h, u, v)``. ``nu`` is the artificial viscosity (0 for the
    inviscid residual).
    """

    def __init__(
        self,
        problem: BaseProblem,
        scales: dict[str, float] | None = None,
    ) -> None:
        self.problem = problem
        self.scales = dict(scales) if scales is not None else compute_scales(problem)
        self._g = float(problem.parameters.get("g", 9.81))

    def _source_tensors(
        self, h: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate ``problem.sources`` and return tensors on ``h``'s device.

        Sources are given in primitive variables: ``mass`` is dh/dt,
        ``accel_x/y`` are du/dt, dv/dt.
        """
        h_np = h.detach().cpu().numpy()
        u_np = u.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        src = self.problem.sources(h_np, u_np, v_np, self.problem.parameters)

        def _to_tensor(arr: np.ndarray | float) -> torch.Tensor:
            if isinstance(arr, (int, float)):
                return torch.full_like(h, float(arr))
            return torch.as_tensor(arr, dtype=h.dtype, device=h.device)

        return (
            _to_tensor(src["mass"]),
            _to_tensor(src["accel_x"]),
            _to_tensor(src["accel_y"]),
        )

    def __call__(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        nu: float = 0.0,
    ) -> torch.Tensor:
        x = x.detach().clone().requires_grad_(True)
        y = y.detach().clone().requires_grad_(True)
        t = t.detach().clone().requires_grad_(True)

        h, u, v = model(x, y, t)
        g = self._g

        hu = h * u
        hv = h * v
        huv = hu * v

        h_t = _partial(h, t)
        hu_t = _partial(hu, t)
        hv_t = _partial(hv, t)

        F1 = hu
        F2 = hu * u + 0.5 * g * h * h
        F3 = huv

        G1 = hv
        G2 = huv
        G3 = hv * v + 0.5 * g * h * h

        R1 = h_t + _partial(F1, x) + _partial(G1, y)
        R2 = hu_t + _partial(F2, x) + _partial(G2, y)
        R3 = hv_t + _partial(F3, x) + _partial(G3, y)

        S_mass, S_accel_x, S_accel_y = self._source_tensors(h, u, v)
        R1 = R1 - S_mass
        R2 = R2 - h * S_accel_x
        R3 = R3 - h * S_accel_y

        if nu > 0.0:
            R1 = R1 - nu * _laplacian(h, x, y)
            R2 = R2 - nu * _laplacian(hu, x, y)
            R3 = R3 - nu * _laplacian(hv, x, y)

        s_mass = self.scales["mass"]
        s_mom = self.scales["momentum"]
        return torch.cat([R1 / s_mass, R2 / s_mom, R3 / s_mom], dim=1)
