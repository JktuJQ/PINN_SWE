"""
Standard loss terms: PDE residual, initial conditions, boundary
conditions.

The boundary-condition term uses the contract from
``BaseProblem.boundary_condition``: it returns a nested dictionary
``{condition_type: {variable: value}}``. Variables that are *absent*
from that dictionary are left free and are not penalized. This is what
allows a slip wall (``v = 0``, ``h`` and ``u`` free) to be expressed
without any geometry-specific branching here.
"""

import torch
import torch.nn as nn

from .base import LossContext, LossTerm
from ..sampling.base import Batch


def _mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((predicted - target) ** 2).mean()


class PDEResidualLoss(LossTerm):
    """Mean squared PDE residual over the interior collocation points."""

    name = "pde"

    def compute(
        self,
        model: nn.Module,
        batch: Batch,
        ctx: LossContext,
    ) -> torch.Tensor:
        x, y, t = batch.interior
        R = ctx.residual_fn(model, x, y, t, nu=ctx.nu)
        return (R**2).mean()


class InitialConditionLoss(LossTerm):
    """MSE between the network and the exact initial state.

    The target is obtained from ``problem.initial_condition``. Only
    variables present in both the model output and the IC dictionary
    are compared.
    """

    name = "ic"

    def compute(
        self,
        model: nn.Module,
        batch: Batch,
        ctx: LossContext,
    ) -> torch.Tensor:
        x, y, t = batch.ic

        ic = ctx.problem.initial_condition(
            x.detach().cpu().numpy(), y.detach().cpu().numpy()
        )

        h_p, u_p, v_p = model(x, y, t)

        loss = torch.zeros((), device=x.device, dtype=h_p.dtype)
        n_terms = 0

        for name, pred in (("h", h_p), ("u", u_p), ("v", v_p)):
            if name in ic:
                target = torch.as_tensor(ic[name], dtype=pred.dtype, device=pred.device)
                loss = loss + _mse(pred, target)
                n_terms += 1

        if n_terms == 0:
            return loss
        return loss / n_terms


class BoundaryConditionLoss(LossTerm):
    """MSE between the network and the prescribed wall values.

    For each wall, ``problem.boundary_condition`` returns a nested
    dictionary. Only variables that appear in that dictionary are
    penalized; the rest are left free. In particular, a slip wall that
    only prescribes ``v = 0`` will not constrain ``h`` or ``u`` at all.

    Boundary conditions are evaluated in the network's primitive
    variables (``h``, ``u``, ``v``). If a problem were to prescribe
    a conserved quantity instead, this term would need to be extended.
    """

    name = "bc"

    def compute(
        self,
        model: nn.Module,
        batch: Batch,
        ctx: LossContext,
    ) -> torch.Tensor:
        device = ctx.device
        loss = torch.zeros((), device=device, dtype=torch.float32)
        n_terms = 0

        for wall, (x, y, t) in batch.walls.items():
            bc = ctx.problem.boundary_condition(
                wall,
                x.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
                t.detach().cpu().numpy(),
            )
            dirichlet = bc.get("dirichlet", {})
            if not dirichlet:
                continue

            h_p, u_p, v_p = model(x, y, t)
            predictions = {"h": h_p, "u": u_p, "v": v_p}

            for var_name, target_np in dirichlet.items():
                if var_name not in predictions:
                    continue
                pred = predictions[var_name]
                target = torch.as_tensor(
                    target_np, dtype=pred.dtype, device=pred.device
                )
                loss = loss + _mse(pred, target)
                n_terms += 1

        if n_terms == 0:
            return loss
        return loss / n_terms
