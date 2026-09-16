"""
Minimal PINN stack for the shallow water equations.

Public API
----------
Model:
    PINN            fully-connected network (x, y, t) -> (h, u, v)
    save_pinn       write a checkpoint (weights + architecture)
    load_pinn       rebuild a PINN from a checkpoint

Physics:
    swe_residual    autograd-based SWE residual at collocation points

Training:
    train_pinn      full training loop (Adam + L-BFGS, curriculum, RAR)
    sample_batch    build one batch of collocation / IC / wall points
    compute_losses  evaluate the individual loss terms (pde, ic, bc)
    rebalance_weights  gradient-norm loss weight balancing
    refine          residual-adaptive refinement of collocation points
    WALLS           canonical wall names matching BaseProblem.boundary_condition
"""

from .network import PINN, load_pinn, save_pinn
from .residual import swe_residual
from .train import (
    WALLS,
    compute_losses,
    rebalance_weights,
    refine,
    sample_batch,
    train_pinn,
)
