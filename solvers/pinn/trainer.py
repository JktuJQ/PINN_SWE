"""
PINN trainer: wires everything together and runs the training loop.

The trainer is deliberately thin. It owns the loop structure, the
sequencing of strategy calls, and the history log; it does not know how
the optimizer updates weights, how the curriculum schedules time, or
what the loss terms actually compute. All of that lives in the strategy
objects and the loss terms.

Loop skeleton::

    for epoch in range(total_steps):
        nu    = viscosity.nu(epoch, total)
        t_max = curriculum.t_max(epoch, total)
        batch = sampler.sample(t_max)
        if should_refine(epoch):    sampler.refine(...)
        if should_rebalance(epoch): weights = balancer.rebalance(...)
        value = optimizer.step(loss_fn)
        history.record(...)

The only subtleties are (a) the loss closure is recomputed on every
inner optimizer call, because LBFGS's line search needs to see updated
parameters, and (b) the weights dictionary is read from ``self`` inside
the closure, so reassigning it before the step takes effect without
rebuilding the closure.
"""

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from core import BaseProblem

from .config import PINNConfig
from .history import TrainingHistory
from .losses import LossContext, LossTerm, build_loss_terms
from .sampling import BaseSampler, build_sampler
from .strategies import (
    build_balancer,
    build_curriculum,
    build_optimizer,
    build_viscosity,
)


__all__ = ["Trainer", "resolve_device"]


def resolve_device(spec: str) -> torch.device:
    """Resolve ``"auto"`` to the best available device, or parse a name.

    ``"auto"`` prefers CUDA, then Apple MPS, then CPU. Anything else is
    passed to ``torch.device`` verbatim, so ``"cpu"``, ``"cuda:1"``,
    ``"mps"`` all work.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


class Trainer:
    """Orchestrates a PINN training run.

    Args:
        model:       The network (typically a ``PINN``).
        problem:     Mathematical problem (IC, BC, parameters).
        residual_fn: Callable ``(model, x, y, t, nu) -> Tensor`` returning
                     the PDE residual. Usually a ``SWEResidual`` instance.
        config:      Full training configuration.
        loss_terms:  Optional override of the loss terms. When ``None``,
                     they are built from ``config.losses.terms``.
        device:      Optional override of ``config.training.device``.

    Attributes:
        weights:     Current per-term weights. Updated by the balancer;
                     can be inspected after ``train()`` for diagnostics.
    """

    def __init__(
        self,
        model: nn.Module,
        problem: BaseProblem,
        residual_fn: Callable[..., torch.Tensor],
        config: PINNConfig,
        loss_terms: list[LossTerm] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.problem = problem
        self.residual_fn = residual_fn
        self.config = config

        spec = device if device is not None else config.training.device
        self.device = resolve_device(spec) if isinstance(spec, str) else spec
        self.model.to(self.device)

        self.loss_terms: list[LossTerm] = (
            list(loss_terms)
            if loss_terms is not None
            else build_loss_terms(config.losses.terms)
        )

        self.weights: dict[str, float] = dict(config.losses.weights)
        for term in self.loss_terms:
            self.weights.setdefault(term.name, 1.0)

        self.optimizer = build_optimizer(config.training, self.model.parameters())
        self.curriculum = build_curriculum(
            config.curriculum,
            t_start=problem.domain.t_min,
            t_end=problem.domain.t_max,
        )
        self.viscosity = build_viscosity(config.viscosity)
        self.balancer = build_balancer(config.losses)

        self.sampler: BaseSampler = build_sampler(config.sampling, problem, self.device)

    def _total_steps(self) -> int:
        """Number of optimizer.step() calls in the whole run.

        Adam contributes ``adam_epochs`` steps; LBFGS contributes one
        step (its inner loop handles ``lbfgs_steps`` iterations);
        Adam-then-LBFGS is the sum of both.
        """
        tr = self.config.training
        if tr.optimizer == "adam":
            return max(tr.adam_epochs, 1)
        if tr.optimizer == "lbfgs":
            return 1
        return max(tr.adam_epochs, 1) + 1

    def _phase(self, epoch: int) -> str:
        tr = self.config.training
        if tr.optimizer == "adam":
            return "adam"
        if tr.optimizer == "lbfgs":
            return "lbfgs"
        return "adam" if epoch < tr.adam_epochs else "lbfgs"

    def _should_refine(self, epoch: int) -> bool:
        """RAR runs only during the Adam phase, on a fixed interval."""
        s = self.config.sampling
        if s.rar_every <= 0 or epoch == 0:
            return False
        if epoch % s.rar_every != 0:
            return False
        return self._phase(epoch) == "adam"

    def _should_rebalance(self, epoch: int) -> bool:
        """Balancing runs only during the Adam phase, on a fixed interval."""
        l = self.config.losses
        if l.balancer == "fixed" or l.adapt_every <= 0:
            return False
        if epoch == 0 or epoch % l.adapt_every != 0:
            return False
        return self._phase(epoch) == "adam"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> TrainingHistory:
        """Run the full training loop and return the recorded history."""
        torch.manual_seed(self.config.training.seed)
        np.random.seed(self.config.training.seed)

        history = TrainingHistory()
        total = self._total_steps()

        for epoch in range(total):
            nu = self.viscosity.nu(epoch, total)
            t_max = self.curriculum.t_max(epoch, total)

            batch = self.sampler.sample(t_max)

            if self._should_refine(epoch):
                self.sampler.refine(self.model, self.residual_fn, t_max, nu)
                batch = self.sampler.sample(t_max)

            ctx = LossContext(
                problem=self.problem,
                residual_fn=self.residual_fn,
                nu=nu,
                t_max=t_max,
                device=self.device,
            )

            if self._should_rebalance(epoch):
                with torch.enable_grad():
                    probe = {
                        t.name: t.compute(self.model, batch, ctx)
                        for t in self.loss_terms
                    }
                self.weights = self.balancer.rebalance(self.weights, probe, self.model)
                del probe

            last_terms: dict[str, torch.Tensor] = {}

            def loss_fn() -> torch.Tensor:
                nonlocal last_terms
                last_terms = {
                    t.name: t.compute(self.model, batch, ctx) for t in self.loss_terms
                }
                total_loss = torch.zeros((), device=self.device)
                for name, value in last_terms.items():
                    total_loss = total_loss + self.weights[name] * value
                return total_loss

            total_loss_value = self.optimizer.step(loss_fn)

            term_values = {k: float(v.detach().item()) for k, v in last_terms.items()}
            history.record(
                epoch=epoch,
                phase=self._phase(epoch),
                loss=total_loss_value,
                terms=term_values,
                weights=dict(self.weights),
                nu=nu,
                t_max=t_max,
            )

        return history
