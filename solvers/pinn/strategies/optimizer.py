"""
Optimizer strategies.

A strategy wraps a torch optimizer and exposes a uniform ``step`` method
that takes a zero-argument loss function (a closure). Adam ignores the
closure semantics beyond calling it once; L-BFGS uses it for its line
search. Composing the two is a strategy of its own, driven by an
internal epoch counter.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn

from ..config import TrainingConfig


LossFn = Callable[[], torch.Tensor]


class OptimizerStrategy(ABC):
    """Uniform interface over a torch optimizer."""

    @abstractmethod
    def step(self, loss_fn: LossFn) -> float:
        """Perform one optimization step. Return the loss value as float."""

    def state_dict(self) -> dict:
        """Optional, for checkpointing. Default: empty."""
        return {}


class AdamStrategy(OptimizerStrategy):
    """Adam with exponential LR decay from ``lr`` to ``lr_final``."""

    def __init__(self, params, cfg: TrainingConfig) -> None:
        self.optimizer = torch.optim.Adam(params, lr=cfg.lr)
        epochs = max(cfg.adam_epochs, 1)
        gamma = (cfg.lr_final / cfg.lr) ** (1.0 / epochs)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=gamma
        )

    def step(self, loss_fn: LossFn) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g["params"]],
            max_norm=1.0,
        )
        self.optimizer.step()
        self.scheduler.step()
        return float(loss.detach().item())

    def state_dict(self) -> dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }


class LBFGSStrategy(OptimizerStrategy):
    """L-BFGS with strong-Wolfe line search."""

    def __init__(self, params, max_iter: int) -> None:
        self.optimizer = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=max_iter,
            history_size=50,
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

    def step(self, loss_fn: LossFn) -> float:
        # L-BFGS calls the closure multiple times; we don't get a single
        # scalar back from `optimizer.step`, so recompute the loss once
        # at the end for logging.
        def closure() -> torch.Tensor:
            self.optimizer.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            return loss

        self.optimizer.step(closure)
        with torch.no_grad():
            value = float(loss_fn().detach().item())
        return value


class AdamThenLBFGSStrategy(OptimizerStrategy):
    """Run Adam for ``adam_epochs``, then L-BFGS for ``lbfgs_steps``.

    The epoch counter is internal: each call to ``step`` increments it
    and dispatches to whichever phase is currently active. This makes
    the strategy self-contained and lets the trainer stay agnostic of
    the schedule.
    """

    def __init__(self, params, cfg: TrainingConfig) -> None:
        self.adam = AdamStrategy(params, cfg)
        self.lbfgs = LBFGSStrategy(params, max_iter=cfg.lbfgs_steps)
        self.adam_epochs = cfg.adam_epochs
        self.lbfgs_steps = cfg.lbfgs_steps
        self._epoch = 0

    @property
    def phase(self) -> str:
        return "adam" if self._epoch < self.adam_epochs else "lbfgs"

    def step(self, loss_fn: LossFn) -> float:
        value = (
            self.adam.step(loss_fn)
            if self.phase == "adam"
            else self.lbfgs.step(loss_fn)
        )
        self._epoch += 1
        return value

    def state_dict(self) -> dict:
        return {"adam": self.adam.state_dict(), "epoch": self._epoch}
