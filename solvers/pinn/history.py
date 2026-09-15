"""
Training history as a structured dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingHistory:
    """Per-epoch records of a training run.

    Attributes:
        epochs:    Global step counter (Adam + L-BFGS share the axis).
        phase:     ``"adam"`` or ``"lbfgs"`` for each step.
        loss:      Total weighted loss.
        terms:     ``{term_name: [value per step]}`` for each loss term.
        weights:   ``{term_name: [weight per step]}`` as used by the
                   balancer. Weights may change over training.
        nu:        Artificial viscosity at each step.
        t_max:     Time horizon at each step.
        meta:      Free-form slot for scalar extras (e.g. learning rate).
    """

    epochs: list[int] = field(default_factory=list)
    phase: list[str] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    terms: dict[str, list[float]] = field(default_factory=dict)
    weights: dict[str, list[float]] = field(default_factory=dict)
    nu: list[float] = field(default_factory=list)
    t_max: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def record(
        self,
        epoch: int,
        phase: str,
        loss: float,
        terms: dict[str, float],
        weights: dict[str, float],
        nu: float,
        t_max: float,
    ) -> None:
        """Append one step to the history, creating term slots lazily."""
        self.epochs.append(epoch)
        self.phase.append(phase)
        self.loss.append(float(loss))
        for name, value in terms.items():
            self.terms.setdefault(name, []).append(float(value))
        for name, value in weights.items():
            self.weights.setdefault(name, []).append(float(value))
        self.nu.append(float(nu))
        self.t_max.append(float(t_max))

    def __len__(self) -> int:
        return len(self.epochs)
