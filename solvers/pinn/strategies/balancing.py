"""
Loss weight balancing.

Different loss terms have different gradient magnitudes. With fixed
weights the smallest-gradient term is drowned out by the largest, so
the network optimizes the dominant term and ignores the rest. Balancing
adjusts weights so that all terms contribute comparably to the gradient.

The gradient-norm strategy follows Wang et al. (2021): for each term
compute the norm of the gradient of that term w.r.t. the model
parameters, and set the weight so the term's contribution matches the
reference term's. The new weights are smoothed with an EMA to avoid
oscillation.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..config import LossConfig


class WeightBalancer(ABC):
    """Adjusts the weights of loss terms between training steps."""

    @abstractmethod
    def rebalance(
        self,
        weights: dict[str, float],
        losses: dict[str, torch.Tensor],
        model: nn.Module,
    ) -> dict[str, float]: ...


class FixedBalancer(WeightBalancer):
    """No balancing: return weights unchanged."""

    def rebalance(self, weights, losses, model):
        return dict(weights)


class GradientNormBalancer(WeightBalancer):
    """Gradient-norm balancing with EMA smoothing and clipping.

    For each non-reference term ``k``:

        target_k = ||grad reference|| / ||grad k||
        w_k_new  = alpha * w_k_old + (1 - alpha) * target_k

    then clip to ``[w_min, w_max]``. The reference term keeps its
    configured weight.
    """

    def __init__(self, cfg: LossConfig) -> None:
        self.cfg = cfg

    def _grad_norm(self, loss: torch.Tensor, model: nn.Module) -> float:
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        total = 0.0
        for gi in grads:
            if gi is not None:
                total += float((gi**2).sum().item())
        return total**0.5

    def rebalance(
        self,
        weights: dict[str, float],
        losses: dict[str, torch.Tensor],
        model: nn.Module,
    ) -> dict[str, float]:
        ref_name = self.cfg.reference_term
        if ref_name not in losses:
            return dict(weights)

        ref_norm = self._grad_norm(losses[ref_name], model)
        if ref_norm < 1e-20:
            return dict(weights)

        out = dict(weights)
        for name, loss in losses.items():
            if name == ref_name:
                continue
            g_norm = self._grad_norm(loss, model)
            if g_norm < 1e-20:
                continue
            target = ref_norm / g_norm
            old = out.get(name, 1.0)
            new = self.cfg.adapt_alpha * old + (1.0 - self.cfg.adapt_alpha) * target
            out[name] = float(min(max(new, self.cfg.w_min), self.cfg.w_max))
        return out
