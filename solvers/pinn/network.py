"""
PINN module: a fully-connected network ``(x, y, t) -> (h, u, v)``.

Design notes
------------
* Only smooth activations. ReLU has zero second derivative and a
  piecewise-constant first derivative, which makes the PDE residual
  meaningless;
* Inputs are normalized to [-1, 1] using buffers, so the network sees
  O(1) numbers at the first layer regardless of the physical domain size.
* ``h`` is produced as ``h_floor + softplus(raw)``, so it is strictly
  positive by construction. No separate positivity penalty is needed.
* Optional Fourier features project the normalized inputs through a
  fixed random matrix (registered as a buffer, not a parameter).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NetworkConfig


__all__ = ["PINN", "build_network"]


def _inv_softplus(z: float) -> float:
    """Return ``x`` such that ``softplus(x) = z`` for ``z > 0``."""
    return math.log(math.expm1(z))


class _Sin(nn.Module):
    """Sine activation, provided locally to avoid depending on
    ``torch.nn.Sin`` (only available in torch >= 1.13)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def _make_activation(name: str) -> nn.Module:
    """Map a config string to an activation module.

    Note: ``"sin"`` is a plain sine with unit frequency. Combined with
    Fourier features this is fine; used alone it tends to be unstable.
    """
    if name == "tanh":
        return nn.Tanh()
    if name == "sin":
        return _Sin()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name!r}")


class PINN(nn.Module):
    """Fully-connected network mapping ``(x, y, t)`` to ``(h, u, v)``.

    Args:
        config: Network architecture configuration.
        x_range: ``(x_min, x_max)`` of the training domain.
        y_range: ``(y_min, y_max)`` of the training domain.
        t_range: ``(t_min, t_max)`` of the training domain.

    The spatial and temporal ranges are used only for input
    normalization; they are stored as buffers, so they move with the
    module when it is sent to a device, and they are saved in the
    ``state_dict``.
    """

    B: torch.Tensor
    in_lo: torch.Tensor
    in_hi: torch.Tensor

    def __init__(
        self,
        config: NetworkConfig,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        t_range: tuple[float, float],
    ) -> None:
        super().__init__()
        self.config = config

        in_dim = 3
        if config.use_fourier:
            B = torch.randn(3, config.fourier_features) * config.fourier_scale
            self.register_buffer("B", B)
            in_dim = 2 * config.fourier_features

        layers: list[nn.Module] = [
            nn.Linear(in_dim, config.hidden),
            _make_activation(config.activation),
        ]
        for _ in range(config.depth - 1):
            layers += [
                nn.Linear(config.hidden, config.hidden),
                _make_activation(config.activation),
            ]
        layers += [nn.Linear(config.hidden, 3)]
        self.net = nn.Sequential(*layers)

        lo = torch.tensor([[x_range[0], y_range[0], t_range[0]]], dtype=torch.float32)
        hi = torch.tensor([[x_range[1], y_range[1], t_range[1]]], dtype=torch.float32)
        self.register_buffer("in_lo", lo)
        self.register_buffer("in_hi", hi)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init for weights, zeros for biases, then a bias shift
        on the h-output so that the initial prediction is close to
        ``h_init`` rather than ``h_floor``."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        target = max(self.config.h_init - self.config.h_floor, 1e-6)
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.bias[0] = _inv_softplus(target)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the network.

        Args:
            x, y, t: Column tensors of shape ``(N, 1)`` on the module's
                device.

        Returns:
            Tuple ``(h, u, v)`` of column tensors of shape ``(N, 1)``.
        """
        inp = torch.cat([x, y, t], dim=1)
        z = 2.0 * (inp - self.in_lo) / (self.in_hi - self.in_lo) - 1.0

        if self.config.use_fourier:
            proj = 2.0 * math.pi * (z @ self.B)
            z = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)

        out = self.net(z)
        h = self.config.h_floor + F.softplus(out[:, 0:1])
        u = out[:, 1:2]
        v = out[:, 2:3]
        return h, u, v

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters (for logging)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"hidden={c.hidden}, depth={c.depth}, activation={c.activation!r}, "
            f"fourier={c.use_fourier}, params={self.num_parameters}"
        )


def build_network(config: NetworkConfig, domain) -> PINN:
    """Construct a ``PINN`` sized to the given domain.

    ``domain`` is a ``RectangularDomain`` from ``core``. This helper
    keeps the caller from passing the three ranges by hand.
    """
    x_range = (domain.x_min, domain.x_max)
    y_range = (domain.y_min, domain.y_max)
    t_range = (domain.t_min, domain.t_max)
    return PINN(config, x_range, y_range, t_range)
