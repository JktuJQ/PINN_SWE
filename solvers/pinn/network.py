"""
PINN network: (x, y, t) -> (h, u, v).

h is positive by construction via softplus; inputs are normalized to
[-1, 1] using the domain bounds; only smooth activations are used
(ReLU breaks the PDE residual).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import RectangularDomain


def _inv_softplus(z: float) -> float:
    return math.log(math.expm1(z))


class PINN(nn.Module):
    """Fully-connected network mapping (x, y, t) to (h, u, v).

    Architecture parameters are stored as attributes so that a
    checkpoint can be rebuilt without external bookkeeping.
    """

    in_lo: torch.Tensor
    in_hi: torch.Tensor

    def __init__(
        self,
        domain: RectangularDomain,
        hidden: int = 128,
        depth: int = 6,
        h_floor: float = 1e-3,
        h_init: float = 1.5,
    ) -> None:
        super().__init__()
        self.h_floor = h_floor

        self.hidden = hidden
        self.depth = depth
        self.h_init = h_init

        layers: list[nn.Module] = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 3)]
        self.net = nn.Sequential(*layers)

        lo = torch.tensor(
            [[domain.x_min, domain.y_min, domain.t_min]], dtype=torch.float32
        )
        hi = torch.tensor(
            [[domain.x_max, domain.y_max, domain.t_max]], dtype=torch.float32
        )
        self.register_buffer("in_lo", lo)
        self.register_buffer("in_hi", hi)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.bias[0] = _inv_softplus(max(self.h_init - self.h_floor, 1e-6))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp = torch.cat([x, y, t], dim=1)
        z = 2.0 * (inp - self.in_lo) / (self.in_hi - self.in_lo) - 1.0
        out = self.net(z)
        h = self.h_floor + F.softplus(out[:, 0:1])
        return h, out[:, 1:2], out[:, 2:3]


def save_pinn(model: PINN, path: str) -> None:
    """Save a PINN checkpoint: weights + architecture + domain bounds."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": model.hidden,
            "depth": model.depth,
            "h_floor": model.h_floor,
            "h_init": model.h_init,
        },
        path,
    )


def _infer_arch_from_state_dict(state_dict: dict) -> dict:
    """Recover ``hidden`` and ``depth`` from a raw state_dict.

    Layer layout in ``PINN.net`` (see ``__init__``):

        index 0            : Linear(3, hidden)       -- input
        indices 2, 4, ...  : Linear(hidden, hidden)  -- (depth-1) hidden
        index 2 * depth    : Linear(hidden, 3)       -- output

    So the total number of ``Linear`` layers is ``depth + 1``, and

        hidden = state_dict["net.0.weight"].shape[0]
        depth  = (number of Linear layers) - 1
    """
    linear_indices = sorted(
        int(k.split(".")[1])
        for k in state_dict
        if k.startswith("net.") and k.endswith(".weight")
    )
    if not linear_indices:
        raise ValueError("state_dict has no 'net.*.weight' entries")

    first = state_dict[f"net.{linear_indices[0]}.weight"]
    hidden = int(first.shape[0])
    n_linears = len(linear_indices)
    depth = n_linears - 1

    return {"hidden": hidden, "depth": depth}


def load_pinn(domain: RectangularDomain, path: str) -> PINN:
    """Rebuild a PINN from a checkpoint and load its weights.

    Accepts two formats:

    * new: ``{"state_dict": ..., "hidden": ..., "depth": ...,
      "h_floor": ..., "h_init": ...}`` — produced by ``save_pinn``.
    * legacy: a bare state_dict (as saved by the older ``pinn_model.py``).
      Architecture is inferred from tensor shapes; ``h_floor`` and
      ``h_init`` fall back to their defaults.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        hidden = int(ckpt["hidden"])
        depth = int(ckpt["depth"])
        h_floor = float(ckpt.get("h_floor", 1e-3))
        h_init = float(ckpt.get("h_init", 1.5))
    else:
        state_dict = ckpt
        inferred = _infer_arch_from_state_dict(state_dict)
        hidden = inferred["hidden"]
        depth = inferred["depth"]
        h_floor = 1e-3
        h_init = 1.5

    model = PINN(
        domain,
        hidden=hidden,
        depth=depth,
        h_floor=h_floor,
        h_init=h_init,
    )
    model.load_state_dict(state_dict)
    return model
