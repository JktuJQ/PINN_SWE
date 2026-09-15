"""
Samplers and the factory that builds one from a config.
"""

import torch

from core import BaseProblem
from ..config import SamplingConfig
from .base import WALLS, BaseSampler, Batch, Tensor3
from .uniform import UniformSampler
from .cone import ConeSampler, ConeRARSampler


__all__ = [
    "Batch",
    "BaseSampler",
    "Tensor3",
    "WALLS",
    "UniformSampler",
    "ConeSampler",
    "ConeRARSampler",
    "build_sampler",
]


def build_sampler(
    config: SamplingConfig,
    problem: BaseProblem,
    device: torch.device,
) -> BaseSampler:
    """Instantiate a sampler from a config.

    Raises:
        ValueError: if ``config.type`` is not a registered sampler.
    """
    if config.type == "uniform":
        return UniformSampler(problem, config, device)
    if config.type == "cone":
        return ConeSampler(problem, config, device)
    if config.type == "cone_rar":
        return ConeRARSampler(problem, config, device)
    raise ValueError(f"Unknown sampler type: {config.type!r}")
