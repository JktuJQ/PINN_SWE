"""
Strategy factories: map config strings to concrete objects.

Each factory takes the sub-config, the objects it needs (model
parameters, time bounds, ...), and returns an instance. This is the
only place where string names from YAML are translated into classes.
"""

from __future__ import annotations

import torch

from ..config import (
    CurriculumConfig,
    LossConfig,
    TrainingConfig,
    ViscosityConfig,
)
from .optimizer import (
    AdamStrategy,
    AdamThenLBFGSStrategy,
    LBFGSStrategy,
    OptimizerStrategy,
)
from .curriculum import (
    ConstantCurriculum,
    Curriculum,
    TimeMarchingCurriculum,
)
from .viscosity import (
    AnnealedViscosity,
    ConstantViscosity,
    NoViscosity,
    ViscositySchedule,
)
from .balancing import (
    FixedBalancer,
    GradientNormBalancer,
    WeightBalancer,
)


__all__ = [
    "OptimizerStrategy",
    "AdamStrategy",
    "LBFGSStrategy",
    "AdamThenLBFGSStrategy",
    "Curriculum",
    "ConstantCurriculum",
    "TimeMarchingCurriculum",
    "ViscositySchedule",
    "NoViscosity",
    "ConstantViscosity",
    "AnnealedViscosity",
    "WeightBalancer",
    "FixedBalancer",
    "GradientNormBalancer",
    "build_optimizer",
    "build_curriculum",
    "build_viscosity",
    "build_balancer",
]


def build_optimizer(
    cfg: TrainingConfig,
    params,
) -> OptimizerStrategy:
    if cfg.optimizer == "adam":
        return AdamStrategy(params, cfg)
    if cfg.optimizer == "lbfgs":
        return LBFGSStrategy(params, max_iter=cfg.lbfgs_steps)
    if cfg.optimizer == "adam_then_lbfgs":
        return AdamThenLBFGSStrategy(params, cfg)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")


def build_curriculum(
    cfg: CurriculumConfig,
    t_start: float,
    t_end: float,
) -> Curriculum:
    if cfg.type == "constant":
        return ConstantCurriculum(t_start, t_end)
    if cfg.type == "time_marching":
        return TimeMarchingCurriculum(t_start, t_end, cfg.stages, cfg.frac)
    raise ValueError(f"Unknown curriculum: {cfg.type!r}")


def build_viscosity(cfg: ViscosityConfig) -> ViscositySchedule:
    if cfg.type == "none":
        return NoViscosity()
    if cfg.type == "constant":
        return ConstantViscosity(cfg.end)
    if cfg.type == "annealed":
        return AnnealedViscosity(cfg.start, cfg.end, cfg.frac)
    raise ValueError(f"Unknown viscosity schedule: {cfg.type!r}")


def build_balancer(cfg: LossConfig) -> WeightBalancer:
    if cfg.balancer == "fixed":
        return FixedBalancer()
    if cfg.balancer == "gradient_norm":
        return GradientNormBalancer(cfg)
    raise ValueError(f"Unknown balancer: {cfg.balancer!r}")
