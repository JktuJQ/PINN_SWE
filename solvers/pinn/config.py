"""
Configuration schemas for PINN training (pydantic v2).

Design notes
------------
* Every schema is a frozen pydantic model. Concrete experiments live as
  YAML files under ``scripts/configs/``; this module loads them, validates
  them against the schemas, and applies CLI overrides.
* ``extra="forbid"`` on every model: unknown keys in YAML are an error,
  not silently ignored.
* Nested models are declared as fields, so writing
  ``PINNConfig(training={"adam_epochs": 400})`` from Python works without
  any boilerplate — pydantic unpacks the dict automatically.
* Values are validated at construction time, so a bad YAML fails before
  any training starts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


__all__ = [
    "ConfigError",
    "NetworkConfig",
    "SamplingConfig",
    "LossConfig",
    "ViscosityConfig",
    "CurriculumConfig",
    "TrainingConfig",
    "PINNConfig",
    "load_config",
    "save_config",
    "apply_overrides",
]


ConfigError = ValidationError


class _ConfigBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NetworkConfig(_ConfigBase):
    """Architecture of the PINN module."""

    hidden: int = Field(default=128, gt=0)
    depth: int = Field(default=6, ge=1)
    activation: Literal["tanh", "sin", "gelu"] = "tanh"
    use_fourier: bool = False
    fourier_features: int = Field(default=64, gt=0)
    fourier_scale: float = Field(default=2.0, gt=0.0)
    h_floor: float = Field(default=1e-3, gt=0.0)
    h_init: float = Field(default=1.5, gt=0.0)


class SamplingConfig(_ConfigBase):
    """Where collocation, initial and boundary points come from."""

    type: Literal["uniform", "cone", "cone_rar"] = "cone_rar"
    n_interior: int = Field(default=8192, gt=0)
    n_ic: int = Field(default=2048, gt=0)
    n_bc: int = Field(default=1024, gt=0)
    cone_frac: float = Field(default=0.5, ge=0.0, le=1.0)
    cone_margin: float = Field(default=0.5, ge=0.0)
    rar_every: int = Field(default=500, ge=0)
    rar_pool: int = Field(default=40000, gt=0)
    rar_keep: int = Field(default=2048, gt=0)


class LossConfig(_ConfigBase):
    """Which loss terms to use and how to weight them."""

    terms: tuple[str, ...] = ("pde", "ic", "bc")
    weights: dict[str, float] = Field(
        default_factory=lambda: {"pde": 1.0, "ic": 10.0, "bc": 10.0}
    )
    balancer: Literal["fixed", "gradient_norm"] = "gradient_norm"
    adapt_every: int = Field(default=250, ge=0)
    adapt_alpha: float = Field(default=0.9, ge=0.0, le=1.0)
    w_min: float = Field(default=1e-2, gt=0.0)
    w_max: float = Field(default=1e4, gt=0.0)
    reference_term: str = "pde"


class ViscosityConfig(_ConfigBase):
    """Artificial viscosity schedule (vanishing-viscosity method)."""

    type: Literal["none", "constant", "annealed"] = "annealed"
    start: float = Field(default=0.10, ge=0.0)
    end: float = Field(default=0.005, ge=0.0)
    frac: float = Field(default=0.6, gt=0.0, le=1.0)


class CurriculumConfig(_ConfigBase):
    """Time-horizon schedule and causal weighting."""

    type: Literal["constant", "time_marching"] = "time_marching"
    stages: int = Field(default=4, ge=1)
    frac: float = Field(default=0.4, gt=0.0, le=1.0)
    n_time_bins: int = Field(default=16, ge=1)
    causal_eps: float = Field(default=2.0, ge=0.0)


class TrainingConfig(_ConfigBase):
    """Outer training loop."""

    optimizer: Literal["adam", "lbfgs", "adam_then_lbfgs"] = "adam_then_lbfgs"
    adam_epochs: int = Field(default=8000, ge=0)
    lbfgs_steps: int = Field(default=400, ge=0)
    lr: float = Field(default=2e-3, gt=0.0)
    lr_final: float = Field(default=1e-5, gt=0.0)
    seed: int = 0
    device: str = "auto"


class PINNConfig(_ConfigBase):
    """Top-level configuration for one PINN training run."""

    network: NetworkConfig = Field(default_factory=NetworkConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    losses: LossConfig = Field(default_factory=LossConfig)
    viscosity: ViscosityConfig = Field(default_factory=ViscosityConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def load_config(path: str | Path) -> PINNConfig:
    """Load a YAML file into a validated ``PINNConfig``.

    Raises:
        ConfigError: if the file is missing, not a mapping, contains
            unknown keys, or has a field set to an unsupported value.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    try:
        return PINNConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"{path}:\n{e}") from e


def save_config(cfg: PINNConfig, path: str | Path) -> None:
    """Serialize a config to YAML, for reproducibility records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            cfg.model_dump(mode="python"),
            fh,
            sort_keys=False,
            default_flow_style=False,
        )


def _set_nested(data: dict, keys: list[str], value) -> None:
    cur = data
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            raise ConfigError(f"override path {'.'.join(keys)}: no such key {k!r}")
        cur = cur[k]
    if keys[-1] not in cur:
        raise ConfigError(f"override path {'.'.join(keys)}: no such key {keys[-1]!r}")
    cur[keys[-1]] = value


def _parse_override(s: str) -> tuple[list[str], object]:
    if "=" not in s:
        raise ConfigError(f"override must be KEY=VALUE, got {s!r}")
    dotted, _, raw = s.partition("=")
    keys = dotted.strip().split(".")
    if not all(keys):
        raise ConfigError(f"empty key in override {s!r}")
    return keys, yaml.safe_load(raw)


def apply_overrides(cfg: PINNConfig, overrides: list[str]) -> PINNConfig:
    """Return a new config with dot-notation overrides applied.

    Values are parsed as YAML, so ``training.adam_epochs=400`` sets an int
    and ``network.use_fourier=true`` sets a bool. The result is
    re-validated, so an out-of-range value fails here, not at training
    time.
    """
    if not overrides:
        return cfg

    data = cfg.model_dump(mode="python")
    for ov in overrides:
        keys, value = _parse_override(ov)
        _set_nested(data, keys, value)

    return PINNConfig.model_validate(data)
