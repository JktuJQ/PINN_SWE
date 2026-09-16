"""
Problem definitions and a small registry mapping names to constructors.

The registry lets scripts pick a problem with ``--problem NAME``
without importing the classes explicitly.
"""

from __future__ import annotations

from core import RectangularDomain
from .circular_dam_break import CircularDamBreak
from .dam_break_1d import DamBreak1D


__all__ = [
    "DamBreak1D",
    "CircularDamBreak",
    "make_problem",
    "available_problems",
    "default_domain",
]


_DEFAULTS = {
    "1d": {
        "domain": ((-6.0, 6.0), (-6.0, 6.0), (0.0, 1.0)),
        "cls": DamBreak1D,
    },
    "circular": {
        "domain": ((-8.0, 8.0), (-8.0, 8.0), (0.0, 1.0)),
        "cls": CircularDamBreak,
    },
}


def available_problems() -> list[str]:
    return sorted(_DEFAULTS)


def default_domain(name: str) -> RectangularDomain:
    if name not in _DEFAULTS:
        raise ValueError(f"Unknown problem {name!r}; available: {available_problems()}")
    xr, yr, tr = _DEFAULTS[name]["domain"]
    return RectangularDomain(xr, yr, tr)


def make_problem(name: str, domain: RectangularDomain):
    if name not in _DEFAULTS:
        raise ValueError(f"Unknown problem {name!r}; available: {available_problems()}")
    return _DEFAULTS[name]["cls"](domain)
