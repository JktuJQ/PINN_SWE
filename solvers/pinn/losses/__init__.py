"""
Loss terms and the factory that builds them from a config.

Built-in terms live in ``standard.py``. Custom terms can be added by
instantiating them directly and passing the list to the trainer; the
factory below only handles the built-ins.
"""

from .base import LossContext, LossTerm
from .standard import (
    BoundaryConditionLoss,
    InitialConditionLoss,
    PDEResidualLoss,
)

__all__ = [
    "LossContext",
    "LossTerm",
    "PDEResidualLoss",
    "InitialConditionLoss",
    "BoundaryConditionLoss",
    "BUILTIN_TERMS",
    "build_loss_terms",
]


BUILTIN_TERMS: dict[str, type[LossTerm]] = {
    "pde": PDEResidualLoss,
    "ic": InitialConditionLoss,
    "bc": BoundaryConditionLoss,
}


def build_loss_terms(terms: tuple[str, ...]) -> list[LossTerm]:
    """Instantiate the requested loss terms, preserving order.

    Raises:
        ValueError: if a name is not a registered built-in.
    """
    out: list[LossTerm] = []
    for name in terms:
        cls = BUILTIN_TERMS.get(name)
        if cls is None:
            known = sorted(BUILTIN_TERMS)
            raise ValueError(
                f"unknown loss term {name!r}; known built-ins: {known}. "
                f"Pass custom terms as instances to the Trainer, not by name."
            )
        out.append(cls())
    return out
