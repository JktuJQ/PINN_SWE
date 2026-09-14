"""
Shallow Water Equations: FVM-specific operators.

These functions handle conservative variables (h, hu, hv) and physical fluxes."""

import numpy as np

H_MIN = 1e-8


def primitive_to_conservative(
    h: np.ndarray, u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert primitive (h, u, v) to conservative (h, hu, hv)."""
    return h, h * u, h * v


def conservative_to_primitive(
    h: np.ndarray, hu: np.ndarray, hv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert conservative (h, hu, hv) to primitive (h, u, v)."""
    h_safe = np.maximum(h, H_MIN)
    u = np.where(h > H_MIN, hu / h_safe, 0.0)
    v = np.where(h > H_MIN, hv / h_safe, 0.0)
    return h, u, v


def flux_x(
    h: np.ndarray, hu: np.ndarray, hv: np.ndarray, g: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical flux in x-direction: F = [hu, hu^2 + 0.5*g*h^2, huv]."""
    h, u, v = conservative_to_primitive(h, hu, hv)
    return (
        hu,
        hu * u + 0.5 * g * h**2,
        hu * v,
    )


def flux_y(
    h: np.ndarray, hu: np.ndarray, hv: np.ndarray, g: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical flux in y-direction: G = [hv, huv, hv^2 + 0.5*g*h^2]."""
    h, u, v = conservative_to_primitive(h, hu, hv)
    return (
        hv,
        hu * v,
        hv * v + 0.5 * g * h**2,
    )


def max_wave_speeds(
    h: np.ndarray, u: np.ndarray, v: np.ndarray, g: float
) -> tuple[float, float]:
    """Maximum wave speeds in x and y for the 2D CFL condition."""
    c = np.sqrt(g * np.maximum(h, 0.0))
    return float(np.max(np.abs(u) + c)), float(np.max(np.abs(v) + c))
