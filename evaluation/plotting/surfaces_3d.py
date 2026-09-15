"""
3D surface plots for presentation-quality visualization.

All functions take the physical domain as a single argument; the sampling
lattice is inferred from the shape of the field.
"""

import numpy as np
import matplotlib.pyplot as plt

from core import RectangularDomain
from .styles import apply_paper_style


def _mesh_from_domain(
    domain: RectangularDomain, n_x: int, n_y: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-center meshgrid ``(X, Y)`` of shape ``(n_x, n_y)``."""
    x = np.linspace(domain.x_min, domain.x_max, n_x)
    y = np.linspace(domain.y_min, domain.y_max, n_y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y


def plot_3d_surface(
    field_2d: dict[str, np.ndarray],
    domain: RectangularDomain,
    var: str = "h",
    t: float = 0.0,
    elevation: float = 30,
    azimuth: float = -60,
    cmap: str = "viridis",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot a 3D surface of a field variable."""
    apply_paper_style()

    data = field_2d[var]
    Nx, Ny = data.shape
    X, Y = _mesh_from_domain(domain, Nx, Ny)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, data, cmap=cmap, edgecolor="none", alpha=0.9, antialiased=True
    )
    ax.view_init(elev=elevation, azim=azimuth)

    ax.set_xlabel("x", fontsize=12, labelpad=10)
    ax.set_ylabel("y", fontsize=12, labelpad=10)
    ax.set_zlabel(var, fontsize=12, labelpad=10)
    ax.set_title(f"{var}(x, y, t={t:.2f})", fontsize=14, pad=20)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=var)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_3d_comparison(
    field_a: dict[str, np.ndarray],
    field_b: dict[str, np.ndarray],
    domain: RectangularDomain,
    var: str = "h",
    labels: tuple[str, str] = ("Method A", "Method B"),
    t: float = 0.0,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot two 3D surfaces side by side."""
    apply_paper_style()

    data_a = field_a[var]
    data_b = field_b[var]
    Nx, Ny = data_a.shape
    X, Y = _mesh_from_domain(domain, Nx, Ny)

    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = ax1.plot_surface(X, Y, data_a, cmap="viridis", edgecolor="none", alpha=0.9)
    ax1.view_init(elev=30, azim=-60)
    ax1.set_xlabel("x", fontsize=11)
    ax1.set_ylabel("y", fontsize=11)
    ax1.set_zlabel(var, fontsize=11)
    ax1.set_title(labels[0], fontsize=13, pad=15)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(X, Y, data_b, cmap="viridis", edgecolor="none", alpha=0.9)
    ax2.view_init(elev=30, azim=-60)
    ax2.set_xlabel("x", fontsize=11)
    ax2.set_ylabel("y", fontsize=11)
    ax2.set_zlabel(var, fontsize=11)
    ax2.set_title(labels[1], fontsize=13, pad=15)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    fig.suptitle(f"3D Comparison: {var}(x, y) at t = {t:.2f}", fontsize=15, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_3d_wireframe(
    field_2d: dict[str, np.ndarray],
    domain: RectangularDomain,
    var: str = "h",
    t: float = 0.0,
    stride: int = 5,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot a 3D wireframe (useful for seeing the sampling lattice)."""
    apply_paper_style()

    data = field_2d[var]
    Nx, Ny = data.shape
    X, Y = _mesh_from_domain(domain, Nx, Ny)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_wireframe(X, Y, data, stride=stride, color="steelblue", alpha=0.7)
    ax.view_init(elev=30, azim=-60)
    ax.set_xlabel("x", fontsize=12, labelpad=10)
    ax.set_ylabel("y", fontsize=12, labelpad=10)
    ax.set_zlabel(var, fontsize=12, labelpad=10)
    ax.set_title(f"{var}(x, y, t={t:.2f}) - Wireframe", fontsize=14, pad=20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
