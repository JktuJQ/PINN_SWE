"""
2D heatmap plots for shallow water fields.

These plots visualize h(x,y), u(x,y), v(x,y) as heatmaps and detect
2D artifacts like y-invariance violations.

All functions take the physical domain as a single argument. The number
of cells along each axis is inferred from the shape of the field, so the
same call works for FVM output (a 200×200 grid) and for a PINN sampled on
an arbitrary 500×500 lattice.
"""

import numpy as np
import matplotlib.pyplot as plt

from core import RectangularDomain
from .styles import apply_paper_style


def _axes_from_domain(
    domain: RectangularDomain, n_x: int, n_y: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-center coordinates ``(X, Y)`` of shape ``(n_x, n_y)``."""
    x = np.linspace(domain.x_min, domain.x_max, n_x)
    y = np.linspace(domain.y_min, domain.y_max, n_y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y


def plot_heatmap(
    field_2d: dict[str, np.ndarray],
    domain: RectangularDomain,
    var: str = "h",
    t: float = 0.0,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot a 2D heatmap of a field variable.

    Args:
        field_2d: Dictionary ``{'h': (Nx,Ny), 'u': (Nx,Ny), 'v': (Nx,Ny)}``.
        domain:   Physical domain; determines axis limits and coordinates.
        var:      Variable to plot (``'h'``, ``'u'`` or ``'v'``).
        t:        Time (used only for the title).
        save_path: If provided, save the figure to this path.
        show:      If True, display the plot interactively.
    """
    apply_paper_style()

    data = field_2d[var]
    Nx, Ny = data.shape
    X, Y = _axes_from_domain(domain, Nx, Ny)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(X, Y, data, shading="auto", cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{var}(x, y) at t = {t:.2f}")
    ax.set_aspect("equal", adjustable="box")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(var)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_y_deviation(
    field_2d: dict[str, np.ndarray],
    domain: RectangularDomain,
    var: str = "h",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot deviation from y-invariance: ``field(x,y) − <field>_y``.

    For 1D problems embedded in 2D the exact solution does not depend on
    ``y``, so any non-zero deviation is a genuine 2D artifact rather than
    a physical feature.
    """
    apply_paper_style()

    data = field_2d[var]
    Nx, Ny = data.shape
    X, Y = _axes_from_domain(domain, Nx, Ny)

    deviation = data - data.mean(axis=1, keepdims=True)
    max_dev = float(np.max(np.abs(deviation)))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(
        X,
        Y,
        deviation,
        shading="auto",
        cmap="coolwarm",
        vmin=-max_dev,
        vmax=max_dev,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Deviation from y-invariance: {var} − <{var}>_y\n" f"max |dev| = {max_dev:.2e}"
    )
    ax.set_aspect("equal", adjustable="box")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"{var} deviation")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_comparison_2d(
    field_a: dict[str, np.ndarray],
    field_b: dict[str, np.ndarray],
    domain: RectangularDomain,
    var: str = "h",
    labels: tuple[str, str] = ("Method A", "Method B"),
    t: float = 0.0,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot two 2D fields side by side together with their difference."""
    apply_paper_style()

    data_a = field_a[var]
    data_b = field_b[var]
    diff = data_a - data_b

    Nx, Ny = data_a.shape
    X, Y = _axes_from_domain(domain, Nx, Ny)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].pcolormesh(X, Y, data_a, shading="auto", cmap="viridis")
    axes[0].set_title(f"{labels[0]}: {var}(x, y)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal", adjustable="box")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, Y, data_b, shading="auto", cmap="viridis")
    axes[1].set_title(f"{labels[1]}: {var}(x, y)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal", adjustable="box")
    plt.colorbar(im1, ax=axes[1])

    vmax = float(np.max(np.abs(diff)))
    if vmax <= 0.0:
        vmax = 1e-12
    im2 = axes[2].pcolormesh(
        X, Y, diff, shading="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax
    )
    axes[2].set_title(f"Difference: {labels[0]} − {labels[1]}\nmax |diff| = {vmax:.2e}")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal", adjustable="box")
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(f"Comparison at t = {t:.2f}", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
