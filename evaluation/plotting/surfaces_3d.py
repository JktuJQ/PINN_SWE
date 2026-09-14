"""
3D surface plots for presentation-quality visualization.

These plots use matplotlib's 3D toolkit to create surface plots of h(x,y), u(x,y), v(x,y).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .styles import apply_paper_style, get_colors


def plot_3d_surface(
    field_2d: dict[str, np.ndarray],
    var: str = "h",
    t: float = 0.0,
    x_range: tuple[float, float] = (-6.0, 6.0),
    y_range: tuple[float, float] = (-6.0, 6.0),
    elevation: float = 30,
    azimuth: float = -60,
    cmap: str = "viridis",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot a 3D surface of a field variable.

    Args:
        field_2d: Dictionary {'h': (Nx,Ny), 'u': (Nx,Ny), 'v': (Nx,Ny)}.
        var: Variable to plot ('h', 'u', or 'v'). Defaults to 'h'.
        t: Time at which the field is evaluated (for title).
        x_range: (x_min, x_max) for axis labels.
        y_range: (y_min, y_max) for axis labels.
        elevation: Viewing elevation angle in degrees. Defaults to 30.
        azimuth: Viewing azimuth angle in degrees. Defaults to -60.
        cmap: Colormap name. Defaults to 'viridis'.
        save_path: If provided, save the figure to this path.
        show: If True, display the plot interactively.
    """
    apply_paper_style()

    data = field_2d[var]
    Nx, Ny = data.shape

    x = np.linspace(x_range[0], x_range[1], Nx)
    y = np.linspace(y_range[0], y_range[1], Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        data,
        cmap=cmap,
        edgecolor="none",
        alpha=0.9,
        antialiased=True,
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
    var: str = "h",
    labels: tuple[str, str] = ("Method A", "Method B"),
    t: float = 0.0,
    x_range: tuple[float, float] = (-6.0, 6.0),
    y_range: tuple[float, float] = (-6.0, 6.0),
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot two 3D surfaces side-by-side for comparison.

    Args:
        field_a: First field dictionary.
        field_b: Second field dictionary.
        var: Variable to compare ('h', 'u', or 'v').
        labels: Labels for the two methods.
        t: Time at which the fields are evaluated (for title).
        x_range: (x_min, x_max) for axis labels.
        y_range: (y_min, y_max) for axis labels.
        save_path: If provided, save the figure to this path.
        show: If True, display the plot interactively.
    """
    apply_paper_style()

    data_a = field_a[var]
    data_b = field_b[var]

    Nx, Ny = data_a.shape
    x = np.linspace(x_range[0], x_range[1], Nx)
    y = np.linspace(y_range[0], y_range[1], Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = ax1.plot_surface(X, Y, data_a, cmap="viridis", edgecolor="none", alpha=0.9)
    ax1.view_init(elev=30, azim=-60)
    ax1.set_xlabel("x", fontsize=11)
    ax1.set_ylabel("y", fontsize=11)
    ax1.set_zlabel(var, fontsize=11)
    ax1.set_title(f"{labels[0]}", fontsize=13, pad=15)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(X, Y, data_b, cmap="viridis", edgecolor="none", alpha=0.9)
    ax2.view_init(elev=30, azim=-60)
    ax2.set_xlabel("x", fontsize=11)
    ax2.set_ylabel("y", fontsize=11)
    ax2.set_zlabel(var, fontsize=11)
    ax2.set_title(f"{labels[1]}", fontsize=13, pad=15)
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
    var: str = "h",
    t: float = 0.0,
    x_range: tuple[float, float] = (-6.0, 6.0),
    y_range: tuple[float, float] = (-6.0, 6.0),
    stride: int = 5,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot a 3D wireframe (useful for seeing the grid structure).

    Args:
        field_2d: Dictionary {'h': (Nx,Ny), 'u': (Nx,Ny), 'v': (Nx,Ny)}.
        var: Variable to plot ('h', 'u', or 'v'). Defaults to 'h'.
        t: Time at which the field is evaluated (for title).
        x_range: (x_min, x_max) for axis labels.
        y_range: (y_min, y_max) for axis labels.
        stride: Step size for wireframe lines. Defaults to 5.
        save_path: If provided, save the figure to this path.
        show: If True, display the plot interactively.
    """
    apply_paper_style()

    data = field_2d[var]
    Nx, Ny = data.shape

    x = np.linspace(x_range[0], x_range[1], Nx)
    y = np.linspace(y_range[0], y_range[1], Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

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
