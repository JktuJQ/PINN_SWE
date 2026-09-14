"""
2D heatmap plots for shallow water fields.

These plots visualize h(x,y), u(x,y), v(x,y) as heatmaps and detect
2D artifacts like y-invariance violations."""

import numpy as np
import matplotlib.pyplot as plt

from .styles import apply_paper_style, get_colors


def plot_heatmap(
    field_2d: dict[str, np.ndarray],
    var: str = "h",
    t: float = 0.0,
    x_range: tuple[float, float] = (-6.0, 6.0),
    y_range: tuple[float, float] = (-6.0, 6.0),
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot a 2D heatmap of a field variable.

    Args:
        field_2d: Dictionary {'h': (Nx,Ny), 'u': (Nx,Ny), 'v': (Nx,Ny)}.
        var: Variable to plot ('h', 'u', or 'v'). Defaults to 'h'.
        t: Time at which the field is evaluated (for title).
        x_range: (x_min, x_max) for axis labels.
        y_range: (y_min, y_max) for axis labels.
        save_path: If provided, save the figure to this path.
        show: If True, display the plot interactively.
    """
    apply_paper_style()
    colors = get_colors()

    data = field_2d[var]
    Nx, Ny = data.shape

    x = np.linspace(x_range[0], x_range[1], Nx)
    y = np.linspace(y_range[0], y_range[1], Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.pcolormesh(X, Y, data, shading="auto", cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{var}(x, y) at t = {t:.2f}")

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
    var: str = "h",
    x_range: tuple[float, float] = (-6.0, 6.0),
    y_range: tuple[float, float] = (-6.0, 6.0),
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot deviation from y-invariance: h(x,y) - <h>_y.

    For the 1D dam break problem embedded in 2D, the exact solution does not
    depend on y. This plot shows how much the numerical solution "wobbles"
    in the y-direction, which is a sign of 2D artifacts.

    Args:
        field_2d: Dictionary {'h': (Nx,Ny), 'u': (Nx,Ny), 'v': (Nx,Ny)}.
        var: Variable to analyze ('h', 'u', or 'v'). Defaults to 'h'.
        x_range: (x_min, x_max) for axis labels.
        y_range: (y_min, y_max) for axis labels.
        save_path: If provided, save the figure to this path.
        show: If True, display the plot interactively.
    """
    apply_paper_style()

    data = field_2d[var]
    Nx, Ny = data.shape

    x = np.linspace(x_range[0], x_range[1], Nx)
    y = np.linspace(y_range[0], y_range[1], Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    mean_y = data.mean(axis=1, keepdims=True)
    deviation = data - mean_y
    max_dev = np.max(np.abs(deviation))

    fig, ax = plt.subplots(figsize=(8, 6))

    vmax = max_dev
    im = ax.pcolormesh(
        X, Y, deviation, shading="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Deviation from y-invariance: {var} - <{var}>_y\n" f"max |dev| = {max_dev:.2e}"
    )

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
    var: str = "h",
    labels: tuple[str, str] = ("Method A", "Method B"),
    t: float = 0.0,
    x_range: tuple[float, float] = (-6.0, 6.0),
    y_range: tuple[float, float] = (-6.0, 6.0),
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot two 2D fields side-by-side with their difference.

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
    diff = data_a - data_b

    Nx, Ny = data_a.shape
    x = np.linspace(x_range[0], x_range[1], Nx)
    y = np.linspace(y_range[0], y_range[1], Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].pcolormesh(X, Y, data_a, shading="auto", cmap="viridis")
    axes[0].set_title(f"{labels[0]}: {var}(x, y)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, Y, data_b, shading="auto", cmap="viridis")
    axes[1].set_title(f"{labels[1]}: {var}(x, y)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1])

    vmax = np.max(np.abs(diff))
    im2 = axes[2].pcolormesh(
        X, Y, diff, shading="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax
    )
    axes[2].set_title(
        f"Difference: {labels[0]} - {labels[1]}\n" f"max |diff| = {vmax:.2e}"
    )
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(f"Comparison at t = {t:.2f}", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
