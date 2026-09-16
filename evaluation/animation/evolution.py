"""
Time-evolution animations for 1D profiles, 2D fields, and 3D surfaces.

Architecture:
  - Low-level functions (animate_1d, animate_2d, animate_3d) accept ready
    data together with the physical domain, and render the animation.
  - High-level wrappers (animate_*_fvm, animate_*_pinn) extract data via
    extraction/profiles.py and delegate to the low-level functions.

The domain is passed explicitly at every level. This keeps the rendering
independent of whichever solver produced the data: FVM grids and PINN
samplings render through the same code path.
"""

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from core import RectangularDomain
from evaluation.plotting.styles import apply_paper_style, get_colors
from evaluation.extraction.profiles import (
    extract_1d_from_fvm,
    evaluate_pinn_1d,
    evaluate_pinn_2d,
)


def animate_1d(
    x_list: list[np.ndarray],
    h_list: list[np.ndarray],
    t_list: list[float],
    domain: RectangularDomain,
    exact_fn: Callable[[np.ndarray, float], dict[str, np.ndarray]] | None = None,
    var: str = "h",
    y_label: str = "h (depth)",
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate a 1D profile from pre-extracted data.

    Args:
        x_list:   List of x-coordinate arrays (one per frame).
        h_list:   List of h-profile arrays (one per frame).
        t_list:   List of times (one per frame).
        domain:   Physical domain; sets the x-axis limits.
        exact_fn: Optional function ``exact_fn(x, t) -> {var: array}`` for
                  an overlay of the exact solution.
        var:      Variable name (used in title and to pick the exact field).
        y_label:  Label for the y-axis.
        fps:      Frames per second.
        save_path: If provided, save animation to this path.
        show:      If True, display interactively.

    Returns:
        The FuncAnimation object (the caller may need to keep a reference
        until the animation is saved).
    """
    apply_paper_style()
    colors = get_colors()

    fig, ax = plt.subplots(figsize=(10, 5))

    y_min = min(float(v.min()) for v in h_list)
    y_max = max(float(v.max()) for v in h_list)
    margin = 0.1 * (y_max - y_min + 1e-6)

    ax.set_xlim(domain.x_min, domain.x_max)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel("x")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)

    (line_num,) = ax.plot([], [], color=colors["fvm"], linewidth=1.5, label="Numerical")
    (line_exact,) = ax.plot(
        [], [], color=colors["exact"], linewidth=2.0, linestyle="--", label="Exact"
    )
    title = ax.set_title("")
    ax.legend(loc="upper right")

    def init() -> tuple:
        line_num.set_data([], [])
        line_exact.set_data([], [])
        title.set_text("")
        return line_num, line_exact, title

    def update(frame: int) -> tuple:
        t = t_list[frame]
        line_num.set_data(x_list[frame], h_list[frame])
        title.set_text(f"{var}(x, y=0) at t = {t:.3f}")

        if exact_fn is not None:
            exact = exact_fn(x_list[frame], t)
            if exact is not None and var in exact:
                line_exact.set_data(x_list[frame], exact[var])

        return line_num, line_exact, title

    anim = FuncAnimation(
        fig, update, init_func=init, frames=len(x_list), interval=1000 // fps, blit=True
    )

    if save_path:
        writer = "ffmpeg" if save_path.endswith(".mp4") else "pillow"
        anim.save(save_path, writer=writer, fps=fps, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    return anim


def animate_2d(
    field_list: list[dict[str, np.ndarray]],
    times: list[float],
    domain: RectangularDomain,
    var: str = "h",
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate a 2D heatmap from pre-extracted data."""
    apply_paper_style()

    data_0 = field_list[0][var]
    Nx, Ny = data_0.shape
    x = np.linspace(domain.x_min, domain.x_max, Nx)
    y = np.linspace(domain.y_min, domain.y_max, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    all_vals = [f[var] for f in field_list]
    vmin = min(float(v.min()) for v in all_vals)
    vmax = max(float(v.max()) for v in all_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(
        X, Y, data_0, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    title = ax.set_title(f"{var}(x, y) at t = {times[0]:.3f}")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(var)

    def update(frame: int) -> tuple:
        im.set_array(field_list[frame][var].ravel())
        title.set_text(f"{var}(x, y) at t = {times[frame]:.3f}")
        return im, title

    anim = FuncAnimation(
        fig, update, frames=len(field_list), interval=1000 // fps, blit=False
    )

    if save_path:
        writer = "ffmpeg" if save_path.endswith(".mp4") else "pillow"
        anim.save(save_path, writer=writer, fps=fps, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    return anim


def animate_3d(
    field_list: list[dict[str, np.ndarray]],
    times: list[float],
    domain: RectangularDomain,
    var: str = "h",
    elevation: float = 30,
    azimuth: float = -60,
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate a 3D surface from pre-extracted data."""
    apply_paper_style()

    data_0 = field_list[0][var]
    Nx, Ny = data_0.shape
    x = np.linspace(domain.x_min, domain.x_max, Nx)
    y = np.linspace(domain.y_min, domain.y_max, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    all_vals = [f[var] for f in field_list]
    zmin = min(float(v.min()) for v in all_vals)
    zmax = max(float(v.max()) for v in all_vals)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, data_0, cmap="viridis", edgecolor="none", alpha=0.9, vmin=zmin, vmax=zmax
    )
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlabel("x", fontsize=12, labelpad=10)
    ax.set_ylabel("y", fontsize=12, labelpad=10)
    ax.set_zlabel(var, fontsize=12, labelpad=10)
    title = ax.set_title(f"{var}(x, y) at t = {times[0]:.3f}", fontsize=14, pad=20)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=var)

    xlim = (domain.x_min, domain.x_max)
    ylim = (domain.y_min, domain.y_max)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(zmin - 0.2 * zmin, zmax + 0.2 * zmax)

    def update(frame: int) -> list:
        ax.clear()
        ax.plot_surface(
            X,
            Y,
            field_list[frame][var],
            cmap="viridis",
            edgecolor="none",
            alpha=0.9,
            vmin=zmin,
            vmax=zmax,
        )
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_xlabel("x", fontsize=12, labelpad=10)
        ax.set_ylabel("y", fontsize=12, labelpad=10)
        ax.set_zlabel(var, fontsize=12, labelpad=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(zmin - 0.2 * zmin, zmax + 0.2 * zmax)
        ax.set_title(f"{var}(x, y) at t = {times[frame]:.3f}", fontsize=14, pad=20)
        return []

    anim = FuncAnimation(
        fig, update, frames=len(field_list), interval=1000 // fps, blit=False
    )

    if save_path:
        writer = "ffmpeg" if save_path.endswith(".mp4") else "pillow"
        anim.save(save_path, writer=writer, fps=fps, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    return anim


def animate_1d_fvm(
    snapshots: list[dict[str, np.ndarray]],
    grid,
    domain: RectangularDomain,
    times: list[float],
    exact_fn: Callable[[np.ndarray, float], dict[str, np.ndarray]] | None = None,
    var: str = "h",
    y_val: float = 0.0,
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate a 1D profile from FVM snapshots.

    ``grid`` is used to locate the y-slice; ``domain`` is used only for
    rendering.
    """
    x_list, h_list = [], []
    for snap in snapshots:
        x, h, _, _ = extract_1d_from_fvm(snap, grid, y_val=y_val)
        x_list.append(x)
        h_list.append(h)

    return animate_1d(
        x_list,
        h_list,
        times,
        domain,
        exact_fn=exact_fn,
        var=var,
        fps=fps,
        save_path=save_path,
        show=show,
    )


def animate_2d_fvm(
    snapshots: list[dict[str, np.ndarray]],
    domain: RectangularDomain,
    times: list[float],
    var: str = "h",
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate a 2D heatmap from FVM snapshots."""
    return animate_2d(snapshots, times, domain, var, fps, save_path, show)


def animate_3d_fvm(
    snapshots: list[dict[str, np.ndarray]],
    domain: RectangularDomain,
    times: list[float],
    var: str = "h",
    elevation: float = 30,
    azimuth: float = -60,
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate a 3D surface from FVM snapshots."""
    return animate_3d(
        snapshots,
        times,
        domain,
        var,
        elevation,
        azimuth,
        fps,
        save_path,
        show,
    )


def animate_1d_pinn(
    model,
    domain: RectangularDomain,
    t_start: float = 0.0,
    t_end: float = 1.0,
    n_frames: int = 50,
    y_val: float = 0.0,
    n_points: int = 500,
    exact_fn: Callable[[np.ndarray, float], dict[str, np.ndarray]] | None = None,
    var: str = "h",
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate a 1D profile from a PINN model."""
    times = np.linspace(t_start, t_end, n_frames).tolist()
    x_list, h_list = [], []

    x_range = (domain.x_min, domain.x_max)
    for t in times:
        x, h, _, _ = evaluate_pinn_1d(model, t, x_range, y_val, n_points)
        x_list.append(x)
        h_list.append(h)

    return animate_1d(
        x_list,
        h_list,
        times,
        domain,
        exact_fn=exact_fn,
        var=var,
        fps=fps,
        save_path=save_path,
        show=show,
    )


def animate_2d_pinn(
    model,
    domain: RectangularDomain,
    t_start: float = 0.0,
    t_end: float = 1.0,
    n_frames: int = 50,
    n_points: int = 200,
    var: str = "h",
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate a 2D heatmap from a PINN model."""
    times = np.linspace(t_start, t_end, n_frames).tolist()
    field_list = []

    x_range = (domain.x_min, domain.x_max)
    y_range = (domain.y_min, domain.y_max)
    for t in times:
        field = evaluate_pinn_2d(model, t, x_range, y_range, n_points)
        field_list.append(field)

    return animate_2d(field_list, times, domain, var, fps, save_path, show)


def animate_3d_pinn(
    model,
    domain: RectangularDomain,
    t_start: float = 0.0,
    t_end: float = 1.0,
    n_frames: int = 50,
    n_points: int = 200,
    var: str = "h",
    elevation: float = 30,
    azimuth: float = -60,
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate a 3D surface from a PINN model."""
    times = np.linspace(t_start, t_end, n_frames).tolist()
    field_list = []

    x_range = (domain.x_min, domain.x_max)
    y_range = (domain.y_min, domain.y_max)
    for t in times:
        field = evaluate_pinn_2d(model, t, x_range, y_range, n_points)
        field_list.append(field)

    return animate_3d(
        field_list,
        times,
        domain,
        var,
        elevation,
        azimuth,
        fps,
        save_path,
        show,
    )
