"""
Time-evolution animations for 1D profiles, 2D fields, and 3D surfaces.

Architecture:
  - Low-level functions (animate_1d, animate_2d, animate_3d) accept ready data.
  - High-level wrappers (animate_*_fvm, animate_*_pinn) extract data via
    extraction/profiles.py and delegate to low-level functions.
"""

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from core import StructuredGrid2D
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
    u_list: list[np.ndarray] | None = None,
    exact_fn: Callable[[np.ndarray, float], dict[str, np.ndarray]] | None = None,
    var: str = "h",
    y_label: str = "h (depth)",
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate a 1D profile from pre-extracted data.

    Args:
        x_list: List of x-coordinate arrays (one per frame).
        h_list: List of h-profile arrays (one per frame).
        t_list: List of times.
        u_list: Optional list of u-profile arrays (unused in current plotting, kept for API).
        exact_fn: Optional function exact_fn(x, t) -> dict with exact solution.
        var: Variable name (for title).
        y_label: Label for y-axis.
        fps: Frames per second.
        save_path: If provided, save animation to this path.
        show: If True, display interactively.

    Returns:
        FuncAnimation object.
    """
    apply_paper_style()
    colors = get_colors()

    fig, ax = plt.subplots(figsize=(10, 5))

    y_min = min(v.min() for v in h_list)
    y_max = max(v.max() for v in h_list)
    margin = 0.1 * (y_max - y_min + 1e-6)

    ax.set_xlim(x_list[0].min(), x_list[0].max())
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
    x_range: tuple[float, float] = (-6.0, 6.0),
    y_range: tuple[float, float] = (-6.0, 6.0),
    var: str = "h",
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate a 2D heatmap from pre-extracted data.

    Args:
        field_list: List of dicts {'h': (Nx,Ny), 'u': (Nx,Ny), 'v': (Nx,Ny)}.
        times: List of times.
        x_range: (x_min, x_max) for axis labels.
        y_range: (y_min, y_max) for axis labels.
        var: Variable to animate.
        fps: Frames per second.
        save_path: If provided, save animation.
        show: If True, display interactively.

    Returns:
        FuncAnimation object.
    """
    apply_paper_style()

    data_0 = field_list[0][var]
    Nx, Ny = data_0.shape
    x = np.linspace(x_range[0], x_range[1], Nx)
    y = np.linspace(y_range[0], y_range[1], Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    all_vals = [f[var] for f in field_list]
    vmin = min(v.min() for v in all_vals)
    vmax = max(v.max() for v in all_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(
        X, Y, data_0, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
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
    x_range: tuple[float, float] = (-6.0, 6.0),
    y_range: tuple[float, float] = (-6.0, 6.0),
    var: str = "h",
    elevation: float = 30,
    azimuth: float = -60,
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate a 3D surface from pre-extracted data.

    Args:
        field_list: List of dicts {'h': (Nx,Ny), 'u': (Nx,Ny), 'v': (Nx,Ny)}.
        times: List of times.
        x_range: (x_min, x_max) for axis labels.
        y_range: (y_min, y_max) for axis labels.
        var: Variable to animate.
        elevation: Viewing elevation angle.
        azimuth: Viewing azimuth angle.
        fps: Frames per second.
        save_path: If provided, save animation.
        show: If True, display interactively.

    Returns:
        FuncAnimation object.
    """
    apply_paper_style()

    data_0 = field_list[0][var]
    Nx, Ny = data_0.shape
    x = np.linspace(x_range[0], x_range[1], Nx)
    y = np.linspace(y_range[0], y_range[1], Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    all_vals = [f[var] for f in field_list]
    zmin = min(v.min() for v in all_vals)
    zmax = max(v.max() for v in all_vals)

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
    grid: StructuredGrid2D,
    times: list[float],
    exact_fn: Callable[[np.ndarray, float], dict[str, np.ndarray]] | None = None,
    var: str = "h",
    y_val: float = 0.0,
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate 1D profile from FVM snapshots.

    Extracts data via extract_1d_from_fvm and delegates to animate_1d.
    """
    x_list, h_list, u_list = [], [], []
    for snap in snapshots:
        x, h, u, v = extract_1d_from_fvm(snap, grid, y_val=y_val)
        x_list.append(x)
        h_list.append(h)
        u_list.append(u)

    return animate_1d(
        x_list,
        h_list,
        times,
        u_list=u_list,
        exact_fn=exact_fn,
        var=var,
        fps=fps,
        save_path=save_path,
        show=show,
    )


def animate_1d_pinn(
    model,
    t_start: float = 0.0,
    t_end: float = 1.0,
    n_frames: int = 50,
    x_range: tuple[float, float] = (-6.0, 6.0),
    y_val: float = 0.0,
    n_points: int = 500,
    exact_fn: Callable[[np.ndarray, float], dict[str, np.ndarray]] | None = None,
    var: str = "h",
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate 1D profile from PINN model.

    Evaluates model via evaluate_pinn_1d and delegates to animate_1d.
    """
    times = np.linspace(t_start, t_end, n_frames).tolist()
    x_list, h_list, u_list = [], [], []

    for t in times:
        x, h, u, v = evaluate_pinn_1d(model, t, x_range, y_val, n_points)
        x_list.append(x)
        h_list.append(h)
        u_list.append(u)

    return animate_1d(
        x_list,
        h_list,
        times,
        u_list=u_list,
        exact_fn=exact_fn,
        var=var,
        fps=fps,
        save_path=save_path,
        show=show,
    )


def animate_2d_fvm(
    snapshots: list[dict[str, np.ndarray]],
    grid: StructuredGrid2D,
    times: list[float],
    var: str = "h",
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate 2D heatmap from FVM snapshots.

    Uses grid bounds to ensure correct axis mapping and delegates to animate_2d.
    """
    x_range = (grid.x_min, grid.x_max)
    y_range = (grid.y_min, grid.y_max)

    return animate_2d(snapshots, times, x_range, y_range, var, fps, save_path, show)


def animate_2d_pinn(
    model,
    t_start: float = 0.0,
    t_end: float = 1.0,
    n_frames: int = 50,
    x_range: tuple[float, float] = (-6.0, 6.0),
    y_range: tuple[float, float] = (-6.0, 6.0),
    n_points: int = 200,
    var: str = "h",
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate 2D heatmap from PINN model.

    Evaluates model via evaluate_pinn_2d and delegates to animate_2d.
    """
    times = np.linspace(t_start, t_end, n_frames).tolist()
    field_list = []

    for t in times:
        field = evaluate_pinn_2d(model, t, x_range, y_range, n_points)
        field_list.append(field)

    return animate_2d(field_list, times, x_range, y_range, var, fps, save_path, show)


def animate_3d_fvm(
    snapshots: list[dict[str, np.ndarray]],
    grid: StructuredGrid2D,
    times: list[float],
    var: str = "h",
    elevation: float = 30,
    azimuth: float = -60,
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate 3D surface from FVM snapshots.

    Uses grid bounds to ensure correct axis mapping and delegates to animate_3d.
    """
    x_range = (grid.x_min, grid.x_max)
    y_range = (grid.y_min, grid.y_max)

    return animate_3d(
        snapshots,
        times,
        x_range,
        y_range,
        var,
        elevation,
        azimuth,
        fps,
        save_path,
        show,
    )


def animate_3d_pinn(
    model,
    t_start: float = 0.0,
    t_end: float = 1.0,
    n_frames: int = 50,
    x_range: tuple[float, float] = (-6.0, 6.0),
    y_range: tuple[float, float] = (-6.0, 6.0),
    n_points: int = 200,
    var: str = "h",
    elevation: float = 30,
    azimuth: float = -60,
    fps: int = 10,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate 3D surface from PINN model.

    Evaluates model via evaluate_pinn_2d and delegates to animate_3d.
    """
    times = np.linspace(t_start, t_end, n_frames).tolist()
    field_list = []

    for t in times:
        field = evaluate_pinn_2d(model, t, x_range, y_range, n_points)
        field_list.append(field)

    return animate_3d(
        field_list,
        times,
        x_range,
        y_range,
        var,
        elevation,
        azimuth,
        fps,
        save_path,
        show,
    )
