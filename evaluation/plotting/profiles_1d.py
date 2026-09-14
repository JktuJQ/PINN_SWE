"""
1D profile plots for comparing solvers against exact solutions.

These plots show h(x), u(x), v(x) along a y-slice and compare numerical
methods (PINN, FVM) against the exact solution."""

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from .styles import apply_paper_style, get_colors, get_color_cycle


def plot_1d_comparison(
    x: np.ndarray,
    h_exact: np.ndarray,
    h_pinn: np.ndarray | None = None,
    h_fvm: np.ndarray | None = None,
    u_exact: np.ndarray | None = None,
    u_pinn: np.ndarray | None = None,
    u_fvm: np.ndarray | None = None,
    t: float = 0.0,
    labels: tuple[str, str, str] = ("Exact", "PINN", "FVM"),
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot 1D profiles of h(x) and u(x) with exact solution.

    Args:
        x: 1D array of x-coordinates.
        h_exact: Exact solution for h.
        h_pinn: PINN solution for h (optional).
        h_fvm: FVM solution for h (optional).
        u_exact: Exact solution for u (optional).
        u_pinn: PINN solution for u (optional).
        u_fvm: FVM solution for u (optional).
        t: Time at which the profiles are evaluated (for title).
        labels: Labels for the three methods.
        save_path: If provided, save the figure to this path.
        show: If True, display the plot interactively.
    """
    apply_paper_style()
    colors = get_colors()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_h = axes[0]
    ax_h.plot(x, h_exact, color=colors["exact"], linewidth=2.0, label=labels[0])
    if h_pinn is not None:
        ax_h.plot(
            x,
            h_pinn,
            color=colors["pinn"],
            linewidth=1.5,
            linestyle="--",
            label=labels[1],
        )
    if h_fvm is not None:
        ax_h.plot(
            x, h_fvm, color=colors["fvm"], linewidth=1.5, linestyle=":", label=labels[2]
        )
    ax_h.set_ylabel("h (depth)")
    ax_h.set_title(f"Water depth h(x, y=0) at t = {t:.2f}")
    ax_h.legend()
    ax_h.grid(alpha=0.3)

    ax_u = axes[1]
    if u_exact is not None:
        ax_u.plot(x, u_exact, color=colors["exact"], linewidth=2.0, label=labels[0])
    if u_pinn is not None:
        ax_u.plot(
            x,
            u_pinn,
            color=colors["pinn"],
            linewidth=1.5,
            linestyle="--",
            label=labels[1],
        )
    if u_fvm is not None:
        ax_u.plot(
            x, u_fvm, color=colors["fvm"], linewidth=1.5, linestyle=":", label=labels[2]
        )
    ax_u.set_xlabel("x")
    ax_u.set_ylabel("u (velocity)")
    ax_u.set_title(f"Velocity u(x, y=0) at t = {t:.2f}")
    ax_u.legend()
    ax_u.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_multi_time_profiles(
    x_list: list[np.ndarray],
    h_list: list[np.ndarray],
    t_list: list[float],
    h_exact_fn: Callable | None = None,
    labels: list[str] | None = None,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot multiple h(x) profiles at different times on one graph.

    Args:
        x_list: List of x-coordinate arrays (one per time).
        h_list: List of h-profile arrays (one per time).
        t_list: List of times.
        h_exact_fn: Optional function exact_h(x, t) for the final time.
        labels: Optional labels for each profile. Defaults to "t = {t:.2f}".
        save_path: If provided, save the figure to this path.
        show: If True, display the plot interactively.
    """
    apply_paper_style()

    cycle = get_color_cycle()

    if labels is None:
        labels = [f"t = {t:.2f}" for t in t_list]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (x, h, t, label) in enumerate(zip(x_list, h_list, t_list, labels)):
        color = cycle[i % len(cycle)]
        ax.plot(x, h, color=color, linewidth=1.5, label=label)

    if h_exact_fn is not None and len(x_list) > 0:
        x_final = x_list[-1]
        t_final = t_list[-1]
        h_exact = h_exact_fn(x_final, t_final)
        ax.plot(
            x_final,
            h_exact,
            color="black",
            linewidth=2.0,
            linestyle="--",
            label=f"Exact, t = {t_final:.2f}",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("h (depth)")
    ax.set_title("Profile h(x, y=0, t)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
