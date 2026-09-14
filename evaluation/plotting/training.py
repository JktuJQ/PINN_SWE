"""
Training history plots for PINN.

These plots visualize the loss curves, viscosity annealing, and time
horizon progression during PINN training."""

import matplotlib.pyplot as plt

from .styles import apply_paper_style


def plot_training_history(
    history: dict[str, list[float]],
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot training history: loss components, viscosity, and time horizon.

    Args:
        history: Dictionary with keys:
            - 'loss': total loss at each epoch
            - 'pde': PDE residual loss
            - 'ic': initial condition loss
            - 'bc': boundary condition loss
            - 'nu': artificial viscosity at each epoch
            - 't_max': time horizon at each epoch
        save_path: If provided, save the figure to this path.
        show: If True, display the plot interactively.
    """
    apply_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax_loss = axes[0]
    ax_loss.semilogy(history["loss"], label="Total loss", linewidth=2.0)
    ax_loss.semilogy(history["pde"], label="PDE residual", alpha=0.75)
    ax_loss.semilogy(history["ic"], label="Initial condition", alpha=0.75)
    ax_loss.semilogy(history["bc"], label="Boundary condition", alpha=0.75)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (log scale)")
    ax_loss.set_title("PINN Training History")
    ax_loss.legend()
    ax_loss.grid(True, which="both", alpha=0.3)

    ax_sched = axes[1]
    ax_sched.plot(history["nu"], color="tab:red", label=r"Viscosity $\nu$")
    ax_sched.set_xlabel("Epoch")
    ax_sched.set_ylabel(r"$\nu$", color="tab:red")
    ax_sched.set_yscale("log")
    ax_sched.tick_params(axis="y", labelcolor="tab:red")

    ax_twin = ax_sched.twinx()
    ax_twin.plot(history["t_max"], color="tab:blue", label=r"Time horizon $t_{max}$")
    ax_twin.set_ylabel(r"Time horizon $t_{max}$", color="tab:blue")
    ax_twin.tick_params(axis="y", labelcolor="tab:blue")

    ax_sched.set_title("Schedule: Viscosity and Time Horizon")
    ax_sched.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
