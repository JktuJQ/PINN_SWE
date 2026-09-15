"""
Global plotting styles for scientific publications.
"""

import matplotlib.pyplot as plt


def apply_paper_style() -> None:
    """Apply a clean, professional style suitable for scientific papers."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.grid.which": "major",
            "axes.axisbelow": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "xtick.labelsize": 10,
            "xtick.major.size": 4,
            "xtick.minor.size": 2,
            "xtick.direction": "in",
            "ytick.labelsize": 10,
            "ytick.major.size": 4,
            "ytick.minor.size": 2,
            "ytick.direction": "in",
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.3",
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "lines.markeredgewidth": 0.5,
            "text.usetex": False,
            "mathtext.fontset": "dejavuserif",
        }
    )


# Explicit palette. Order is stable and independent of rcParams.
_PALETTE: tuple[str, ...] = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def get_color_cycle() -> list[str]:
    """Return the project palette as a list of hex colors."""
    return list(_PALETTE)


def get_colors() -> dict[str, str]:
    """Named colors for the common plot elements.

    ``pinn`` / ``fvm`` / ``exact`` / ``error`` are stable across runs and
    independent of the rcParams cycle.
    """
    return {
        "exact": "black",
        "pinn": "#1f77b4",
        "fvm": "#2ca02c",
        "error": "#d62728",
    }
