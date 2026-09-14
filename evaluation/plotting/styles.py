"""
Global plotting styles for scientific publications.

This module provides consistent styling for all plots in the project.
Call `apply_paper_style()` once at the beginning of your script to ensure
all figures look professional and publication-ready.
"""

import matplotlib.pyplot as plt


def apply_paper_style() -> None:
    """Apply a clean, professional style suitable for scientific papers.

    This sets:
    - Font sizes and families (serif for papers)
    - Line widths and marker sizes
    - Grid styles
    - Color cycles
    - Figure DPI

    Call this once at the start of your plotting script.
    """
    plt.rcParams.update(
        {
            # Font settings
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            # Axes settings
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.grid.which": "major",
            "axes.axisbelow": True,
            # Grid settings
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            # Tick settings
            "xtick.labelsize": 10,
            "xtick.major.size": 4,
            "xtick.minor.size": 2,
            "xtick.direction": "in",
            "ytick.labelsize": 10,
            "ytick.major.size": 4,
            "ytick.minor.size": 2,
            "ytick.direction": "in",
            # Legend settings
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.3",
            # Figure settings
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            # Line settings
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "lines.markeredgewidth": 0.5,
            # Color cycle (professional colors)
            "axes.prop_cycle": plt.cycler(
                color=[
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
                ]
            ),
            # Text settings
            "text.usetex": False,
            "mathtext.fontset": "dejavuserif",
        }
    )


def get_color_cycle() -> list[str]:
    """
    Return the current color cycle as a list of hex colors.

    Useful for manually assigning colors to plot elements.
    """
    return [c["color"] for c in plt.rcParams["axes.prop_cycle"]]


def get_colors() -> dict[str, str]:
    """Return a dictionary of named colors for common plot elements.

    Returns:
        Dictionary with keys like 'pinn', 'fvm', 'exact', etc.
    """
    cycle = get_color_cycle()
    return {
        "exact": "black",
        "pinn": cycle[0],
        "fvm": cycle[2],
        "error": cycle[1],
    }
