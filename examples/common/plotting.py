"""Shared matplotlib setup so all figures look alike."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Render to files; no display needed.

import matplotlib.pyplot as plt  # noqa: E402

# Colorblind-safe palette (Okabe-Ito).
COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]


def new_figure(width: float = 7.0, height: float = 4.0, **subplots_kw):
    """Create a figure and axes with the house style.

    Extra keyword arguments go to ``plt.subplots`` (for example ``nrows``).
    """
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.3,
            "font.size": 10,
        }
    )
    return plt.subplots(figsize=(width, height), layout="constrained", **subplots_kw)


def save(fig, path: Path) -> Path:
    """Save a figure as PNG, create the folder if needed, and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
