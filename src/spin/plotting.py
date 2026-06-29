"""Shared plotting style and small reusable figure helpers.

These keep the example notebooks short. The palette matches the paper figures:
FOM = charcoal, static = grey, baseline adaptive = coral, SPIN = teal.
"""

from __future__ import annotations

import numpy as np

# consistent palette / display names across all figures
PALETTE = {
    "fom": "#2b2b2b",       # full-order model (charcoal)
    "ic": "#9a9a9a",        # initial condition (grey)
    "static": "#8a8a8a",    # static ROM (grey, dashed)
    "baseline": "#e8744f",  # baseline adaptive ROM (coral)
    "spin": "#159a8c",      # SPIN ROM (teal)
    "corr": "#3b6ea5",      # correction-event marker (steel blue)
}
NAMES = {"baseline": "baseline adaptive", "spin": "SPIN"}


def use_paper_style():
    """Apply a clean, readable, vector-friendly Matplotlib style."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.size": 14, "axes.titlesize": 16, "axes.titleweight": "bold",
        "axes.labelsize": 15, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "legend.frameon": False, "lines.linewidth": 2.2,
    })


def plot_profiles(ax_list, x, ic, fom, static, baseline, spin, steps, dt):
    """Plot solution profiles (one axis per snapshot time) for all four models."""
    P = PALETTE
    for ax, k in zip(ax_list, steps):
        ax.plot(x, ic, color=P["ic"], lw=1.6, ls=":", label="initial condition")
        ax.plot(x, fom[:, k], color=P["fom"], lw=3.8, alpha=0.9, label="FOM")
        ax.plot(x, static[:, k], color=P["static"], lw=1.8, ls="--", label="static ROM")
        ax.plot(x, baseline[:, k], color=P["baseline"], lw=2.2, label="baseline adaptive")
        ax.plot(x, spin[:, k], color=P["spin"], lw=2.2, label="SPIN")
        ax.set_title(rf"$t={k*dt:.2f}$  (step {k})", fontweight="bold", fontsize=13)
        ax.set_xlabel(r"$x$")


def plot_error_history(ax, err_static, err_baseline, err_spin, zs, title=None):
    """Plot the relative-L2 error history (log-y) with correction guide lines."""
    P = PALETTE
    n = len(err_static) - 1
    steps = np.arange(n + 1)
    for s in range(zs, n + 1, zs):
        ax.axvline(s, color=P["corr"], lw=0.7, alpha=0.15, zorder=0)
    ax.plot([], [], color=P["corr"], lw=0.9, alpha=0.5,
            label=rf"out-of-span corrections ($z_s={zs}$)")
    ax.semilogy(steps, err_static, color=P["static"], lw=2.0, ls="--", label="static ROM")
    ax.semilogy(steps, err_baseline, color=P["baseline"], lw=2.4, label="baseline adaptive")
    ax.semilogy(steps, err_spin, color=P["spin"], lw=2.4, label="SPIN")
    ax.set_xlabel("time step")
    ax.set_ylabel(r"relative $L_2$ error")
    ax.set_xlim(0, n)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.15, which="both", lw=0.5)
