"""Regenerate every figure in the repository from the model.

Run ``python -m bsdose.figures`` (or ``python src/bsdose/figures.py``) to write
the PNGs into ``figures/``. Nothing here is hand-drawn: each panel is computed
from :mod:`bsdose.ternary`, :mod:`bsdose.corridor` and :mod:`bsdose.screen`.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ternary import trimer, peak_concentration
from .corridor import occupancy, corridor
from .screen import screen_panel, screen_candidate

TEAL, CORAL, AMBER, PURPLE, GREY = "#0E8C6B", "#E24B4A", "#C8860B", "#6A4FB0", "#8A8A82"
BAND = "#E9E6DD"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#8A8A82", "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#5F5E5A", "ytick.color": "#5F5E5A",
    "axes.labelcolor": "#2C2C2A", "figure.dpi": 150,
})

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.normpath(os.path.join(HERE, "..", "..", "figures"))

# reference lead candidate (matches the corridor figure numbers)
LEAD = dict(KA=16.0, KB=1.0, avidity=21.0, MUC1=5.0, CD40=0.5)
WEAK = dict(KA=16.0, KB=1.0, avidity=1.0, MUC1=5.0, CD40=0.5)


def _tumor_curve(pars, grid):
    ka, kb, av = pars["KA"] / pars["avidity"], pars["KB"] / pars["avidity"], 1.0
    return trimer(grid, ka, kb, pars["MUC1"], pars["CD40"])


def fig_trimer_bell(path):
    grid = np.logspace(-3, 3, 4000)
    T = _tumor_curve(LEAD, grid)
    Tn = T / T.max()
    pk = grid[int(np.argmax(T))]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.semilogx(grid, Tn, color=TEAL, lw=2.6)
    ax.axvline(pk, color=TEAL, ls="--", lw=1.0, alpha=0.7)
    ax.plot([pk], [1.0], "o", color=TEAL, ms=7)
    ax.annotate("max trimer\nfree drug $\\approx \\sqrt{K_A K_B}$", xy=(pk, 1.0),
                xytext=(pk * 0.02, 0.82), color="#2C2C2A", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7, ls=":"))
    ax.text(pk * 0.02, 0.30, "too little drug\nno bridging", color="#5F5E5A", fontsize=9.5, ha="left")
    ax.text(pk * 40, 0.55, "prozone\neach arm saturated", color="#5F5E5A", fontsize=9.5, ha="left")
    ax.set_xlabel("free drug concentration (nM, log)")
    ax.set_ylabel("trimer (normalised)")
    ax.set_ylim(0, 1.12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_dosability_corridor(path):
    grid = np.logspace(-3, 2, 5000)
    T = _tumor_curve(LEAD, grid)
    Tn = T / T.max()
    occ = occupancy(grid, LEAD["KB"])
    floor, ceil = corridor(LEAD["KB"], 0.10, 0.25)
    pk = grid[int(np.argmax(T))]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.axvspan(floor, ceil, color=BAND, alpha=0.9, lw=0)
    ax.semilogx(grid, Tn, color=TEAL, lw=2.6, label="tumor trimer (efficacy)")
    ax.semilogx(grid, occ, color=CORAL, lw=2.4, label="systemic CD40 occupancy (safety)")
    ax.axhline(0.10, color=CORAL, ls="--", lw=0.8, alpha=0.5)
    ax.axhline(0.25, color=CORAL, ls="--", lw=0.8, alpha=0.5)
    ax.text(grid[0] * 1.4, 0.11, "10% occ", color="#9A3520", fontsize=9)
    ax.text(grid[0] * 1.4, 0.26, "25% occ", color="#9A3520", fontsize=9)
    for x, c, lab in [(floor, AMBER, "MABEL floor"), (pk, TEAL, "optimum"), (ceil, CORAL, "ceiling")]:
        ax.axvline(x, color=c, ls="--", lw=1.1)
    ax.plot([floor], [0.10], "o", color=AMBER, ms=6)
    ax.plot([ceil], [0.25], "o", color=CORAL, ms=6)
    ax.plot([pk], [1.0], "o", color=TEAL, ms=6)
    ax.annotate("dosable corridor", xy=((floor * ceil) ** 0.5, 1.02),
                ha="center", fontsize=10, color="#2C2C2A")
    ax.set_xlabel("free drug concentration (nM, log)")
    ax.set_ylabel("trimer (norm.) / occupancy")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="center left", frameon=True, facecolor="white", framealpha=0.9,
              edgecolor="none", fontsize=9.5, bbox_to_anchor=(0.015, 0.58))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_avidity_failure(path):
    grid = np.logspace(-3, 2.5, 5000)
    Ts = _tumor_curve(LEAD, grid); Ts /= Ts.max()
    Tw = _tumor_curve(WEAK, grid); Tw /= Tw.max()
    floor, ceil = corridor(LEAD["KB"], 0.10, 0.25)
    # weak-avidity productive shoulder onset
    lo_w = grid[Tw >= 0.5][0]

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.axvspan(floor, ceil, color=BAND, alpha=0.9, lw=0)
    ax.semilogx(grid, Ts, color=TEAL, lw=2.6, label="strong avidity — viable")
    ax.semilogx(grid, Tw, color=TEAL, lw=2.2, ls="--", label="weak avidity — bell slid right")
    ax.axhline(0.5, color=GREY, ls="--", lw=0.8, alpha=0.6)
    _wb = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
    ax.text(grid[-1], 0.53, "productive $\\geq$ 50% of max", ha="right", va="bottom",
            color="#5F5E5A", fontsize=9, bbox=_wb)
    ax.plot([lo_w], [0.5], "o", color=TEAL, ms=6)
    ax.annotate("productive shoulder\nright of the ceiling", xy=(lo_w, 0.5),
                xytext=(lo_w * 1.6, 0.74), fontsize=9.5, color="#2C2C2A", bbox=_wb,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7, ls=":"))
    ax.text(floor * 0.4, 0.30, "no productive\ntrimer in the\ndosable window", ha="right",
            color="#5F5E5A", fontsize=9, bbox=_wb)
    ax.set_xlabel("free drug concentration (nM, log)")
    ax.set_ylabel("trimer (normalised)")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper center", frameon=False, fontsize=9.5, ncol=2, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_candidate_panel(path, csv=None):
    csv = csv or os.path.normpath(os.path.join(HERE, "..", "..", "data", "candidates.csv"))
    df = screen_panel(csv).iloc[::-1].reset_index(drop=True)  # best margin at top

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i, r in df.iterrows():
        # corridor band (MABEL floor -> ceiling)
        ax.plot([r.mabel_floor, r.occupancy_ceiling], [i, i], color=GREY, lw=9, alpha=0.22,
                solid_capstyle="round")
        # productive shoulder
        go = bool(r.dosable)
        col = TEAL if (go and r.optimum_reachable) else (AMBER if go else CORAL)
        ax.plot([r.shoulder_lo, r.shoulder_hi], [i, i], color=col, lw=4, solid_capstyle="round")
        ax.plot([r.peak_dose], [i], "o", color=col, ms=7, zorder=5)
        # ceiling tick
        ax.plot([r.occupancy_ceiling, r.occupancy_ceiling], [i - 0.28, i + 0.28],
                color="#9A3520", lw=1.6)
        tag = "GO" if (go and r.optimum_reachable) else ("GO*" if go else "NO-GO")
        ax.text(1.02, i, f"{tag}  (margin {r.margin_log10:+.2f})",
                va="center", ha="left", fontsize=9, color=col,
                transform=ax.get_yaxis_transform())

    ax.set_xscale("log")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["name"])
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_xlim(1e-3, 1e2)
    ax.set_xlabel("free drug concentration (nM, log)")
    ax.set_title("FIH dosability screen — productive window vs occupancy ceiling",
                 fontsize=11.5, color="#2C2C2A", loc="left", pad=10)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color=GREY, lw=8, alpha=0.25, label="dosable corridor (MABEL\u2192ceiling)"),
        Line2D([0], [0], color=TEAL, lw=4, label="productive shoulder (\u226550% trimer)"),
        Line2D([0], [0], color="#9A3520", lw=1.6, label="occupancy ceiling"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.42, -0.13),
              ncol=3, frameon=False, fontsize=8.5)
    fig.subplots_adjust(right=0.74, bottom=0.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig_trimer_bell(os.path.join(FIGDIR, "trimer_bell.png"))
    fig_dosability_corridor(os.path.join(FIGDIR, "dosability_corridor.png"))
    fig_avidity_failure(os.path.join(FIGDIR, "avidity_failure.png"))
    fig_candidate_panel(os.path.join(FIGDIR, "candidate_panel.png"))
    print("wrote figures to", FIGDIR)


if __name__ == "__main__":
    main()
