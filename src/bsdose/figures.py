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


def fig_kd_position(path):
    """Result 1: the two arm KDs set the peak POSITION at sqrt(KA*KB).

    Asymmetric arms (0.3 / 30) peak at the same place as symmetric (3 / 3) --
    only the product of the KDs matters, not how the affinity is split.
    """
    grid = np.logspace(-2, 2, 4000)
    A = B = 500.0  # targets in strong excess so peaks sit exactly at sqrt(KA*KB)
    curves = [
        (0.3, 0.3, TEAL, "-", "KA=KB=0.3"),
        (3.0, 3.0, PURPLE, "-", "KA=KB=3"),
        (0.3, 30.0, CORAL, "--", "KA=0.3, KB=30"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for KA, KB, col, ls, lab in curves:
        T = trimer(grid, KA, KB, A, B)
        ax.semilogx(grid, T / T.max(), color=col, lw=2.6, ls=ls, label=lab)
    for pk, col, tcol in [(0.3, TEAL, "#0C5E47"), (3.0, PURPLE, "#4A3789")]:
        ax.axvline(pk, color=col, ls="--", lw=0.9, alpha=0.6)
        ax.plot([pk], [1.0], "o", color=col, ms=6)
        ax.text(pk, 1.05, r"$\sqrt{K_A K_B}=%g$" % pk, ha="center", fontsize=9, color=tcol)
    ax.annotate("purple & red peak together\n(same geometric mean)", xy=(3.0, 0.86),
                xytext=(9, 0.55), fontsize=9, color="#2C2C2A",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7, ls=":"))
    ax.set_xlabel("free drug concentration (nM, log)")
    ax.set_ylabel("trimer (normalised)")
    ax.set_ylim(0, 1.14)
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.17))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_density_height(path):
    """Result 2: target density sets peak HEIGHT, capped by the scarcer target.

    KA=KB=1 (peak pinned at 1); MUC1 held fixed while CD40 density is swept.
    Height climbs toward the MUC1 ceiling with diminishing returns; the peak
    never moves.
    """
    grid = np.logspace(-2, 2, 4000)
    KA = KB = 1.0
    MUC1 = 1.0  # fixed scarce target => the ceiling
    densities = [0.5, 2, 8, 32, 128]
    shades = ["#A9E0CE", "#6FC9AC", "#37AC85", "#1B8A63", "#0A5F42"]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for cd, col in zip(densities, shades):
        T = trimer(grid, KA, KB, MUC1, cd)
        ax.semilogx(grid, T, color=col, lw=2.4, label="\u00d7%g" % cd)
    ax.axhline(MUC1, color=CORAL, ls="--", lw=1.0, alpha=0.7)
    ax.text(grid[0] * 1.3, MUC1 * 1.015, "MUC1 ceiling \u2014 the scarcer target caps the trimer",
            color="#9A3520", fontsize=9)
    ax.axvline(1.0, color=PURPLE, ls="--", lw=0.9, alpha=0.5)
    ax.text(1.0, 0.55, "peak pinned at\n$\\sqrt{K_A K_B}=1$\n(KDs, not density)", ha="center",
            fontsize=9, color="#4A3789",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.text(30, 0.86, "more CD40 \u2192\ndiminishing returns", fontsize=9, color="#5F5E5A")
    ax.text(30, 0.10, "scarce CD40 \u2192\nlittle trimer", fontsize=9, color="#5F5E5A")
    ax.set_xlabel("free drug concentration (nM, log)")
    ax.set_ylabel("trimer (abs.)")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper center", ncol=5, frameon=False, fontsize=9,
              title="CD40 density (MUC1 held fixed)", title_fontsize=9.5,
              bbox_to_anchor=(0.5, 1.19))
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
    fig_kd_position(os.path.join(FIGDIR, "kd_position.png"))
    fig_density_height(os.path.join(FIGDIR, "density_height.png"))
    fig_dosability_corridor(os.path.join(FIGDIR, "dosability_corridor.png"))
    fig_avidity_failure(os.path.join(FIGDIR, "avidity_failure.png"))
    fig_candidate_panel(os.path.join(FIGDIR, "candidate_panel.png"))
    print("wrote figures to", FIGDIR)


if __name__ == "__main__":
    main()
