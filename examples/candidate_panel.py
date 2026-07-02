"""Screen the candidate panel and print the ranked go/no-go table.

    python examples/candidate_panel.py

Also regenerates figures/candidate_panel.png.
"""
import os

import pandas as pd

from bsdose import screen_panel
from bsdose.figures import fig_candidate_panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "candidates.csv")


def main():
    df = screen_panel(CSV)
    show = df[[
        "name", "peak_dose", "peak_trimer", "shoulder_lo",
        "mabel_floor", "occupancy_ceiling", "dosable",
        "margin_log10", "optimum_reachable", "verdict",
    ]].copy()
    for c in ["peak_dose", "peak_trimer", "shoulder_lo", "mabel_floor",
              "occupancy_ceiling", "margin_log10"]:
        show[c] = show[c].round(3)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print("\nFIH dosability screen — ranked best-margin first\n")
    print(show.to_string(index=False))

    out = os.path.join(ROOT, "figures", "candidate_panel.png")
    fig_candidate_panel(out, csv=CSV)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
