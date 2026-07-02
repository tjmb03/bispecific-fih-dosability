"""First-in-human dosability screen for a tumor-targeted bispecific.

Decision rule
-------------
A candidate is **dosable** iff a free-drug concentration exists that is at once

* **productive** -- tumor trimer at least ``shoulder_frac`` of its maximum
  (default 50%), i.e. drug above the rising-shoulder onset ``shoulder_lo``; and
* **safe** -- systemic CD40 occupancy at or below the ceiling, i.e. drug at or
  below ``occupancy_ceiling``.

Both conditions can be met simultaneously exactly when the productive-shoulder
onset sits left of the occupancy ceiling::

    dosable  <=>  shoulder_lo <= occupancy_ceiling

The **margin** ``log10(ceiling / shoulder_lo)`` quantifies how much room there
is: positive = dosable with headroom, negative = the productive window has slid
right of the ceiling and no safe dose is efficacious.

Avidity model
-------------
Ternary-complex avidity (the slowed apparent off-rate of the assembled trimer)
is represented as a fold-tightening of each *effective tumor-arm* KD::

    KA_eff = KA / avidity,   KB_eff = KB / avidity

so the tumor optimum sits at ``sqrt(KA*KB) / avidity`` -- higher avidity pulls
the bell left. The systemic corridor is computed from the **monovalent** CD40
KD (``KD_sys``, default ``KB``) and gets *no* avidity benefit.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .ternary import trimer, peak_concentration
from .corridor import corridor

__all__ = ["ScreenResult", "screen_candidate", "screen_panel"]


@dataclass
class ScreenResult:
    name: str
    KA: float
    KB: float
    avidity: float
    MUC1: float
    CD40: float
    peak_dose: float          # free drug at the tumor-trimer optimum (nM)
    peak_trimer: float        # trimer at the optimum (nM) -- efficacy magnitude
    shoulder_lo: float        # rising-shoulder onset, >= 50% of max (nM)
    shoulder_hi: float        # falling-shoulder offset (nM)
    mabel_floor: float        # systemic 10% occupancy dose (nM)
    occupancy_ceiling: float  # systemic 25% occupancy dose (nM)
    dosable: bool
    margin_log10: float
    optimum_reachable: bool   # can the optimum itself be dosed below the ceiling
    verdict: str

    def as_row(self) -> dict:
        return asdict(self)


def _verdict(dosable: bool, optimum_reachable: bool, margin: float) -> str:
    if not dosable:
        return "NO-GO: productive window right of ceiling (avidity/CD40-arm)"
    if not optimum_reachable:
        return "GO: dose to shoulder (optimum sits above ceiling)"
    if margin < 0.3:
        return "GO: thin margin"
    return "GO: optimum reachable, comfortable margin"


def screen_candidate(
    name: str,
    KA: float,
    KB: float,
    avidity: float = 1.0,
    MUC1: float = 5.0,
    CD40: float = 0.5,
    floor_occ: float = 0.10,
    ceil_occ: float = 0.25,
    KD_sys: float | None = None,
    shoulder_frac: float = 0.5,
    grid: np.ndarray | None = None,
) -> ScreenResult:
    """Screen one candidate; see module docstring for the decision rule."""
    KA_eff, KB_eff = KA / avidity, KB / avidity

    if grid is None:
        centre = peak_concentration(KA_eff, KB_eff)
        grid = np.logspace(np.log10(centre) - 3.5, np.log10(centre) + 3.5, 8000)

    T = trimer(grid, KA_eff, KB_eff, MUC1, CD40)
    i = int(np.argmax(T))
    peak_dose, peak_trimer = float(grid[i]), float(T[i])

    mask = T >= shoulder_frac * peak_trimer
    shoulder_lo = float(grid[mask][0])
    shoulder_hi = float(grid[mask][-1])

    kd_sys = KB if KD_sys is None else KD_sys
    floor, ceiling = corridor(kd_sys, floor_occ, ceil_occ)

    dosable = shoulder_lo <= ceiling
    margin = float(np.log10(ceiling / shoulder_lo))
    optimum_reachable = peak_dose <= ceiling

    return ScreenResult(
        name=name, KA=KA, KB=KB, avidity=avidity, MUC1=MUC1, CD40=CD40,
        peak_dose=peak_dose, peak_trimer=peak_trimer,
        shoulder_lo=shoulder_lo, shoulder_hi=shoulder_hi,
        mabel_floor=floor, occupancy_ceiling=ceiling,
        dosable=dosable, margin_log10=margin,
        optimum_reachable=optimum_reachable,
        verdict=_verdict(dosable, optimum_reachable, margin),
    )


def screen_panel(candidates, **kw) -> pd.DataFrame:
    """Screen a panel of candidates, ranked best-margin first.

    ``candidates`` is a path to a CSV (columns
    ``name,KA_nM,KB_nM,avidity,MUC1_nM,CD40_nM[,note]``) or a DataFrame with
    those columns. Extra keyword args pass through to :func:`screen_candidate`.
    """
    if isinstance(candidates, (str, bytes)) or hasattr(candidates, "__fspath__"):
        df = pd.read_csv(candidates)
    else:
        df = pd.DataFrame(candidates)

    results = [
        screen_candidate(
            name=r["name"], KA=r["KA_nM"], KB=r["KB_nM"], avidity=r["avidity"],
            MUC1=r["MUC1_nM"], CD40=r["CD40_nM"], **kw,
        )
        for _, r in df.iterrows()
    ]
    out = pd.DataFrame([r.as_row() for r in results])
    if "note" in df.columns:
        out = out.merge(df[["name", "note"]], on="name", how="left")
    return out.sort_values("margin_log10", ascending=False).reset_index(drop=True)
