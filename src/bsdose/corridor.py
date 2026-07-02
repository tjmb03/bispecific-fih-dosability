"""Systemic-occupancy dosing corridor for an immune-agonist bispecific.

Efficacy happens at the tumor, where both targets co-localise and the trimer is
avidity-stabilised. The *safety* readout is different: systemic engagement of
the agonist receptor (CD40) is **monovalent** -- one drug arm binding CD40 on
circulating / peripheral cells, with no second target to bridge, so no trimer
and no avidity benefit. Cytokine-release risk tracks that systemic CD40
occupancy, which is a plain one-site binding curve in free drug.

That gives a first-in-human **dosable corridor** bounded by two occupancy
thresholds on the *systemic monovalent* CD40 KD:

* the **MABEL floor** -- the minimal-anticipated-pharmacology start dose
  (default: 10% systemic CD40 occupancy);
* the **occupancy ceiling** -- the exposure above which CRS risk is
  unacceptable (default: 25% systemic CD40 occupancy).

Both bounds move with the *monovalent* CD40 arm affinity, not the
avidity-enhanced tumor affinity -- the distinction that makes the screen work.
"""
from __future__ import annotations

import numpy as np

__all__ = ["occupancy", "dose_for_occupancy", "corridor"]


def occupancy(D, KD):
    """Monovalent one-site receptor occupancy, ``D / (D + KD)``."""
    D = np.asarray(D, dtype=float)
    occ = D / (D + KD)
    return float(occ) if occ.ndim == 0 else occ


def dose_for_occupancy(occ: float, KD: float) -> float:
    """Inverse of :func:`occupancy`: free drug giving fractional occupancy ``occ``."""
    if not 0.0 < occ < 1.0:
        raise ValueError("occupancy must be in (0, 1)")
    return float(KD * occ / (1.0 - occ))


def corridor(KD_sys: float, floor_occ: float = 0.10, ceil_occ: float = 0.25):
    """(floor, ceiling) free-drug bounds from the systemic monovalent CD40 KD."""
    if not floor_occ < ceil_occ:
        raise ValueError("floor_occ must be below ceil_occ")
    return dose_for_occupancy(floor_occ, KD_sys), dose_for_occupancy(ceil_occ, KD_sys)
