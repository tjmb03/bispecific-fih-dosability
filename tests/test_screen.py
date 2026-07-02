"""The decision rule and corridor, as executable assertions."""
import numpy as np
import pytest

from bsdose import occupancy, corridor, dose_for_occupancy, screen_candidate


def test_occupancy_monotonic():
    """Systemic occupancy is strictly increasing, so the corridor is well defined."""
    grid = np.logspace(-3, 3, 500)
    occ = occupancy(grid, KD=1.0)
    assert np.all(np.diff(occ) > 0)
    assert occ[0] < 0.01 and occ[-1] > 0.99


def test_corridor_from_occupancy_thresholds():
    floor, ceil = corridor(KD_sys=1.0, floor_occ=0.10, ceil_occ=0.25)
    assert floor == pytest.approx(dose_for_occupancy(0.10, 1.0))
    assert ceil == pytest.approx(dose_for_occupancy(0.25, 1.0))
    assert floor < ceil
    # KD_sys = 1 nM -> 10% at 0.111 nM, 25% at 0.333 nM
    assert floor == pytest.approx(0.1111, rel=1e-3)
    assert ceil == pytest.approx(0.3333, rel=1e-3)


def test_strong_avidity_is_go():
    """Avidity-engineered lead: optimum sits inside the corridor."""
    r = screen_candidate("lead", KA=16.0, KB=1.0, avidity=21.0, MUC1=5.0, CD40=0.5)
    assert r.dosable is True
    assert r.optimum_reachable is True
    assert r.margin_log10 > 0


def test_avidity_loss_is_no_go():
    """Same arms, avidity removed: productive window slides right of the ceiling."""
    r = screen_candidate("monovalent", KA=16.0, KB=1.0, avidity=1.0, MUC1=5.0, CD40=0.5)
    assert r.dosable is False
    assert r.margin_log10 < 0
    assert r.shoulder_lo > r.occupancy_ceiling


def test_margin_sign_matches_dosable():
    """dosable <=> margin >= 0 across a parameter sweep."""
    for av in [1.0, 2.0, 5.0, 10.0, 21.0]:
        r = screen_candidate("t", KA=16.0, KB=1.0, avidity=av, MUC1=5.0, CD40=0.5)
        assert r.dosable == (r.margin_log10 >= 0)


def test_density_sets_efficacy_not_dosability():
    """Lowering the tumor CD40 density cuts the trimer height, not the verdict."""
    hi = screen_candidate("hi", KA=16.0, KB=1.0, avidity=21.0, MUC1=5.0, CD40=0.5)
    lo = screen_candidate("lo", KA=16.0, KB=1.0, avidity=21.0, MUC1=5.0, CD40=0.05)
    assert hi.dosable and lo.dosable
    assert lo.peak_trimer < 0.5 * hi.peak_trimer     # much weaker efficacy
    assert lo.peak_dose == pytest.approx(hi.peak_dose, rel=0.05)  # same optimum dose
