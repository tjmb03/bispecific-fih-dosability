"""The two load-bearing results of the ternary model, as executable assertions."""
import numpy as np
import pytest

from bsdose import trimer, peak_concentration, numerical_peak


@pytest.mark.parametrize("KA,KB", [(1.0, 1.0), (0.3, 0.3), (3.0, 3.0), (0.1, 10.0), (0.2, 5.0)])
def test_peak_at_geometric_mean(KA, KB):
    """Trimer peaks at sqrt(KA*KB) for symmetric AND asymmetric arm KDs.

    Targets held in strong excess so the target-excess analytical result holds.
    Only the product of the two KDs sets the peak location.
    """
    A_tot = B_tot = 500.0
    peak_dose, _ = numerical_peak(KA, KB, A_tot, B_tot)
    assert peak_dose == pytest.approx(peak_concentration(KA, KB), rel=0.03)


def test_asymmetric_same_product_same_peak():
    """Two arms split very differently but with the same product peak together."""
    A_tot = B_tot = 500.0
    p_sym, _ = numerical_peak(3.0, 3.0, A_tot, B_tot)     # sqrt(9) = 3
    p_asym, _ = numerical_peak(0.3, 30.0, A_tot, B_tot)   # sqrt(9) = 3
    assert p_asym == pytest.approx(p_sym, rel=0.03)


def test_peak_height_capped_by_limiting_target():
    """As the abundant target -> infinity, peak trimer -> the scarcer target."""
    A_tot = 1.0
    _, peak_small = numerical_peak(1.0, 1.0, A_tot, B_tot=1.0)
    _, peak_big = numerical_peak(1.0, 1.0, A_tot, B_tot=5000.0)
    assert peak_big > peak_small
    assert peak_big == pytest.approx(A_tot, rel=0.05)   # ceiling = limiting target


def test_bell_falls_off_both_sides():
    """The hallmark bell: trimer vanishes at very low and very high drug."""
    KA = KB = 1.0
    A_tot = B_tot = 50.0
    peak_dose, peak_val = numerical_peak(KA, KB, A_tot, B_tot)
    lo = trimer(peak_dose * 1e-4, KA, KB, A_tot, B_tot)
    hi = trimer(peak_dose * 1e4, KA, KB, A_tot, B_tot)
    assert lo < 0.05 * peak_val   # rising tail ~D
    assert hi < 0.05 * peak_val   # prozone tail ~1/D


def test_trimer_vectorised_matches_scalar():
    grid = np.array([0.1, 1.0, 10.0])
    vec = trimer(grid, 1.0, 1.0, 5.0, 0.5)
    scal = [trimer(float(d), 1.0, 1.0, 5.0, 0.5) for d in grid]
    assert np.allclose(vec, scal)
