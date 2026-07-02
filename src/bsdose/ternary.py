"""Ternary-complex equilibrium for a tumor-targeted bispecific.

A bispecific antibody ``drug`` bridges a tumor antigen ``A`` (e.g. MUC1) and a
receptor ``B`` (e.g. CD40) into a productive ternary complex (trimer)
``A - drug - B``. Only the trimer signals, so efficacy is a *bell-shaped*
function of drug concentration: too little drug gives no bridging, too much
drug saturates each arm with its own drug molecule (the prozone / hook effect)
and the trimer collapses.

At rapid binding equilibrium, with no cooperativity between the two arms
(alpha = 1),

    [A.drug.B] = [drug] * [A_free] * [B_free] / (KA * KB)

where ``[A_free]`` and ``[B_free]`` are the *free* target concentrations, which
deplete as drug binds. Target conservation gives a pair of coupled equations
solved here by fixed-point iteration.

Two results carry the whole model and are asserted in the test-suite:

1.  The trimer peaks at the **geometric mean of the two arm dissociation
    constants**, ``sqrt(KA * KB)`` -- exact in the target-excess limit and
    independent of how asymmetrically the affinity is split between the arms.
2.  The peak **height** is capped by the **limiting (scarcer) target**: raising
    the abundant target gives diminishing returns toward a ceiling set by the
    scarcer one.
"""
from __future__ import annotations

import numpy as np

__all__ = ["free_targets", "trimer", "peak_concentration", "numerical_peak"]


def free_targets(D, KA, KB, A_tot, B_tot, n_iter: int = 300):
    """Free target concentrations at equilibrium for drug concentration ``D``.

    Solves the coupled target-conservation equations

        A_free = A_tot / (1 + D/KA + D*B_free/(KA*KB))
        B_free = B_tot / (1 + D/KB + D*A_free/(KA*KB))

    by fixed-point iteration. ``D`` may be a scalar or a NumPy array; all
    concentrations share the same (molar) units.
    """
    D = np.asarray(D, dtype=float)
    A = np.full(D.shape, float(A_tot))
    B = np.full(D.shape, float(B_tot))
    for _ in range(n_iter):
        A = A_tot / (1.0 + D / KA + D * B / (KA * KB))
        B = B_tot / (1.0 + D / KB + D * A / (KA * KB))
    return A, B


def trimer(D, KA, KB, A_tot, B_tot, n_iter: int = 300):
    """Productive ternary-complex (trimer) concentration for drug ``D``."""
    A, B = free_targets(D, KA, KB, A_tot, B_tot, n_iter)
    T = np.asarray(D, dtype=float) * A * B / (KA * KB)
    return float(T) if T.ndim == 0 else T


def peak_concentration(KA, KB) -> float:
    """Drug concentration that maximises the trimer: ``sqrt(KA * KB)``.

    This is the analytical optimum in the target-excess limit. Maximising
    ``D / ((1 + D/KA)(1 + D/KB))`` gives ``D^2 = KA*KB`` exactly, regardless of
    the KA/KB asymmetry -- only the *product* of the two arm KDs sets the peak
    location.
    """
    return float(np.sqrt(KA * KB))


def numerical_peak(KA, KB, A_tot, B_tot, grid=None):
    """Depletion-aware numerical peak ``(dose, trimer)`` over a dose grid."""
    if grid is None:
        centre = peak_concentration(KA, KB)
        grid = np.logspace(np.log10(centre) - 3.0, np.log10(centre) + 3.0, 6000)
    T = trimer(grid, KA, KB, A_tot, B_tot)
    i = int(np.argmax(T))
    return float(grid[i]), float(T[i])
