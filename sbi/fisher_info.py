"""
Shared per-event κλ Fisher-information helpers (HEFT LO reweight).

Used by sbi/info_acceptance_map.py (Dir 1) and
sbi/dir8_offline_tagger_control.py (Dir 8 control). All quantities derive from
the analytic |M(κλ)|² = a + b·κλ + c·κλ² of sbi/kl_reweight.py.

Decomposition of the extended-likelihood information (the "correct" split from
the Dir-1 plan note):

  shape term  = S(κλ) · Var_w[t]  ∝  Σ_sel w_i (t_i − t̄_w)²   (t̄_w subtracted!)
  yield term  = (∂_κλ S)² / S     ∝  (Σ_sel w_i t_i)² / Σ_sel w_i

with t_i(κλ) = ∂_κλ ln w_i(κλ) = (b + 2cκλ)/(a + bκλ + cκλ²) the per-event
score and w_i(κλ) the HEFT reweight. Because both terms share the same sig_w
prefactor, information *ratios* R between selection stages are prefactor-free.

HEFT caveat: |M|² vanishes at the triangle/box cancellation (m_HH → 2m_H for
the SM denominator; a κλ-dependent point for the numerator), so t diverges
there. We floor the denominator like reweight_heft and expose the Kish N_eff.
"""
from __future__ import annotations

import numpy as np

from sbi.kl_reweight import me2_coeffs_heft, reweight_heft


def kish_n_eff(w):
    """Kish effective sample size (Σw)²/Σw² of an arbitrary weight vector."""
    w = np.asarray(w, np.float64)
    s2 = np.sum(w * w)
    return float(np.sum(w) ** 2 / s2) if s2 > 0 else 0.0


def event_score_t(m_hh, kl, denom_floor=1e-6, clip=None):
    """Per-event score t_i(κλ) = ∂_κλ ln w_i = (b + 2cκλ)/(a + bκλ + cκλ²).

    The denominator is floored at denom_floor × its max (same convention as
    reweight_heft) to tame the HEFT cancellation; pass clip to hard-clip |t|.
    """
    a, b, c = me2_coeffs_heft(np.asarray(m_hh, np.float64))
    den = a + b * kl + c * kl * kl
    den = np.maximum(den, denom_floor * np.nanmax(den))
    t = (b + 2.0 * c * kl) / den
    if clip is not None:
        t = np.clip(t, -clip, clip)
    return t


def event_weight_w(m_hh, kl):
    """Per-event HEFT reweight w_i(κλ) (SM reference, floored denominator)."""
    return reweight_heft(np.asarray(m_hh, np.float64), kl)


def shape_info_sum(w, t, sel=None):
    """Σ_sel w (t − t̄_w)² — the prefactor-free shape-information sum.

    t̄_w is the *weighted mean over the selected set* (each stage carries its
    own mean; the raw Σ w t² leaks yield information into the shape term).
    Returns 0.0 for an empty/zero-weight selection.
    """
    if sel is not None:
        w, t = w[sel], t[sel]
    sw = np.sum(w)
    if sw <= 0:
        return 0.0
    tbar = np.sum(w * t) / sw
    return float(np.sum(w * (t - tbar) ** 2))


def yield_info_sum(w, t, sel=None):
    """(Σ_sel w t)² / Σ_sel w — the prefactor-free yield-information sum."""
    if sel is not None:
        w, t = w[sel], t[sel]
    sw = np.sum(w)
    return float(np.sum(w * t) ** 2 / sw) if sw > 0 else 0.0


def ullrich_xu_efficiency(k, n):
    """Bayesian binomial efficiency (Ullrich–Xu, physics/0701199): posterior
    mean (k+1)/(n+2) and standard deviation, elementwise on arrays."""
    k = np.asarray(k, np.float64)
    n = np.asarray(n, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (k + 1.0) / (n + 2.0)
        var = (k + 1.0) * (k + 2.0) / ((n + 2.0) * (n + 3.0)) - mean**2
    mean = np.where(n > 0, mean, np.nan)
    err = np.where(n > 0, np.sqrt(np.maximum(var, 0.0)), np.nan)
    return mean, err
