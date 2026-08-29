"""
Analytic LO gg -> HH reweighting in the Higgs trilinear coupling kappa_lambda (kl).

The single SM (kl = 1) HH->4b sample is turned into a continuous simulator family
p(x | kl) by per-event matrix-element reweighting. At leading order the gg->HH
amplitude is the coherent sum of a *triangle* diagram (one top Yukawa kappa_t and
the trilinear coupling kl) and a *box* diagram (two top Yukawas):

    M(kl, kt) = kt * kl * C_tri * F_tri  +  kt^2 * C_box * F_box        (+ G_box, spin-2 piece)

so |M|^2 is quadratic in kl:

    |M(kl)|^2 = a + b * kl + c * kl^2

with a, b, c functions of the gen-level kinematics (m_HH and the scattering angle
cos(theta*) in the di-Higgs rest frame). The per-event reweight to a target kl is

    w_i(kl) = |M(kl)|^2 / |M(kl_ref=1)|^2 .

Two matrix-element models are provided:

* ``ME_HEFT``  -- heavy-top (infinite m_t) effective theory. Form factors collapse to
  constants (F_tri = 2/3, F_box = -2/3, G_box = 0), giving an m_HH-only reweight.
  Exact in the EFT limit, fast, no loop integrals. Known artefact: the SM amplitude
  vanishes at the 2*m_H threshold (triangle/box cancellation), so weights for kl != 1
  diverge as m_HH -> 2 m_H. We floor the denominator and expose the effective sample
  size so this is visible, not silent.

* finite-m_t LO -- TODO (drop-in via the same a/b/c interface); removes the threshold
  artefact and adds genuine cos(theta*) dependence.

This replaces the crude m_HH-only ``(1 + kl*r)^2`` form used on the stale ``sbi-test``
branch, while keeping the same event-level, jet-free spirit.
"""

from __future__ import annotations

import numpy as np

# Physical constants (GeV)
M_H = 125.0          # Higgs mass
M_H2 = M_H * M_H

# HEFT form-factor constants (heavy-top limit)
_F_TRI_HEFT = 2.0 / 3.0
_F_BOX_HEFT = -2.0 / 3.0

KL_SM = 1.0          # Standard Model reference value of kappa_lambda


def _triangle_propagator(m_hh):
    """R(m_HH) = 3 m_H^2 / (m_HH^2 - m_H^2): the s-channel Higgs propagator times the
    HHH vertex, i.e. the relative weight of the triangle vs box diagram in HEFT."""
    m_hh = np.asarray(m_hh, dtype=np.float64)
    return 3.0 * M_H2 / (m_hh * m_hh - M_H2)


def me2_coeffs_heft(m_hh, kappa_t=1.0):
    """Return (a, b, c) such that |M|^2 = a + b*kl + c*kl^2 in the HEFT limit.

    |M|^2 ∝ (kt * kl * R * F_tri + kt^2 * F_box)^2
          = (kt^2 F_box)^2 + 2 (kt^2 F_box)(kt R F_tri) kl + (kt R F_tri)^2 kl^2
    Overall positive constants cancel in the reweight ratio but are kept for clarity.
    """
    R = _triangle_propagator(m_hh)
    box = kappa_t ** 2 * _F_BOX_HEFT
    tri = kappa_t * R * _F_TRI_HEFT
    a = box * box
    b = 2.0 * box * tri
    c = tri * tri
    return a, b, c


def me2_heft(m_hh, kappa_lambda, kappa_t=1.0):
    """|M|^2 (up to a global constant) at the given kappa_lambda, HEFT limit."""
    a, b, c = me2_coeffs_heft(m_hh, kappa_t=kappa_t)
    kl = np.asarray(kappa_lambda, dtype=np.float64)
    return a + b * kl + c * kl * kl


def reweight_heft(m_hh, kappa_lambda, kappa_lambda_ref=KL_SM, kappa_t=1.0,
                  denom_floor=1e-6):
    """Per-event weight w_i(kl) = |M(kl)|^2 / |M(kl_ref)|^2 (HEFT).

    Parameters
    ----------
    m_hh : array-like
        Gen-level di-Higgs invariant mass per event (GeV).
    kappa_lambda : float or array-like
        Target kl. If array, must broadcast against m_hh.
    denom_floor : float
        Floor on the SM |M|^2 to tame the HEFT threshold cancellation. The
        fraction of events hitting the floor is reported by ``effective_nsample``.
    """
    num = me2_heft(m_hh, kappa_lambda, kappa_t=kappa_t)
    den = me2_heft(m_hh, kappa_lambda_ref, kappa_t=kappa_t)
    den = np.maximum(den, denom_floor * np.max(den) if np.ndim(den) else den)
    return num / den


def cross_section_ratio(m_hh, kappa_lambda, sm_weights=None, kappa_t=1.0):
    """sigma(kl) / sigma(SM) estimated from the SM sample by summing reweights.

    This is the canonical validation observable: it must be a convex parabola in kl
    with a minimum in kl ~ [2, 3] (destructive triangle/box interference).
    """
    kl = np.atleast_1d(np.asarray(kappa_lambda, dtype=np.float64))
    m_hh = np.asarray(m_hh, dtype=np.float64)
    w0 = np.ones_like(m_hh) if sm_weights is None else np.asarray(sm_weights, np.float64)
    a, b, c = me2_coeffs_heft(m_hh, kappa_t=kappa_t)
    den = a + b * KL_SM + c * KL_SM * KL_SM
    den = np.maximum(den, 1e-12)
    out = np.empty_like(kl)
    for i, k in enumerate(kl):
        num = a + b * k + c * k * k
        out[i] = np.sum(w0 * num / den) / np.sum(w0)
    return out if out.size > 1 else float(out[0])


def effective_nsample(m_hh, kappa_lambda, sm_weights=None, kappa_t=1.0):
    """Kish effective sample size of the reweighted SM sample at a target kl.

    N_eff = (sum w)^2 / sum(w^2). A small N_eff/N flags that reweighting one SM
    sample to this kl is statistically unreliable (the headline limitation of
    single-sample reweighting, worst near threshold and for kl far from SM)."""
    w_rel = reweight_heft(m_hh, kappa_lambda, kappa_t=kappa_t)
    w0 = np.ones_like(np.asarray(m_hh, np.float64)) if sm_weights is None \
        else np.asarray(sm_weights, np.float64)
    w = w0 * w_rel
    s1 = np.sum(w)
    s2 = np.sum(w * w)
    n_eff = (s1 * s1) / s2 if s2 > 0 else 0.0
    return float(n_eff), float(n_eff / len(w))


# --------------------------------------------------------------------------- #
# Gen-level di-Higgs kinematics from a NanoAOD-style GenPart record
# --------------------------------------------------------------------------- #
def gen_higgs_kinematics(gen_pt, gen_eta, gen_phi, gen_mass, gen_pdgid,
                         gen_statusflags):
    """Reconstruct (m_HH, cos_theta_star) per event from the two last-copy Higgs.

    Inputs are awkward arrays (one jagged list per event) as read from uproot.
    Returns numpy arrays (m_hh, cos_theta_star) of length n_events; events without
    exactly two last-copy Higgs are returned as NaN and should be masked out.

    cos(theta*) is |cos| of the Higgs polar angle in the di-Higgs rest frame
    (Collins-Soper-like, boosting only along z), the standard HH variable.
    """
    import awkward as ak
    import vector
    vector.register_awkward()

    is_h = gen_pdgid == 25
    is_last = (gen_statusflags & (1 << 13)) != 0
    sel = is_h & is_last

    h = ak.zip({"pt": gen_pt[sel], "eta": gen_eta[sel],
                "phi": gen_phi[sel], "mass": gen_mass[sel]},
               with_name="Momentum4D")

    n = ak.num(h)
    ok = n == 2
    h2 = h[ok]
    h1 = h2[:, 0]
    h2b = h2[:, 1]
    hh = h1 + h2b
    m_hh_ok = np.asarray(hh.mass)

    # cos(theta*): boost H1 into the HH rest frame, take |cos| wrt beam (z) axis
    h1_rest = h1.boostCM_of(hh)
    cts_ok = np.abs(np.asarray(h1_rest.pz) / np.asarray(h1_rest.p))

    m_hh = np.full(len(n), np.nan)
    cts = np.full(len(n), np.nan)
    idx = np.asarray(ok)
    m_hh[idx] = m_hh_ok
    cts[idx] = cts_ok
    return m_hh, cts
