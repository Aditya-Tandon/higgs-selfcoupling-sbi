"""
Validation for sbi/kl_reweight.py.

Reads a few SM HH->4b signal ROOT files, reconstructs gen (m_HH, cos_theta*),
and checks the reweighting reproduces the expected physics:

  1. sigma(kl)/sigma(SM) is a convex parabola with a minimum in kl ~ [2, 3]
     (destructive triangle/box interference).
  2. Effective sample size N_eff/N degrades for kl far from SM (single-sample
     reweighting limitation) -- reported, not asserted.

Light enough for a login node (a few 22 MB files, gen branches only).
"""
import glob
import sys

import numpy as np
import uproot

sys.path.insert(0, ".")
from sbi.kl_reweight import (
    gen_higgs_kinematics, cross_section_ratio, effective_nsample,
)

GEN_BRANCHES = ["GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass",
                "GenPart_pdgId", "GenPart_statusFlags"]


def load_m_hh(file_glob, max_files=4):
    files = sorted(glob.glob(file_glob))[:max_files]
    m_hh_all, cts_all = [], []
    for f in files:
        t = uproot.open(f)["Events"]
        a = t.arrays(GEN_BRANCHES)
        m_hh, cts = gen_higgs_kinematics(
            a["GenPart_pt"], a["GenPart_eta"], a["GenPart_phi"],
            a["GenPart_mass"], a["GenPart_pdgId"], a["GenPart_statusFlags"])
        m_hh_all.append(m_hh)
        cts_all.append(cts)
    m_hh = np.concatenate(m_hh_all)
    cts = np.concatenate(cts_all)
    good = np.isfinite(m_hh)
    return m_hh[good], cts[good]


def main():
    m_hh, cts = load_m_hh("data/hh4b/data_*.root", max_files=4)
    print(f"[validate] events with 2 last-copy Higgs: {len(m_hh)}")
    print(f"[validate] m_HH: min={m_hh.min():.1f} median={np.median(m_hh):.1f} "
          f"max={m_hh.max():.1f} GeV ; threshold 2*m_H = 250")
    print(f"[validate] cos(theta*): min={cts.min():.3f} max={cts.max():.3f}")

    kl_grid = np.array([-2, -1, 0, 1, 2, 2.45, 3, 4, 5, 10], dtype=float)
    ratio = cross_section_ratio(m_hh, kl_grid)
    print("\n[validate] sigma(kl)/sigma(SM):")
    for k, r in zip(kl_grid, np.atleast_1d(ratio)):
        print(f"    kl={k:6.2f}  ratio={r:8.3f}")

    # locate the parabola minimum on a fine grid
    fine = np.linspace(-2, 8, 1001)
    rfine = np.atleast_1d(cross_section_ratio(m_hh, fine))
    kl_min = fine[np.argmin(rfine)]
    print(f"\n[validate] sigma(kl) minimum at kl = {kl_min:.2f} "
          f"(expected ~2.0-3.0 for destructive interference)")

    print("\n[validate] effective sample size N_eff/N:")
    for k in [0.0, 1.0, 2.45, 5.0]:
        n_eff, frac = effective_nsample(m_hh, k)
        print(f"    kl={k:5.2f}  N_eff={n_eff:9.0f}  N_eff/N={frac:.3f}")

    ok = (1.5 < kl_min < 3.5) and np.atleast_1d(cross_section_ratio(m_hh, 1.0))[0] == 1.0
    print(f"\n[validate] GATE: parabola min in (1.5,3.5) and SM ratio==1 -> "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
