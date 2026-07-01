"""
Discrimination bottleneck diagnostic (HH->4b vs QCD, L1 event level).

Uses the NSBI cache (event ParT score + reco m_HH + QCD sigma) to measure the *physics-
weighted* discrimination and decompose where the background sits. Answers: is signal/
background discrimination the bottleneck for kappa_lambda sensitivity, and what would move it?

QCD yields use sigma_bin * 1000 * L / n_loaded_this_bin (F1 fix: the loaded subset represents
the full bin cross-section), so B and the background shape are correct.
"""
import argparse
import json
import sys
import numpy as np

sys.path.insert(0, ".")

L_FB = 1000.0


def load(cache="data/event_level/nsbi_cache.npz", cfg="hh-bbbb-obj-config.json", n_loaded=None):
    """n_loaded: per-bin loaded count. If None, inferred from the cache (correct only when the
    cache stores ALL loaded QCD events, i.e. no preselection). For a preselected cache that stores
    only survivors, pass --n-loaded = the original --qcd-max-per-bin so the selection efficiency
    is preserved (yield = sigma*1000*L/n_loaded, NOT /n_survived)."""
    d = np.load(cache, allow_pickle=True)
    c = json.load(open(cfg))
    sig_w = c["physics"]["signal_xsec_pb"] * 1000.0 * L_FB / c["physics"]["n_gen_signal"]
    qsig = d["qcd_sigma"].astype(np.float64)
    qyield = np.empty_like(qsig)
    for s in np.unique(qsig):
        m = qsig == s
        denom = float(m.sum()) if n_loaded is None else float(n_loaded)
        qyield[m] = s * 1000.0 * L_FB / denom
    return d, sig_w, qyield, c


def sig_bkg_at(thr, ss, sw, qs, qy, s_extra=None, q_extra=None):
    sm = ss >= thr
    qm = qs >= thr
    if s_extra is not None:
        sm &= s_extra
        qm &= q_extra
    S = sw * sm.sum()
    B = qy[qm].sum()
    return S, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/event_level/nsbi_cache.npz")
    ap.add_argument("--n-loaded", type=float, default=None,
                    help="per-bin loaded count (pass qcd-max-per-bin for a preselected cache)")
    args = ap.parse_args()
    d, sig_w, qyield, cfg = load(cache=args.cache, n_loaded=args.n_loaded)
    ss, sm_reco = d["sig_score"], np.isfinite(d["sig_reco_mhh"])
    qs, qm_reco = d["qcd_score"], np.isfinite(d["qcd_reco_mhh"])
    s_mhh, q_mhh = d["sig_reco_mhh"], d["qcd_reco_mhh"]
    qsig = d["qcd_sigma"].astype(np.float64)

    print(f"signal events {len(ss)} (reco {sm_reco.sum()}); QCD {len(qs)} (reco {qm_reco.sum()})")
    S_tot = sig_w * len(ss)
    B_tot = qyield.sum()
    print(f"total yields @ {L_FB:.0f}/fb: S={S_tot:.1f}  B={B_tot:.3e}  S/B={S_tot/B_tot:.2e}")
    print(f"baseline S/sqrt(B) (no cut): {S_tot/np.sqrt(B_tot):.4f}")

    # --- ROC in physics terms + S/sqrt(B) scan over score threshold ---
    print("\n=== score-threshold scan (S/sqrt(B)) ===")
    best = (0, 0, 0, 0, 0)
    for thr in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]:
        S, B = sig_bkg_at(thr, ss, sig_w, qs, qyield)
        eps_s = (ss >= thr).mean()
        rej = len(qs) / max((qs >= thr).sum(), 1)
        z = S / np.sqrt(B) if B > 0 else 0
        print(f"  score>={thr:6.3f}: eps_s={eps_s:5.3f} QCD_rej={rej:8.1f}x  S={S:7.1f} B={B:.3e} S/sqrt(B)={z:.4f}")
        if z > best[0]:
            best = (z, thr, eps_s, S, B)
    print(f"  BEST score-only: S/sqrt(B)={best[0]:.4f} @ score>={best[1]:.3f}")

    # --- add di-Higgs mass window on top of a tight score cut ---
    print("\n=== + di-Higgs mass window (score>=0.9, reco required) ===")
    for win in [(250, 550), (300, 500), (330, 450), (350, 420)]:
        s_extra = sm_reco & (ss >= 0.9) & (s_mhh >= win[0]) & (s_mhh <= win[1])
        q_extra = qm_reco & (qs >= 0.9) & (q_mhh >= win[0]) & (q_mhh <= win[1])
        S = sig_w * s_extra.sum()
        B = qyield[q_extra].sum()
        z = S / np.sqrt(B) if B > 0 else 0
        print(f"  mHH in {win}: S={S:6.1f} B={B:.3e} S/sqrt(B)={z:.4f}")

    # --- decompose B by QCD pt bin at score>=0.9 ---
    print("\n=== B decomposition by QCD pt-bin (score>=0.9) ===")
    qm = qs >= 0.9
    s2name = {float(v["weight"]): k for k, v in cfg["QCD_background"].items()}
    for s in sorted(np.unique(qsig), reverse=True):
        binm = (qsig == s) & qm
        Bb = qyield[binm].sum()
        name = s2name.get(float(s), s2name.get(min(s2name, key=lambda x: abs(x - s)), "?"))
        print(f"  {name:16s}: kept {binm.sum():5d}  B={Bb:.3e} ({100*Bb/max(qyield[qm].sum(),1e-30):4.1f}%)")

    # --- score separation summary ---
    print("\n=== score distribution (signal vs QCD) ===")
    for q in [0.5, 0.9, 0.99, 0.999]:
        print(f"  QCD score {q*100:.1f}%-ile = {np.quantile(qs,q):.4f} ; "
              f"signal frac above = {(ss>np.quantile(qs,q)).mean():.3f}")


if __name__ == "__main__":
    main()
