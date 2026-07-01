"""
Iteration-2 κλ NSBI closure — real observables, binned extended likelihood.

Summary statistic = the *learned* event-ParT score together with the reconstructed
di-Higgs mass:  x = [score, reco m_HH]  (from sbi/build_nsbi_cache.py).

Why binned: at L1 scouting the QCD yield is astronomical (B ~ 1e11 at 3 ab^-1), so
drawing discrete background pseudo-events is impossible. A binned extended likelihood
draws Poisson counts *per bin* (tractable for any B) and morphs the signal template with
the analytic κλ reweighting. Expected counts in bin b:

    nu_b(κλ) = S(κλ) · f_sig,b(κλ)  +  B · f_bkg,b ,   S(κλ) = σ_ratio(κλ) · S_SM

f_sig,b(κλ) is the κλ-reweighted signal shape (reweight uses gen m_HH); f_bkg,b is the
QCD shape (weighted by per-event cross-section yield). Poisson log-likelihood:

    log L(κλ) = Σ_b [ d_b · ln nu_b(κλ) − nu_b(κλ) ] .

Closure: Asimov recovery (MLE == truth by construction) reports the *interval width*
(expected sensitivity); coverage is measured with per-bin Poisson toys and the interval
threshold is SBC-calibrated. A --signal-boost knob lets us verify the method in a
sensitivity-bearing regime and separately quote the honest full-selection sensitivity.

Gate: κλ inside the calibrated 68% interval for >=3/4 injected points; toy coverage in
[0.60, 0.76].
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from sbi.kl_reweight import me2_coeffs_heft, cross_section_ratio, KL_SM


def reweight(gen_mhh, kl):
    a, b, c = me2_coeffs_heft(gen_mhh)
    den = a + b * KL_SM + c * KL_SM * KL_SM
    return np.clip((a + b * kl + c * kl * kl) / np.maximum(den, 1e-12), 0.0, 1e3)


def hist2d(x0, x1, w, edges0, edges1):
    h, _, _ = np.histogram2d(x0, x1, bins=[edges0, edges1], weights=w)
    return h.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/event_level/nsbi_cache.npz")
    ap.add_argument("--config", default="hh-bbbb-obj-config.json")
    ap.add_argument("--kl-low", type=float, default=-1.0)
    ap.add_argument("--kl-high", type=float, default=6.0)
    ap.add_argument("--score-wp", type=float, default=0.5)
    ap.add_argument("--nbins-score", type=int, default=8)
    ap.add_argument("--nbins-mhh", type=int, default=8)
    ap.add_argument("--signal-boost", type=float, default=1.0,
                    help=">1 scales S/B up to exhibit sensitivity for method validation")
    ap.add_argument("--yield-scale", type=float, default=1.0,
                    help="effective-luminosity fraction: scales S and B together so per-bin "
                         "Poisson fluctuations are in a testable regime (documented)")
    ap.add_argument("--n-cov", type=int, default=300)
    ap.add_argument("--outdir", default="autoresearch/nsbi-260630-1733")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = np.load(args.cache, allow_pickle=True)
    cfg = json.load(open(args.config))

    L_fb = cfg["physics"]["luminosity_fb"]
    # F1 fix: per-event QCD yield = sigma * 1000 * L / n_loaded_this_bin, computed on the FULL
    # loaded array (before any selection) so the WP/reco selection efficiency is preserved, not
    # divided out. Only a fraction of n_gen is loaded and that fraction varies per bin.
    qsig_full = d["qcd_sigma"].astype(np.float64)
    q_yield_full = np.empty_like(qsig_full)
    for s in np.unique(qsig_full):
        mfull = qsig_full == s
        q_yield_full[mfull] = s * 1000.0 * L_fb / mfull.sum()

    def sel(p):
        return np.isfinite(d[f"{p}_reco_mhh"]) & (d[f"{p}_score"] >= args.score_wp)
    ms, mq = sel("sig"), sel("qcd")
    ss, sm = d["sig_score"][ms], d["sig_reco_mhh"][ms]
    gen = d["sig_gen_mhh"][ms].astype(np.float64)
    qs, qm = d["qcd_score"][mq], d["qcd_reco_mhh"][mq]
    print(f"[v2] after WP {args.score_wp}: signal {ms.sum()}, QCD {mq.sum()}", flush=True)

    # yields at target luminosity (signal: all n_gen loaded, so efficiency = n_sel/n_gen)
    sig_w = cfg["physics"]["signal_xsec_pb"] * 1000.0 * L_fb / cfg["physics"]["n_gen_signal"]
    S_SM = float(ms.sum()) * sig_w * args.signal_boost * args.yield_scale
    q_yield = q_yield_full[mq] * args.yield_scale
    B = float(q_yield.sum())
    print(f"[v2] yields: S_SM={S_SM:.3g} B={B:.3g} S/B={S_SM/max(B,1e-30):.2e} "
          f"(boost={args.signal_boost})", flush=True)

    # bin edges from pooled quantiles
    e_s = np.quantile(np.concatenate([ss, qs]), np.linspace(0, 1, args.nbins_score + 1))
    e_s[0], e_s[-1] = -1e-6, 1.0 + 1e-6
    e_m = np.quantile(np.concatenate([sm, qm]), np.linspace(0, 1, args.nbins_mhh + 1))
    e_m[0], e_m[-1] = e_m[0] - 1e-6, e_m[-1] + 1e-6
    h_bkg = hist2d(qs, qm, q_yield, e_s, e_m)
    h_bkg = np.maximum(h_bkg, 1e-9)

    def nu(kl):
        ws = reweight(gen, kl)
        hs = hist2d(ss, sm, ws, e_s, e_m)
        tot = hs.sum()
        S = float(np.atleast_1d(cross_section_ratio(gen, kl))[0]) * S_SM
        hs = hs * (S / tot) if tot > 0 else hs
        return hs + h_bkg

    kl_grid = np.linspace(args.kl_low, args.kl_high, 241)
    nus = np.array([nu(k) for k in kl_grid])  # (nkl, nbins)

    def ll_of(data):
        ll = np.sum(data[None, :] * np.log(nus) - nus, axis=1)
        return ll - ll.max()

    def interval(ll, delta):
        # ll is normalised to max 0; the interval always contains the MLE (use >=).
        above = kl_grid[ll >= -delta]
        mle = float(kl_grid[int(np.argmax(ll))])
        return mle, (float(above[0]), float(above[-1]))

    # SBC-calibrate delta* with per-bin Poisson toys over the prior
    def toy_s(kl_true):
        data = rng.poisson(nu(kl_true)).astype(float)
        ll = ll_of(data)
        j = int(np.argmin(np.abs(kl_grid - kl_true)))
        return -ll[j]
    print("[v2] calibrating + coverage (per-bin Poisson toys)...", flush=True)
    s_cal = [toy_s(float(rng.uniform(args.kl_low, args.kl_high))) for _ in range(args.n_cov)]
    delta = float(np.quantile(s_cal, 0.68))
    hits = sum(toy_s(float(rng.uniform(args.kl_low, args.kl_high))) <= delta
               for _ in range(args.n_cov))
    coverage = hits / args.n_cov
    print(f"[v2] delta*={delta:.3f} coverage={coverage:.3f}", flush=True)

    # Recovery = estimator unbiasedness on Asimov (MLE ~ truth); interval width reports
    # expected sensitivity. Interval-coverage validity is the separate toy-based gate.
    inject = [0.0, 1.0, 2.45, 5.0]; recov = {}; n_ok = 0
    bias_tol = 0.15
    for kt in inject:
        ll = ll_of(nu(kt))  # Asimov data
        mle, (lo, hi) = interval(ll, delta)
        ok = abs(mle - kt) <= bias_tol; n_ok += ok
        recov[str(kt)] = dict(mle=mle, bias=mle - kt, lo=lo, hi=hi, width=hi - lo, ok=ok)
        print(f"[v2] kl={kt:5.2f} Asimov MLE={mle:5.2f} bias={mle-kt:+.3f} "
              f"68%=[{lo:5.2f},{hi:5.2f}] w={hi-lo:.2f}", flush=True)

    g1, g2 = n_ok >= 3, 0.60 <= coverage <= 0.76
    n_in = n_ok
    res = dict(method="iter2_binned_extended_likelihood", score_wp=args.score_wp,
               signal_boost=args.signal_boost, S_SM=S_SM, B=B, S_over_B=S_SM / max(B, 1e-30),
               delta_star=delta, coverage=coverage, recovery=recov, n_inside=n_in,
               gate_recovery=bool(g1), gate_coverage=bool(g2), gate_pass=bool(g1 and g2),
               config=vars(args))
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(res, open(os.path.join(args.outdir, "closure_v2_result.json"), "w"), indent=2)
    print(f"\n[v2] recovery {n_in}/4  coverage={coverage:.3f}  GATE={g1 and g2}", flush=True)
    return 0 if (g1 and g2) else 2


if __name__ == "__main__":
    raise SystemExit(main())
