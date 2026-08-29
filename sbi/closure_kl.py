"""
kappa_lambda NSBI closure test.

Pipeline: analytic LO reweighting -> parameterized CARL ratio estimator (weighted) ->
unbinned profile-likelihood scan -> SBC-calibrated 68% interval -> recovery + coverage.

Observable (iteration 1): x = [m_HH_reco, cos_theta*], m_HH_reco = gen m_HH smeared by
a 10% Gaussian (documented detector-resolution proxy; reweighting acts on gen m_HH while
inference sees reco m_HH, so it is a genuine inference problem). Iteration 2 replaces the
proxy with the real reconstructed di-Higgs mass + the event-level ParT score (GPU/qsub).

Two upgrades over the first closure run (I3):
  * WEIGHTED CARL training (all events, weights w_i(kl)) instead of low-N_eff resampling
    -> removes the MLE bias from diversity collapse.
  * SBC-CALIBRATED interval threshold delta* (set so 68% of calibration trials cover truth)
    instead of the asymptotic Delta(lnL)=0.5, which is invalid for a learned ratio.

Train/test split: the estimator trains on a disjoint set of SM events from the ones used
to draw pseudo-data, so the closure is not self-referential.

Gate: (1) kl_true inside the calibrated 68% interval for >= 3/4 injected points;
      (2) held-out coverage at delta* in [0.60, 0.76].
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import uproot

sys.path.insert(0, ".")
from sbi.kl_reweight import gen_higgs_kinematics, me2_coeffs_heft, KL_SM
from sbi.snre import (BoxUniform, RatioEstimator, WeightedCARLTrainer,
                      event_level_loglik_scan)

GEN_BRANCHES = ["GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass",
                "GenPart_pdgId", "GenPart_statusFlags"]
SMEAR = 0.10


def load_sm(file_glob, max_files=None):
    files = sorted(glob.glob(file_glob))
    if max_files:
        files = files[:max_files]
    mh, ct = [], []
    for f in files:
        a = uproot.open(f)["Events"].arrays(GEN_BRANCHES)
        m, c = gen_higgs_kinematics(a["GenPart_pt"], a["GenPart_eta"],
                                    a["GenPart_phi"], a["GenPart_mass"],
                                    a["GenPart_pdgId"], a["GenPart_statusFlags"])
        mh.append(m)
        ct.append(c)
    mh = np.concatenate(mh)
    ct = np.concatenate(ct)
    g = np.isfinite(mh)
    return mh[g], ct[g]


def reweight(m_hh, kl):
    a, b, c = me2_coeffs_heft(m_hh)
    den = a + b * KL_SM + c * KL_SM * KL_SM
    w = (a + b * kl + c * kl * kl) / np.maximum(den, 1e-12)
    return np.clip(w, 0.0, 1e3)


def make_reco(m_hh, cos_ts, rng):
    m_reco = m_hh * (1.0 + SMEAR * rng.standard_normal(len(m_hh)))
    return np.column_stack([m_reco, cos_ts]).astype(np.float32)


def pseudo_data(x_pool, m_pool, kl, n, rng):
    w = reweight(m_pool, kl)
    idx = rng.choice(len(x_pool), size=n, replace=True, p=w / w.sum())
    return x_pool[idx]


def s_at_truth(model, x_obs, kl_grid, kl_true, device):
    """-loglik at the truth (0 if truth is the MLE). Truth is inside the interval
    at threshold delta iff this value <= delta."""
    ll = event_level_loglik_scan(model, x_obs, kl_grid, device)
    j = int(np.argmin(np.abs(kl_grid - kl_true)))
    return -ll[j], ll


def interval_at(kl_grid, ll, delta):
    above = kl_grid[ll > -delta]
    mle = float(kl_grid[int(np.argmax(ll))])
    if len(above):
        return mle, float(above[0]), float(above[-1])
    return mle, float(kl_grid[0]), float(kl_grid[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="data/hh4b/data_*.root")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--kl-low", type=float, default=-1.0)
    ap.add_argument("--kl-high", type=float, default=6.0)
    ap.add_argument("--n-steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--n-obs", type=int, default=2000)
    ap.add_argument("--n-cal", type=int, default=300)
    ap.add_argument("--n-cov", type=int, default=300)
    ap.add_argument("--outdir", default="autoresearch/nsbi-260630-1733")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    print(f"[closure] device={device}", flush=True)

    m_hh, cos_ts = load_sm(args.files, args.max_files)
    x_all = make_reco(m_hh, cos_ts, rng)
    n = len(m_hh)
    print(f"[closure] SM events: {n}", flush=True)

    # disjoint train (estimator) / test (pseudo-data) split
    perm = rng.permutation(n)
    tr, te = perm[: int(0.6 * n)], perm[int(0.6 * n):]
    mu, sd = x_all[tr].mean(0), x_all[tr].std(0)
    sd[sd < 1e-8] = 1.0
    std = lambda x: (x - mu) / sd
    x_tr_s, x_te_s = std(x_all[tr]), std(x_all[te])
    m_tr, m_te = m_hh[tr], m_hh[te]

    prior = BoxUniform([args.kl_low], [args.kl_high])

    print("[closure] training weighted CARL ratio estimator...", flush=True)
    est = RatioEstimator(x_dim=x_tr_s.shape[1], theta_dim=1)
    trainer = WeightedCARLTrainer(prior, reweight, est, device=device, lr=1e-3)
    model = trainer.train(x_tr_s, m_tr, n_steps=args.n_steps, batch=args.batch)

    kl_grid = np.linspace(args.kl_low, args.kl_high, 181)

    # ---- calibrate the interval threshold delta* for 68% coverage ----
    print("[closure] calibrating interval threshold (SBC)...", flush=True)
    s_cal = []
    for _ in range(args.n_cal):
        kl_true = float(prior.sample(1)[0, 0])
        x_obs = std(pseudo_data(x_all[te], m_te, kl_true, args.n_obs, rng))
        s, _ = s_at_truth(model, x_obs, kl_grid, kl_true, device)
        s_cal.append(s)
    delta_star = float(np.quantile(s_cal, 0.68))
    print(f"[closure] delta* (68%) = {delta_star:.3f} "
          f"(asymptotic would be 0.5)", flush=True)

    # ---- held-out coverage at delta* ----
    print("[closure] measuring held-out coverage...", flush=True)
    hits = 0
    for _ in range(args.n_cov):
        kl_true = float(prior.sample(1)[0, 0])
        x_obs = std(pseudo_data(x_all[te], m_te, kl_true, args.n_obs, rng))
        s, _ = s_at_truth(model, x_obs, kl_grid, kl_true, device)
        hits += (s <= delta_star)
    coverage = hits / args.n_cov

    # ---- recovery on injected points (calibrated interval) ----
    inject = [0.0, 1.0, 2.45, 5.0]
    recov, n_in = {}, 0
    for kl_true in inject:
        x_obs = std(pseudo_data(x_all[te], m_te, kl_true, args.n_obs, rng))
        _, ll = s_at_truth(model, x_obs, kl_grid, kl_true, device)
        mle, lo, hi = interval_at(kl_grid, ll, delta_star)
        inside = bool(lo <= kl_true <= hi)
        n_in += inside
        recov[str(kl_true)] = dict(mle=mle, lo=lo, hi=hi, inside=inside)
        print(f"[closure] kl_true={kl_true:5.2f} -> MLE={mle:5.2f} "
              f"68%=[{lo:5.2f},{hi:5.2f}] inside={inside}", flush=True)

    gate1 = n_in >= 3
    gate2 = 0.60 <= coverage <= 0.76
    result = dict(method="weighted_carl+sbc_calibrated", n_sm=int(n),
                  delta_star=delta_star, recovery=recov, n_inside=int(n_in),
                  coverage=float(coverage), gate_recovery=bool(gate1),
                  gate_coverage=bool(gate2), gate_pass=bool(gate1 and gate2),
                  config=vars(args))
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "closure_result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n[closure] recovery {n_in}/4 inside 68% (gate>=3: {gate1})", flush=True)
    print(f"[closure] coverage={coverage:.3f} at delta*={delta_star:.3f} "
          f"(gate [0.60,0.76]: {gate2})", flush=True)
    print(f"[closure] GATE PASS = {gate1 and gate2}", flush=True)
    return 0 if (gate1 and gate2) else 2


if __name__ == "__main__":
    raise SystemExit(main())
