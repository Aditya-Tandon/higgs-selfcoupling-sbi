"""
kappa_lambda NSBI closure test (iteration 1).

Pipeline under test: analytic LO reweighting -> parameterized SNRE-B ratio
estimator -> unbinned profile-likelihood scan -> kl recovery + coverage.

Observable (iteration 1): x = [m_HH_reco, cos_theta*], where m_HH_reco is the gen
m_HH smeared by a 10% Gaussian as a *documented detector-resolution proxy*. The
reweighting acts on gen m_HH while inference sees the smeared reco m_HH, so this is
a genuine (non-circular) inference problem. Iteration 2 replaces the proxy with the
real reconstructed di-Higgs mass + the event-level ParT score (GPU, via qsub).

Gate (see autoresearch plan):
  (1) kl_true inside the recovered 68% interval for >= 3/4 injected points;
  (2) 68%-interval coverage in [0.60, 0.76] over the coverage trials.
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
from sbi.snre import (BoxUniform, RatioEstimator, SNRETrainer,
                      event_level_loglik_scan, extract_confidence_interval)

GEN_BRANCHES = ["GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass",
                "GenPart_pdgId", "GenPart_statusFlags"]
SMEAR = 0.10  # detector resolution proxy on m_HH (iteration-1)


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
    num = a + b * kl + c * kl * kl
    den = a + b * KL_SM + c * KL_SM * KL_SM
    w = num / np.maximum(den, 1e-12)
    return np.clip(w, 0.0, 1e3)


def reco_observable(m_hh, cos_ts, rng):
    """[m_HH_reco, cos_theta*] with a 10% Gaussian smear on m_HH."""
    m_reco = m_hh * (1.0 + SMEAR * rng.standard_normal(len(m_hh)))
    return np.column_stack([m_reco, cos_ts]).astype(np.float32)


def resample_at_kl(m_hh, cos_ts, kl, n, rng):
    """Importance-resample n SM events ~ w(kl), return their reco observable."""
    w = reweight(m_hh, kl)
    p = w / w.sum()
    idx = rng.choice(len(m_hh), size=n, replace=True, p=p)
    return reco_observable(m_hh[idx], cos_ts[idx], rng)


def build_training_set(m_hh, cos_ts, prior, n_kl, n_per_kl, rng):
    thetas, xs = [], []
    for _ in range(n_kl):
        kl = float(prior.sample(1)[0, 0])
        x = resample_at_kl(m_hh, cos_ts, kl, n_per_kl, rng)
        thetas.append(np.full((n_per_kl, 1), kl, np.float32))
        xs.append(x)
    return np.concatenate(thetas), np.concatenate(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="data/hh4b/data_*.root")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--kl-low", type=float, default=-1.0)
    ap.add_argument("--kl-high", type=float, default=6.0)
    ap.add_argument("--n-kl", type=int, default=400)
    ap.add_argument("--n-per-kl", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n-obs", type=int, default=2000, help="events per pseudo-dataset")
    ap.add_argument("--n-cov", type=int, default=200, help="coverage trials")
    ap.add_argument("--outdir", default="autoresearch/nsbi-260630-1733")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    print(f"[closure] device={device}", flush=True)

    m_hh, cos_ts = load_sm(args.files, args.max_files)
    print(f"[closure] SM events: {len(m_hh)}", flush=True)

    prior = BoxUniform([args.kl_low], [args.kl_high])

    # standardize observables on an SM (kl=1) reference draw
    x_ref = reco_observable(m_hh, cos_ts, rng)
    mu, sd = x_ref.mean(0), x_ref.std(0)
    sd[sd < 1e-8] = 1.0
    std = lambda x: (x - mu) / sd

    print("[closure] building parameterized training set...", flush=True)
    theta_tr, x_tr = build_training_set(m_hh, cos_ts, prior, args.n_kl,
                                        args.n_per_kl, rng)
    x_tr = std(x_tr)

    print("[closure] training SNRE-B ratio estimator...", flush=True)
    est = RatioEstimator(x_dim=x_tr.shape[1], theta_dim=1)
    tr = SNRETrainer(prior, estimator=est, device=device, lr=1e-3)
    tr.append_simulations(theta_tr, x_tr)
    model = tr.train(n_epochs=args.epochs, batch_size=512, patience=25)

    kl_grid = np.linspace(args.kl_low, args.kl_high, 241)

    # ---- recovery on injected points ----
    inject = [0.0, 1.0, 2.45, 5.0]
    recov = {}
    n_in = 0
    for kl_true in inject:
        x_obs = std(resample_at_kl(m_hh, cos_ts, kl_true, args.n_obs, rng))
        ll = event_level_loglik_scan(model, x_obs, kl_grid, device)
        mle, lo, hi = extract_confidence_interval(kl_grid, ll, delta=0.5)
        inside = bool(lo <= kl_true <= hi)
        n_in += inside
        recov[str(kl_true)] = dict(mle=float(mle), lo=lo, hi=hi, inside=inside)
        print(f"[closure] kl_true={kl_true:5.2f} -> MLE={mle:5.2f} "
              f"68%=[{lo:5.2f},{hi:5.2f}] inside={inside}", flush=True)

    # ---- coverage (SBC-style) over prior draws ----
    cov_hits = 0
    for _ in range(args.n_cov):
        kl_true = float(prior.sample(1)[0, 0])
        x_obs = std(resample_at_kl(m_hh, cos_ts, kl_true, args.n_obs, rng))
        ll = event_level_loglik_scan(model, x_obs, kl_grid, device)
        _, lo, hi = extract_confidence_interval(kl_grid, ll, delta=0.5)
        cov_hits += (lo <= kl_true <= hi)
    coverage = cov_hits / args.n_cov

    gate1 = n_in >= 3
    gate2 = 0.60 <= coverage <= 0.76
    result = dict(n_sm=int(len(m_hh)), recovery=recov, n_inside=int(n_in),
                  coverage=float(coverage), gate_recovery=bool(gate1),
                  gate_coverage=bool(gate2), gate_pass=bool(gate1 and gate2),
                  config=vars(args))
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "closure_result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n[closure] recovery {n_in}/4 inside 68%  (gate>=3: {gate1})", flush=True)
    print(f"[closure] coverage = {coverage:.3f}  (gate [0.60,0.76]: {gate2})", flush=True)
    print(f"[closure] GATE PASS = {gate1 and gate2}", flush=True)
    return 0 if (gate1 and gate2) else 2


if __name__ == "__main__":
    raise SystemExit(main())
