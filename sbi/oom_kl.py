"""
OOM-style κλ observable — Ablation 0: the regime-transport diagnostic.

Scope
-----
This is the *transport diagnostic* of the Fisher-vs-NRE observable plan
(`Experiments/l1-scouting-sbi/kl-observable-fisher-vs-nre-comparison.md`, Ablation 0),
NOT the full head-to-head. It answers one question, in κλ units:

    How much does an observable optimized on the AMBIENT (full) phase space
    over-promise once it is restricted to the OPERATING (preselected) regime?

This is the PF/tracking failure ([[pf-tracking-lever-preselection-result]]) — where the
full-sample AUC ordering *inverted* under preselection — turned into a controlled,
pre-registered check. PF carried more information on the full phase space yet less on the
preselected one; the ambient gain did not *transport*. Here we measure the same effect on
the κλ-parameter axis via Fisher information.

Method (faithful to the Optimal Observable Machine, Mohr et al. 2026, arXiv:2601.08813)
--------------------------------------------------------------------------------------
* Observable: a small MLP  O(x) = sigmoid(Φ(x)) ∈ (0,1)  of the cached per-event features
  (default [score, reco m_HH]) — the low-dim, differentiable summary OOM learns.
* Differentiable histogramming ℋ: a soft (Gaussian-kernel) assignment of O to `nbins`,
  so the binned templates are differentiable in the observable's weights.
* Binned extended likelihood (identical structure to sbi/closure_kl_v2.py):
      ν_b(κλ) = S(κλ) · f_sig,b(κλ)  +  t_bkg,b ,   S(κλ) = σ_ratio(κλ) · S_SM ,
  with the analytic HEFT κλ reweighting on the signal (gen m_HH).
* Loss = the expected κλ uncertainty (Cramér–Rao), Δκλ = 1/sqrt(I_F(κλ_fit)), with the
  Poisson Fisher information of the bin counts
      I_F(κλ) = Σ_b (∂_κλ ν_b)² / ν_b .
  Minimizing Δκλ = maximizing Fisher information about κλ. (∂_κλ ν_b by central finite
  difference over a fine κλ grid — differentiable in the observable weights.)

Transport comparison
--------------------
Fit observable A on the ambient cache and observable B on the operating cache, then
evaluate BOTH on the OPERATING cache (each applied exactly as trained, i.e. carrying its
own feature standardization — so distribution shift is respected). Report, per evaluation
κλ:
    Δκλ  A-on-ambient   (apparent sensitivity — what ambient optimization advertises)
    Δκλ  A-on-operating (transported — the honest number)
    Δκλ  B-on-operating (regime-native reference — the best the operating regime allows)
and the two headline ratios
    over_promise  = Δκλ(A-on-operating) / Δκλ(A-on-ambient)      (>1 ⇒ degrades on transport)
    transport_gap = Δκλ(A-on-operating) / Δκλ(B-on-operating) − 1 (>0 ⇒ worse than native)

A large over_promise / positive transport_gap is the κλ-unit instantiation of the PF
inversion and a confirmation of prediction P2 in the field-link synthesis
([[sbi-epiplexity-categorical-discovery-field-link]]).

Caveats carried from the plan
-----------------------------
* Optimize on Asimov-Fisher (here) but *gate* on SBC toys (the full experiment / closure_kl_v2)
  — do not report the Asimov Δκλ as the final measured uncertainty.
* Fisher is *local* (κλ_fit); for non-local κλ sensitivity report I_F across the whole grid,
  not just at the fit point (that locality is exactly what the multi-point arm A2 addresses).
* All information is measured on the OPERATING regime — never rank an observable by an
  ambient-phase-space number.

Example
-------
    python sbi/oom_kl.py --transport \
        --cache-ambient   data/event_level/nsbi_cache.npz \
        --cache-operating data/event_level/nsbi_cache_trig.npz \
        --operating-qcd-n-loaded 40000 \
        --signal-boost 6e5 --yield-scale 1e-5 \
        --outdir autoresearch/oom-transport
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from sbi.kl_reweight import me2_coeffs_heft, cross_section_ratio, KL_SM

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise SystemExit("oom_kl requires torch (env hep-root-ml has torch 2.9.1)") from e


# --------------------------------------------------------------------------- #
# κλ reweight (mirrors sbi/closure_kl_v2.py:reweight — numpy, constant wrt the observable)
# --------------------------------------------------------------------------- #
def reweight(gen_mhh, kl):
    a, b, c = me2_coeffs_heft(gen_mhh)
    den = a + b * KL_SM + c * KL_SM * KL_SM
    return np.clip((a + b * kl + c * kl * kl) / np.maximum(den, 1e-12), 0.0, 1e3)


# --------------------------------------------------------------------------- #
# Regime = one cache + its yields, with the same reco/WP selection as closure_kl_v2
# --------------------------------------------------------------------------- #
FEATURE_COLS = {"score": "score", "reco_mhh": "reco_mhh"}


class Regime:
    """Selected signal/QCD events of one cache, with per-event yields and κλ machinery."""

    def __init__(self, cache_path, cfg, features, score_wp, signal_boost, yield_scale,
                 qcd_n_loaded=None):
        d = np.load(cache_path, allow_pickle=True)
        self.name = os.path.basename(cache_path)

        def sel(p):
            return np.isfinite(d[f"{p}_reco_mhh"]) & (d[f"{p}_score"] >= score_wp)

        ms, mq = sel("sig"), sel("qcd")

        # features (raw, per event) for signal and QCD
        self.feat_sig = np.column_stack([d[f"sig_{FEATURE_COLS[f]}"][ms] for f in features]).astype(np.float64)
        self.feat_qcd = np.column_stack([d[f"qcd_{FEATURE_COLS[f]}"][mq] for f in features]).astype(np.float64)
        self.gen = d["sig_gen_mhh"][ms].astype(np.float64)

        # yields (identical recipe to closure_kl_v2)
        L_fb = cfg["physics"]["luminosity_fb"]
        sig_w = cfg["physics"]["signal_xsec_pb"] * 1000.0 * L_fb / cfg["physics"]["n_gen_signal"]
        self.S_SM = float(ms.sum()) * sig_w * signal_boost * yield_scale

        qsig_full = d["qcd_sigma"].astype(np.float64)          # over ALL loaded QCD (pre-sel)
        q_yield_full = np.empty_like(qsig_full)
        for s in np.unique(qsig_full):
            mfull = qsig_full == s
            denom = float(mfull.sum()) if qcd_n_loaded is None else float(qcd_n_loaded)
            q_yield_full[mfull] = s * 1000.0 * L_fb / denom
        self.q_yield = q_yield_full[mq] * yield_scale
        self.B = float(self.q_yield.sum())

        print(f"[regime {self.name}] sig {ms.sum()} qcd {mq.sum()}  "
              f"S_SM={self.S_SM:.3g} B={self.B:.3g} S/B={self.S_SM/max(self.B,1e-30):.2e}",
              flush=True)

    def standardization(self):
        pooled = np.vstack([self.feat_sig, self.feat_qcd])
        mu = pooled.mean(0)
        sd = pooled.std(0) + 1e-9
        return mu, sd

    def sig_reweights(self, kl_grid):
        """Per-event signal reweights ws_i(κλ) and the scalar S(κλ) on this regime's events."""
        ws = np.stack([reweight(self.gen, k) for k in kl_grid])              # (nkl, n_sig)
        xsec = np.atleast_1d(cross_section_ratio(self.gen, kl_grid)).astype(np.float64)
        S = xsec * self.S_SM                                                 # (nkl,)
        return ws, S


# --------------------------------------------------------------------------- #
# Observable (small MLP → (0,1)) and differentiable soft histogram
# --------------------------------------------------------------------------- #
class Observable(torch.nn.Module):
    def __init__(self, n_features, hidden=(64, 64)):
        super().__init__()
        layers, d = [], n_features
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
            d = h
        layers += [torch.nn.Linear(d, 1)]
        self.net = torch.nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))

    def set_standardization(self, mu, sd):
        self.mu.copy_(torch.as_tensor(mu, dtype=self.mu.dtype))
        self.sd.copy_(torch.as_tensor(sd, dtype=self.sd.dtype))

    def forward(self, x):  # x: (n, n_features) raw
        z = (x - self.mu) / self.sd
        return torch.sigmoid(self.net(z)).squeeze(-1)          # O ∈ (0,1)


def soft_hist(o, weights, nbins, bandwidth=None):
    """Differentiable histogram of o∈(0,1): Gaussian-kernel soft assignment to bin centers.

    Returns an (nbins,) tensor of weighted soft counts (differentiable in `o`)."""
    if bandwidth is None:
        bandwidth = 0.75 / nbins
    centers = (torch.arange(nbins, dtype=o.dtype, device=o.device) + 0.5) / nbins
    d = (o[:, None] - centers[None, :]) / bandwidth
    a = torch.softmax(-0.5 * d * d, dim=1)                      # (n, nbins), rows sum to 1
    return (weights[:, None] * a).sum(0)                        # (nbins,)


# --------------------------------------------------------------------------- #
# Templates ν_b(κλ) and the κλ Fisher-information profile
# --------------------------------------------------------------------------- #
def nu_grid(obs, reg, kl_grid, ws, S, nbins, device):
    """ν_b(κλ) for every κλ in kl_grid, differentiable in the observable weights.

    ws:(nkl,n_sig) sig reweights, S:(nkl,) sig yields — numpy constants wrt obs weights."""
    xsig = torch.as_tensor(reg.feat_sig, dtype=torch.float32, device=device)
    xqcd = torch.as_tensor(reg.feat_qcd, dtype=torch.float32, device=device)
    o_sig = obs(xsig)
    o_qcd = obs(xqcd)
    qy = torch.as_tensor(reg.q_yield, dtype=torch.float32, device=device)
    t_bkg = soft_hist(o_qcd, qy, nbins)                         # sums ≈ B (differentiable)

    ws_t = torch.as_tensor(ws, dtype=torch.float32, device=device)
    S_t = torch.as_tensor(S, dtype=torch.float32, device=device)
    rows = []
    for k in range(kl_grid.shape[0]):
        t_sig = soft_hist(o_sig, ws_t[k], nbins)
        f_sig = t_sig / torch.clamp(t_sig.sum(), min=1e-12)     # shape, Σ_b = 1
        rows.append(S_t[k] * f_sig + t_bkg)
    return torch.stack(rows, dim=0)                             # (nkl, nbins)


def fisher_profile(nu, kl_grid_t):
    """I_F(κλ) = Σ_b (∂_κλ ν_b)² / ν_b via central differences over the κλ grid."""
    dnu = torch.zeros_like(nu)
    dnu[1:-1] = (nu[2:] - nu[:-2]) / (kl_grid_t[2:, None] - kl_grid_t[:-2, None])
    dnu[0] = (nu[1] - nu[0]) / (kl_grid_t[1] - kl_grid_t[0])
    dnu[-1] = (nu[-1] - nu[-2]) / (kl_grid_t[-1] - kl_grid_t[-2])
    return (dnu * dnu / torch.clamp(nu, min=1e-9)).sum(dim=1)   # (nkl,)


# --------------------------------------------------------------------------- #
# Fit an observable on a regime by minimizing Cramér–Rao Δκλ at κλ_fit
# --------------------------------------------------------------------------- #
def fit_observable(reg, features, kl_grid, kl_fit, nbins, steps, lr, seed, device):
    torch.manual_seed(seed)
    obs = Observable(len(features)).to(device)
    obs.set_standardization(*reg.standardization())
    ws, S = reg.sig_reweights(kl_grid)
    kl_grid_t = torch.as_tensor(kl_grid, dtype=torch.float32, device=device)
    i_fit = int(np.argmin(np.abs(kl_grid - kl_fit)))
    opt = torch.optim.Adam(obs.parameters(), lr=lr)
    for it in range(steps):
        opt.zero_grad()
        nu = nu_grid(obs, reg, kl_grid, ws, S, nbins, device)
        IF = fisher_profile(nu, kl_grid_t)
        loss = -torch.log(IF[i_fit] + 1e-12)                   # maximize Fisher at κλ_fit
        loss.backward()
        opt.step()
        if it % max(1, steps // 5) == 0 or it == steps - 1:
            print(f"    [fit {reg.name} @κλ={kl_fit}] step {it:4d}  "
                  f"Δκλ={float(1.0/torch.sqrt(IF[i_fit]+1e-12)):.4f}", flush=True)
    return obs


@torch.no_grad()
def evaluate(obs, reg, kl_grid, eval_kls, nbins, device):
    """Δκλ = 1/sqrt(I_F) at each eval κλ, applying `obs` to regime `reg` (as-trained)."""
    ws, S = reg.sig_reweights(kl_grid)
    kl_grid_t = torch.as_tensor(kl_grid, dtype=torch.float32, device=device)
    nu = nu_grid(obs, reg, kl_grid, ws, S, nbins, device)
    IF = fisher_profile(nu, kl_grid_t).cpu().numpy()
    out = {}
    for kt in eval_kls:
        j = int(np.argmin(np.abs(kl_grid - kt)))
        out[str(kt)] = dict(I_F=float(IF[j]), dkl=float(1.0 / np.sqrt(IF[j] + 1e-12)))
    return out, (kl_grid.tolist(), IF.tolist())


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transport", action="store_true",
                    help="run the ambient-vs-operating transport diagnostic (Ablation 0)")
    ap.add_argument("--cache-ambient", default="data/event_level/nsbi_cache.npz",
                    help="full (pre-preselection) phase-space cache")
    ap.add_argument("--cache-operating", default="data/event_level/nsbi_cache_trig.npz",
                    help="preselected (trigger-applied) cache — the operating regime")
    ap.add_argument("--operating-qcd-n-loaded", type=float, default=40000,
                    help="per-QCD-bin loaded count for the preselected cache (survivors-only); "
                         "REQUIRED so trigger efficiency is not divided out (cf. closure_kl_v2)")
    ap.add_argument("--ambient-qcd-n-loaded", type=float, default=None)
    ap.add_argument("--config", default="hh-bbbb-obj-config.json")
    ap.add_argument("--features", default="score,reco_mhh")
    ap.add_argument("--score-wp", type=float, default=0.5)
    ap.add_argument("--kl-low", type=float, default=-1.0)
    ap.add_argument("--kl-high", type=float, default=6.0)
    ap.add_argument("--nkl", type=int, default=121, help="κλ grid points for templates/Fisher")
    ap.add_argument("--kl-fit", type=float, default=1.0, help="working point for the Fisher loss")
    ap.add_argument("--eval-kls", default="-1,0,1,2.45,5")
    ap.add_argument("--nbins", type=int, default=12)
    ap.add_argument("--signal-boost", type=float, default=1.0,
                    help="scale S/B up to a sensitivity-bearing regime (as closure_kl_v2 I6-valid); "
                         "the transport RATIOS are the headline and are boost-robust")
    ap.add_argument("--yield-scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", default="autoresearch/oom-transport")
    ap.add_argument("--plot", action="store_true", help="write I_F(κλ) transport plot if matplotlib")
    args = ap.parse_args()

    if not args.transport:
        ap.error("only --transport (Ablation 0) is implemented in this scoped draft")

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    for f in features:
        if f not in FEATURE_COLS:
            ap.error(f"unknown feature '{f}'; known: {list(FEATURE_COLS)}")
    eval_kls = [float(x) for x in args.eval_kls.split(",")]
    cfg = json.load(open(args.config))
    kl_grid = np.linspace(args.kl_low, args.kl_high, args.nkl)
    dev = args.device
    print(f"[oom-transport] device={dev} features={features} nbins={args.nbins} "
          f"boost={args.signal_boost}", flush=True)

    amb = Regime(args.cache_ambient, cfg, features, args.score_wp, args.signal_boost,
                 args.yield_scale, qcd_n_loaded=args.ambient_qcd_n_loaded)
    op = Regime(args.cache_operating, cfg, features, args.score_wp, args.signal_boost,
                args.yield_scale, qcd_n_loaded=args.operating_qcd_n_loaded)

    print("[oom-transport] fitting observable A on AMBIENT ...", flush=True)
    obsA = fit_observable(amb, features, kl_grid, args.kl_fit, args.nbins,
                          args.steps, args.lr, args.seed, dev)
    print("[oom-transport] fitting observable B on OPERATING ...", flush=True)
    obsB = fit_observable(op, features, kl_grid, args.kl_fit, args.nbins,
                          args.steps, args.lr, args.seed, dev)

    A_on_amb, prof_A_amb = evaluate(obsA, amb, kl_grid, eval_kls, args.nbins, dev)
    A_on_op, prof_A_op = evaluate(obsA, op, kl_grid, eval_kls, args.nbins, dev)
    B_on_op, prof_B_op = evaluate(obsB, op, kl_grid, eval_kls, args.nbins, dev)

    print("\n[oom-transport] κλ-unit transport report (Δκλ = Cramér–Rao 1/sqrt(I_F)):")
    print(f"    {'κλ':>6} {'A@ambient':>11} {'A@operating':>12} {'B@operating':>12} "
          f"{'over_promise':>13} {'transport_gap':>14}")
    per_kl = {}
    for kt in eval_kls:
        k = str(kt)
        da_a, da_o, db_o = A_on_amb[k]["dkl"], A_on_op[k]["dkl"], B_on_op[k]["dkl"]
        over = da_o / max(da_a, 1e-12)
        gap = da_o / max(db_o, 1e-12) - 1.0
        per_kl[k] = dict(dkl_A_ambient=da_a, dkl_A_operating=da_o, dkl_B_operating=db_o,
                         over_promise=over, transport_gap=gap,
                         I_F_A_ambient=A_on_amb[k]["I_F"], I_F_A_operating=A_on_op[k]["I_F"],
                         I_F_B_operating=B_on_op[k]["I_F"])
        print(f"    {kt:6.2f} {da_a:11.4f} {da_o:12.4f} {db_o:12.4f} "
              f"{over:13.2f} {gap:+14.2f}", flush=True)

    res = dict(mode="transport_ablation0", features=features, nbins=args.nbins,
               kl_fit=args.kl_fit, score_wp=args.score_wp, signal_boost=args.signal_boost,
               yields=dict(ambient=dict(S_SM=amb.S_SM, B=amb.B),
                           operating=dict(S_SM=op.S_SM, B=op.B)),
               per_kl=per_kl,
               profiles=dict(kl_grid=prof_A_op[0], I_F_A_operating=prof_A_op[1],
                             I_F_B_operating=prof_B_op[1], I_F_A_ambient=prof_A_amb[1]),
               config=vars(args))
    os.makedirs(args.outdir, exist_ok=True)
    out_json = os.path.join(args.outdir, "oom_transport_result.json")
    json.dump(res, open(out_json, "w"), indent=2)
    print(f"\n[oom-transport] wrote {out_json}", flush=True)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            kg = np.array(prof_A_op[0])
            plt.figure(figsize=(6, 4))
            plt.plot(kg, prof_A_amb[1], "--", label="A optimized@ambient, on ambient (apparent)")
            plt.plot(kg, prof_A_op[1], "-", label="A optimized@ambient, on operating (transported)")
            plt.plot(kg, prof_B_op[1], "-", label="B optimized@operating, on operating (native)")
            plt.xlabel(r"$\kappa_\lambda$"); plt.ylabel(r"$I_F(\kappa_\lambda)$")
            plt.yscale("log"); plt.legend(fontsize=8); plt.tight_layout()
            p = os.path.join(args.outdir, "oom_transport_IF.png")
            plt.savefig(p, dpi=130)
            print(f"[oom-transport] wrote {p}", flush=True)
        except Exception as e:  # pragma: no cover
            print(f"[oom-transport] plot skipped: {e}", flush=True)

    # informative exit: nonzero if the ambient observable does NOT degrade on transport
    # (i.e. the diagnostic found no over-promise — worth a human look at the setup)
    worst_over = max(per_kl[str(k)]["over_promise"] for k in eval_kls)
    return 0 if worst_over > 1.05 else 3


if __name__ == "__main__":
    raise SystemExit(main())
