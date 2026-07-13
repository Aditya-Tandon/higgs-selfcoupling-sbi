"""
Direction 1 — κλ acceptance-vs-information map (production version).

Vault plan: Experiments/l1-scouting-sbi/kl-acceptance-vs-information-map.md
Gate metric: fraction of total κλ shape Fisher information surviving each
acceptance stage, resolved in gen m_HH (primary map 2D in (m_HH, |cosθ*|)).

Runs entirely on the extended ambient cache (nsbi_cache.npz, skip-trigger:
ALL loaded signal events with gen kinematics + per-jet pT/HT/b-tag), so every
preselection component is a *mask* — the plan's step 2-iii trigger-emulation
re-runs are no longer needed post-rebuild. The replicated full-preselection
mask is validated row-for-row against the operating cache (nsbi_cache_trig.npz).

Stages (masks on the finite-gen universe):
  total      all signal events with gen (m_HH, cosθ*)
  reco4j     ≥4-jet constituent reco succeeded (finite sig_reco_mhh)
  pt4        leading-4 l1ext jet pT > (75, 60, 45, 40) GeV
  ht330      all-jet HT > 330 GeV
  btag3      ≥3 l1ext jets with btagScore > config WP (top-10 jets stored;
             validated exact vs the trig cache)
  presel     pt4 & ht330 & btag3 (= trigger emulation)
  presel+reco  presel & reco4j (the analysis operating set)

Outputs (to --outdir): results.json (R table, yield/shape split, gate
verdict, N_eff, bootstrap errors), overlay + 2D-map + eps(κλ) figures.
Caveat stamped on every figure: LO HEFT reweighting (m_HH-only), shape
fidelity unvalidated — see forward-model-fidelity-validation.

Usage: python sbi/info_acceptance_map.py --outdir autoresearch/info-acceptance-map
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from sbi.fisher_info import (event_score_t, event_weight_w, kish_n_eff,
                             shape_info_sum, yield_info_sum,
                             ullrich_xu_efficiency)
from sbi.kl_reweight import me2_coeffs_heft

CAVEAT = "LO HEFT reweighting (m_HH-only); shape fidelity unvalidated"


def mhh_edges(lo=250.0, split=600.0, hi=1000.0, n_lin=25, n_log=15):
    """Plan binning: ~40 bins, linear below 600 GeV, log-spaced above."""
    return np.concatenate([np.linspace(lo, split, n_lin + 1)[:-1],
                           np.geomspace(split, hi, n_log + 1)])


def stage_masks(d, wp):
    """Selection-stage masks on the finite-gen universe (bool arrays)."""
    gen = d["sig_gen_mhh"].astype(np.float64)
    uni = np.isfinite(gen) & np.isfinite(d["sig_cos_star"])
    jp = d["sig_jet_pt"]          # (N, k), pT-sorted, NaN-padded
    with np.errstate(invalid="ignore"):
        pt4 = ((jp[:, 0] > 75) & (jp[:, 1] > 60)
               & (jp[:, 2] > 45) & (jp[:, 3] > 40))
        ht = d["sig_ht"] > 330.0
        btag = (np.nan_to_num(d["sig_jet_btag"], nan=-np.inf) > wp).sum(1) >= 3
    reco = np.isfinite(d["sig_reco_mhh"])
    presel = pt4 & ht & btag
    stages = {
        "total": uni,
        "reco4j": uni & reco,
        "pt4": uni & pt4,
        "ht330": uni & ht,
        "btag3": uni & btag,
        "presel": uni & presel,
        "presel+reco": uni & presel & reco,
    }
    return stages, uni, presel


def validate_against_trig(d, presel, trig_path):
    """The replicated preselection mask must reproduce the operating cache
    row-for-row (build order is deterministic). The one legitimate source of
    mismatch is the top-10 b-tag truncation (real trigger counts ALL jets)."""
    if not os.path.exists(trig_path):
        return {"checked": False, "reason": f"{trig_path} not found"}
    t = np.load(trig_path)
    n_trig = len(t["sig_gen_mhh"])
    n_mask = int(presel.sum())
    rows_equal = (n_trig == n_mask and np.allclose(
        d["sig_gen_mhh"][presel], t["sig_gen_mhh"], equal_nan=True))
    out = {"checked": True, "n_trig_cache": n_trig, "n_replicated": n_mask,
           "rows_equal": bool(rows_equal)}
    print(f"[dir1] presel-mask validation vs {trig_path}: "
          f"replicated {n_mask} vs trig-cache {n_trig} "
          f"({'EXACT row match' if rows_equal else 'MISMATCH'})")
    if not rows_equal:
        print("[dir1] WARNING: replicated preselection differs from the trig "
              "cache (likely top-10 b-tag truncation) — R_presel uses the "
              "replicated mask; interpret with the count difference in mind.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/event_level/nsbi_cache.npz")
    ap.add_argument("--cache-trig", default="data/event_level/nsbi_cache_trig.npz")
    ap.add_argument("--config", default="hh-bbbb-obj-config.json")
    ap.add_argument("--kls", default="-1,0,1,2.45,5")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--neff-grey", type=float, default=50.0)
    ap.add_argument("--outdir", default="autoresearch/info-acceptance-map")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(20260713)

    cfg = json.load(open(args.config))
    wp = float(cfg["l1ext"]["b_tag_cut"])
    phys = cfg["physics"]
    sig_w = phys["signal_xsec_pb"] * 1000.0 * phys["luminosity_fb"] \
        / phys["n_gen_signal"]
    kls = [float(k) for k in args.kls.split(",")]

    d = np.load(args.cache)
    stages, uni, presel = stage_masks(d, wp)
    gen = d["sig_gen_mhh"].astype(np.float64)
    cos = np.abs(d["sig_cos_star"].astype(np.float64))
    print(f"[dir1] universe {uni.sum()}/{len(gen)} events with gen kinematics; "
          f"l1ext b-tag WP {wp}")
    for name, m in stages.items():
        print(f"[dir1]   {name:12s} {m.sum():6d}  ({m.sum()/uni.sum():.3f})")
    validation = validate_against_trig(d, presel, args.cache_trig)
    # NaN-safe: non-universe rows never enter any sum/histogram (all stage
    # masks are subsets of uni), but a NaN m_HH would poison the global
    # denominator floor inside reweight_heft — park those rows at a dummy value
    gen = np.where(uni, gen, 350.0)
    cos = np.where(uni, cos, 0.5)

    edges = mhh_edges()
    ctr = 0.5 * (edges[:-1] + edges[1:])
    results = {"stages": {k: int(m.sum()) for k, m in stages.items()},
               "btag_wp": wp, "sig_w_per_event": sig_w, "kls": kls,
               "mask_validation": validation, "caveat": CAVEAT,
               "mhh_edges": edges.tolist()}

    # ---- per-κλ information table (event sums — binning-independent) ----
    per_kl, dens = {}, {}
    for kl in kls:
        w = event_weight_w(gen, kl)
        t = event_score_t(gen, kl)
        row = {}
        shape_tot = shape_info_sum(w, t, stages["total"])
        for name, m in stages.items():
            sh, yl = shape_info_sum(w, t, m), yield_info_sum(w, t, m)
            row[name] = {
                "n": int(m.sum()),
                "shape_info": sig_w * sh,            # S(κλ)·Var_w[t], physical
                "yield_info": sig_w * yl,            # (∂S)²/S (background-free)
                "R_shape": sh / shape_tot if shape_tot > 0 else np.nan,
                "n_eff_kish": kish_n_eff(w[m]),
            }
        per_kl[str(kl)] = row
        # binned densities at this κλ (per stage: own weighted mean)
        dd = {}
        for name in ("total", "reco4j", "presel"):
            m = stages[name]
            sw = w[m].sum()
            tbar = (w[m] * t[m]).sum() / sw if sw > 0 else 0.0
            dd[name], _ = np.histogram(gen[m], edges,
                                       weights=w[m] * (t[m] - tbar) ** 2)
        # per-bin Kish N_eff of the reweighted total sample
        neff_bin = np.zeros(len(ctr))
        bi = np.digitize(gen[stages["total"]], edges) - 1
        wt = w[stages["total"]]
        for i in range(len(ctr)):
            sel = bi == i
            if sel.any():
                neff_bin[i] = kish_n_eff(wt[sel])
        dd["neff_bin"] = neff_bin
        dens[kl] = dd

    # ---- bootstrap errors on R_shape (over events, n=uni size) ----
    print(f"[dir1] bootstrapping R ({args.n_boot} resamples)...")
    uidx = np.flatnonzero(uni)
    boot = {str(kl): {name: [] for name in stages} for kl in kls}
    stage_u = {name: m[uidx] for name, m in stages.items()}  # aligned to uidx
    w_u = {kl: event_weight_w(gen[uidx], kl) for kl in kls}
    t_u = {kl: event_score_t(gen[uidx], kl) for kl in kls}
    for _ in range(args.n_boot):
        ridx = rng.integers(0, len(uidx), len(uidx))
        for kl in kls:
            w, t = w_u[kl][ridx], t_u[kl][ridx]
            tot = shape_info_sum(w, t, stage_u["total"][ridx])
            for name in stages:
                sh = shape_info_sum(w, t, stage_u[name][ridx])
                boot[str(kl)][name].append(sh / tot if tot > 0 else np.nan)
    for kl in kls:
        for name in stages:
            per_kl[str(kl)][name]["R_shape_err"] = float(
                np.nanstd(boot[str(kl)][name]))
    results["info"] = per_kl

    # ---- acceptance curves in gen m_HH (Ullrich–Xu errors) ----
    h_tot, _ = np.histogram(gen[stages["total"]], edges)
    acc = {}
    for name in ("reco4j", "pt4", "ht330", "btag3", "presel", "presel+reco"):
        h_k, _ = np.histogram(gen[stages[name]], edges)
        eff, err = ullrich_xu_efficiency(h_k, h_tot)
        acc[name] = (eff, err)
    results["acceptance_mhh"] = {k: {"eff": v[0].tolist(), "err": v[1].tolist()}
                                 for k, v in acc.items()}

    # ---- gate numbers ----
    lo, hi = gen < 400, gen > 500
    surv = {}
    for name in ("presel", "reco4j"):
        m = stages[name]
        surv[name] = {
            "mhh_lt_400": float(m[stages["total"] & lo].sum()
                                / max(stages["total"][lo].sum(), 1)),
            "mhh_gt_500": float(m[stages["total"] & hi].sum()
                                / max(stages["total"][hi].sum(), 1)),
        }
    R1 = per_kl["1.0"]["presel"]["R_shape"]
    ratio = surv["presel"]["mhh_lt_400"] / max(surv["presel"]["mhh_gt_500"], 1e-12)
    if R1 < 0.3 and ratio < 0.5:
        verdict = ("ACCEPTANCE-BINDING: R_presel(kl=1) < 0.3 and low-mHH "
                   "survival < half of high-mHH — promote threshold relaxation "
                   "/ selection-observable-codesign to top physics priority")
    elif R1 > 0.6:
        verdict = ("BENIGN: R_presel(kl=1) > 0.6 — acceptance is an overall "
                   "efficiency; discrimination ceiling + B statistics remain "
                   "binding")
    else:
        comp_R = {n: per_kl["1.0"][n]["R_shape"]
                  for n in ("pt4", "ht330", "btag3")}
        worst = min(comp_R, key=comp_R.get)
        verdict = (f"INTERMEDIATE: R_presel(kl=1) = {R1:.3f}; dominant "
                   f"information-losing component = {worst} "
                   f"(component R: {comp_R}) — hand to "
                   f"selection-observable-codesign as first scan axis")
    results["gate"] = {"R_presel_kl1": R1, "survival": surv,
                       "low_high_survival_ratio": ratio, "verdict": verdict}
    print(f"\n[dir1] GATE: R_presel(kl=1) = {R1:.4f} ± "
          f"{per_kl['1.0']['presel']['R_shape_err']:.4f}; "
          f"survival(<400) = {surv['presel']['mhh_lt_400']:.4f}, "
          f"survival(>500) = {surv['presel']['mhh_gt_500']:.4f}")
    print(f"[dir1] VERDICT: {verdict}\n")

    # ---- eps(kl): acceptance κλ-dependence (exact parabola ratio) ----
    a, b, c = me2_coeffs_heft(gen[uni])
    den = np.maximum(a + b + c, 1e-12)
    klgrid = np.linspace(-1, 6, 141)
    eps_kl = {}
    A, B, C = a / den, b / den, c / den
    tot = A.sum() + klgrid * B.sum() + klgrid**2 * C.sum()
    for name, m in stages.items():
        mu = m[uni]  # stage mask restricted to the universe rows
        num = (A * mu).sum() + klgrid * (B * mu).sum() \
            + klgrid**2 * (C * mu).sum()
        eps_kl[name] = (num / tot).tolist()
    results["eps_vs_kl"] = {"kl_grid": klgrid.tolist(), **eps_kl}

    # ---- 2D (m_HH, |cosθ*|) primary map ----
    e2m = mhh_edges(n_lin=15, n_log=10)
    e2c = np.linspace(0, 1, 6)
    w1, t1 = event_weight_w(gen, 1.0), event_score_t(gen, 1.0)
    m = stages["total"]
    sw = w1[m].sum()
    tbar = (w1[m] * t1[m]).sum() / sw
    info2d, _, _ = np.histogram2d(gen[m], cos[m], [e2m, e2c],
                                  weights=w1[m] * (t1[m] - tbar) ** 2)
    n_tot2d, _, _ = np.histogram2d(gen[m], cos[m], [e2m, e2c])
    mp = stages["presel"]
    n_pre2d, _, _ = np.histogram2d(gen[mp], cos[mp], [e2m, e2c])
    with np.errstate(invalid="ignore", divide="ignore"):
        acc2d = np.where(n_tot2d > 0, n_pre2d / n_tot2d, np.nan)
    results["map2d"] = {"mhh_edges": e2m.tolist(), "cos_edges": e2c.tolist(),
                        "info_kl1": info2d.tolist(), "acc_presel": acc2d.tolist()}

    with open(f"{args.outdir}/results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"[dir1] wrote {args.outdir}/results.json")

    # =================== figures ===================
    grey = dens[1.0]["neff_bin"] < args.neff_grey

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
    d1 = dens[1.0]["total"] / max(dens[1.0]["total"].sum(), 1e-30)
    ax[0].step(ctr, d1, where="mid", color="C0", label=r"$dI_F/dm_{HH}$ ($\kappa_\lambda$=1)")
    if 2.45 in dens:
        d245 = dens[2.45]["total"] / max(dens[2.45]["total"].sum(), 1e-30)
        ax[0].step(ctr, d245, where="mid", color="C0", ls="--", alpha=0.7,
                   label=r"$dI_F/dm_{HH}$ ($\kappa_\lambda$=2.45, low $N_{eff}$)")
    for x0, x1, g in zip(edges[:-1], edges[1:], grey):
        if g:
            ax[0].axvspan(x0, x1, color="grey", alpha=0.15, lw=0)
    axb = ax[0].twinx()
    for name, color in [("reco4j", "C2"), ("presel", "C3")]:
        eff, err = acc[name]
        axb.errorbar(ctr, eff, err, color=color, fmt=".-", ms=3, lw=1,
                     label=rf"$\varepsilon$({name})")
    ax[0].set_xlabel(r"gen $m_{HH}$ [GeV]")
    ax[0].set_ylabel(r"normalized $\kappa_\lambda$ shape-info density", color="C0")
    axb.set_ylabel("acceptance", color="C3")
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, fontsize=8)
    ax[0].set_title(f"Dir-1: info vs acceptance  (R_presel(1)={R1:.3f}; "
                    f"grey: N_eff<{args.neff_grey:.0f})")

    for name, color in [("pt4", "C1"), ("ht330", "C4"), ("btag3", "C5"),
                        ("presel", "C3")]:
        eff, err = acc[name]
        ax[1].errorbar(ctr, eff, err, color=color, fmt=".-", ms=3, lw=1,
                       label=name)
    ax[1].set_xlabel(r"gen $m_{HH}$ [GeV]")
    ax[1].set_ylabel(r"acceptance $\varepsilon(m_{HH})$")
    ax[1].set_title("preselection components (masks on extended cache)")
    ax[1].legend(fontsize=8)
    fig.text(0.99, 0.01, CAVEAT, ha="right", fontsize=7, style="italic")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/info_acceptance_overlay.png", dpi=140)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    X, Y = np.meshgrid(e2m, e2c, indexing="ij")
    p0 = ax[0].pcolormesh(X, Y, info2d / max(info2d.sum(), 1e-30), cmap="viridis")
    fig.colorbar(p0, ax=ax[0], label=r"norm. shape-info density ($\kappa_\lambda$=1)")
    p1 = ax[1].pcolormesh(X, Y, acc2d, cmap="magma", vmin=0)
    fig.colorbar(p1, ax=ax[1], label=r"$\varepsilon$(presel)")
    for a_ in ax:
        a_.set_xlabel(r"gen $m_{HH}$ [GeV]")
        a_.set_ylabel(r"$|\cos\theta^*|$")
    ax[0].set_title("where the info lives")
    ax[1].set_title("what the preselection keeps")
    fig.text(0.99, 0.01, CAVEAT, ha="right", fontsize=7, style="italic")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/map2d_mhh_costheta.png", dpi=140)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name in ("reco4j", "presel", "presel+reco"):
        ax.plot(klgrid, eps_kl[name], label=name)
    ax.set_xlabel(r"$\kappa_\lambda$")
    ax.set_ylabel(r"$\varepsilon(\kappa_\lambda) = \Sigma_{kept} w / \Sigma_{all} w$")
    ax.set_title("acceptance κλ-dependence (unmodelled systematic + info source)")
    ax.legend(fontsize=8)
    fig.text(0.99, 0.01, CAVEAT, ha="right", fontsize=7, style="italic")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/eps_vs_kl.png", dpi=140)
    print(f"[dir1] wrote figures to {args.outdir}/")

    # ---- console R table ----
    print("\n[dir1] R_shape (info kept / total), rows = stages:")
    hdr = "  ".join(f"kl={kl:>5}" for kl in kls)
    print(f"  {'stage':14s} {hdr}")
    for name in stages:
        row = "  ".join(f"{per_kl[str(kl)][name]['R_shape']:.4f}" for kl in kls)
        print(f"  {name:14s} {row}")
    print("\n[dir1] Kish N_eff of reweighted total sample:")
    print("  " + "  ".join(f"kl={kl}: {per_kl[str(kl)]['total']['n_eff_kish']:.0f}"
                           for kl in kls))


if __name__ == "__main__":
    raise SystemExit(main())
