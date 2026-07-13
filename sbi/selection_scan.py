"""
Direction 2, Stage 0 + Stage 1 — selection scan at fixed observable.

Vault plan: Experiments/l1-scouting-sbi/selection-observable-codesign.md
Promoted to top priority by the Dir-1 gate (ACCEPTANCE-BINDING,
kl-acceptance-vs-information-map: R_presel(kl=1) = 0.031).

Stage 0 (go/no-go): the stored-jet pT floor, read off the extended cache's
per-jet pT arrays (leading-10 L1Ext jets per event) — no ROOT pass needed.
If storage truncates at/above a scanned threshold, that grid arm saturates
and is annotated (not silently scanned).

Stage 1: the full (jet-pT scale x HT x n-jet x b-tag WP) grid — 3x4x2x3 = 72
points — as masks on the extended ambient cache. Per grid point:
  - physics-weighted S/sqrtB at L = 1000/fb, optimized over the event-ParT
    score threshold (global optimum + "reliable" optimum restricted to
    thresholds where B rests on >= 20 raw MC events);
  - Poisson-bootstrap error band on the optimum (grid-argmax rides B noise);
  - surviving raw MC count per QCD sigma-bin (the Dir-3 coupling + sized-MC
    deliverable);
  - post-selection kl Fisher information retention R(kl) (same estimator as
    Dir-1, sbi/fisher_info.py) + signal Kish N_eff at kl = 2.45, 5
    (Fisher axis unreliable below ~500).

The baseline point (pt x1.0, HT>330, 4 jets, current WP) IS the trigger
emulation — validated row-exact against nsbi_cache_trig.npz in Dir-1 — so the
gate comparison is internally consistent.

Pre-registered gate (evaluated + printed, decision in the -result note):
  win        SB_opt >= 1.5x baseline with >= 20 raw MC events, R >= baseline,
             margin exceeding the bootstrap band
  acc-win    R >= 2x baseline at SB within 1.5x
  null       nothing beats baseline outside MC-stat error
Pitfall honored: the score is selection-correlated (trained on the full phase
space) — Stage 1 SHORTLISTS loose selections, it cannot reject them.

Usage: python sbi/selection_scan.py --outdir autoresearch/dir2-selection-scan
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from sbi.fisher_info import event_score_t, event_weight_w, kish_n_eff, shape_info_sum

PT_BASE = (75.0, 60.0, 45.0, 40.0)   # trigger-emulation leading-jet thresholds
PT_SCALES = (1.0, 0.8, 0.6)
HT_CUTS = (330.0, 280.0, 230.0, 0.0)  # 0 = no HT cut
NJETS = (4, 3)
LOOSE_WP = 0.37053                    # make_event_dataset.py default (loose arm)
N_BTAG_REQUIRED = 3                   # fixed, as in the trigger emulation
SCORE_GRID = np.linspace(0.0, 0.99, 100)


def selection_mask(d, prefix, pt_scale, ht_cut, njet, btag_wp,
                   n_btag_required=N_BTAG_REQUIRED):
    """Trigger-emulation-style mask on the extended cache (jet arrays are
    pT-sorted, NaN-padded; ht/n_jets cover ALL jets). btag_wp=None drops the
    b-tag requirement entirely."""
    jp = d[f"{prefix}_jet_pt"]
    m = np.ones(len(jp), bool)
    for i in range(njet):
        m &= np.nan_to_num(jp[:, i], nan=-1.0) > PT_BASE[i] * pt_scale
    if ht_cut > 0:
        m &= d[f"{prefix}_ht"] > ht_cut
    if btag_wp is not None:
        nb = (np.nan_to_num(d[f"{prefix}_jet_btag"], nan=-np.inf) > btag_wp).sum(1)
        m &= nb >= n_btag_required
    return m


def cum_from_top(values, weights=None):
    """Yield (or count) at score >= SCORE_GRID[i], via reverse cumsum."""
    bins = np.append(SCORE_GRID, np.inf)
    h, _ = np.histogram(values, bins=bins, weights=weights)
    return h[::-1].cumsum()[::-1]


def score_optimum(s_scores, q_scores, q_yield, sig_w, min_raw):
    """Best S/sqrtB over the score-threshold grid, restricted to thresholds
    where B rests on >= min_raw raw MC events. Returns None if none qualify."""
    S = cum_from_top(s_scores) * sig_w
    B = cum_from_top(q_scores, q_yield)
    raw = cum_from_top(q_scores)
    ok = (raw >= min_raw) & (B > 0)
    if not ok.any():
        return None
    sb = np.where(ok, S / np.sqrt(np.maximum(B, 1e-300)), -np.inf)
    i = int(np.argmax(sb))
    return {"score_cut": float(SCORE_GRID[i]), "S": float(S[i]), "B": float(B[i]),
            "SB": float(sb[i]), "n_qcd_raw": int(raw[i])}


def bootstrap_sb_opt(s_scores, q_scores, q_sigma_idx, yield_per_sigma, sig_w,
                     min_raw, n_boot, rng):
    """Poisson bootstrap of the optimized S/sqrtB. Exactly equivalent to a
    per-event Poisson bootstrap but binned: within a (score-bin, sigma-bin)
    cell every QCD event carries the same yield, so resampling the ~100x9
    cell counts loses nothing and is ~1000x faster over the 72-point grid."""
    nsb = len(SCORE_GRID)
    bins = np.append(SCORE_GRID, np.inf)
    s_cnt = np.histogram(s_scores, bins)[0].astype(np.float64)
    qb = np.clip(np.digitize(q_scores, bins) - 1, 0, nsb - 1)
    q_cnt = np.bincount(qb * len(yield_per_sigma) + q_sigma_idx,
                        minlength=nsb * len(yield_per_sigma)) \
        .reshape(nsb, -1).astype(np.float64)
    vals = []
    for _ in range(n_boot):
        s_r = rng.poisson(s_cnt)
        q_r = rng.poisson(q_cnt)
        S = s_r[::-1].cumsum()[::-1] * sig_w
        B = (q_r * yield_per_sigma).sum(1)[::-1].cumsum()[::-1]
        raw = q_r.sum(1)[::-1].cumsum()[::-1]
        ok = (raw >= min_raw) & (B > 0)
        vals.append(np.max(np.where(ok, S / np.sqrt(np.maximum(B, 1e-300)),
                                    -np.inf)) if ok.any() else np.nan)
    vals = np.asarray(vals, float)
    return (float(np.nanstd(vals)),
            [float(np.nanpercentile(vals, 16)), float(np.nanpercentile(vals, 84))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/event_level/nsbi_cache.npz")
    ap.add_argument("--config", default="hh-bbbb-obj-config.json")
    ap.add_argument("--kls", default="-1,0,1,2.45,5")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--min-raw-reliable", type=int, default=20)
    ap.add_argument("--outdir", default="autoresearch/dir2-selection-scan")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(20260713)

    cfg = json.load(open(args.config))
    wp_current = float(cfg["l1ext"]["b_tag_cut"])
    phys = cfg["physics"]
    L = phys["luminosity_fb"]
    sig_w = phys["signal_xsec_pb"] * 1000.0 * L / phys["n_gen_signal"]
    kls = [float(k) for k in args.kls.split(",")]
    btag_arms = (("wp", wp_current), ("loose", LOOSE_WP), ("none", None))

    d = np.load(args.cache)
    meta = json.loads(str(d["meta"]))
    assert meta.get("skip_trigger", True), "scan needs the ambient cache"

    # ---------------- Stage 0: stored-jet pT floor ----------------
    all_jp = np.concatenate([d["sig_jet_pt"].ravel(), d["qcd_jet_pt"].ravel()])
    all_jp = all_jp[np.isfinite(all_jp)]
    floor = float(all_jp.min())
    p01 = float(np.percentile(all_jp, 0.1))
    thr_min_scanned = min(PT_BASE) * min(PT_SCALES)   # 24 GeV at -40%
    saturated_scales = [s for s in PT_SCALES if min(PT_BASE) * s <= floor]
    stage0 = {"floor_gev": floor, "p0p1_gev": p01,
              "lowest_scanned_threshold_gev": thr_min_scanned,
              "saturated_pt_scales": saturated_scales,
              "verdict": ("pT axis OPEN down to the -40% arm"
                          if not saturated_scales else
                          f"pT arms {saturated_scales} saturate at the "
                          f"storage floor {floor:.1f} GeV — annotated, kept")}
    print(f"[dir2] STAGE 0: stored L1Ext jet-pT floor = {floor:.2f} GeV "
          f"(0.1th pct {p01:.2f}); lowest scanned threshold "
          f"{thr_min_scanned:.0f} GeV -> {stage0['verdict']}")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.hist(all_jp, bins=np.linspace(0, 150, 151), histtype="step", log=True)
    for s in PT_SCALES:
        ax.axvline(min(PT_BASE) * s, color="C3", ls=":", alpha=0.7)
    ax.axvline(floor, color="k", ls="--", label=f"storage floor {floor:.1f} GeV")
    ax.set_xlabel("stored L1Ext jet pT [GeV] (leading-10, sig+QCD)")
    ax.set_ylabel("jets")
    ax.set_title("Dir-2 Stage 0: jet-pT storage floor vs scanned thresholds")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/stage0_pt_floor.png", dpi=140)

    # ---------------- Stage 1: the 72-point grid ----------------
    gen = d["sig_gen_mhh"].astype(np.float64)
    uni = np.isfinite(gen)
    gen = np.where(uni, gen, 350.0)   # NaN-safety for the reweight floor
    s_scores_all = d["sig_score"]
    q_scores_all = d["qcd_score"]
    q_sigma = d["qcd_sigma"].astype(np.float64)
    sigma_vals = np.unique(q_sigma)
    n_loaded = {s: int((q_sigma == s).sum()) for s in sigma_vals}
    yield_per_sigma = np.array([s * 1000.0 * L / n_loaded[s] for s in sigma_vals])
    q_sigma_idx_all = np.searchsorted(sigma_vals, q_sigma)
    q_yield_all = yield_per_sigma[q_sigma_idx_all]

    w_kl = {kl: event_weight_w(gen, kl) for kl in kls}
    t_kl = {kl: event_score_t(gen, kl) for kl in kls}
    shape_tot = {kl: shape_info_sum(w_kl[kl], t_kl[kl], uni) for kl in kls}

    points = []
    grid = list(itertools.product(PT_SCALES, HT_CUTS, NJETS, btag_arms))
    print(f"[dir2] STAGE 1: scanning {len(grid)} grid points "
          f"(baseline = pt1.0_ht330_nj4_wp)...")
    for pt_scale, ht_cut, njet, (bname, bwp) in grid:
        name = f"pt{pt_scale}_ht{ht_cut:.0f}_nj{njet}_btag-{bname}"
        ms = selection_mask(d, "sig", pt_scale, ht_cut, njet, bwp)
        mq = selection_mask(d, "qcd", pt_scale, ht_cut, njet, bwp)
        s_sc = s_scores_all[ms]
        q_sc = q_scores_all[mq]
        q_yl = q_yield_all[mq]
        raw_per_bin = {f"{s:.3g}": int(((q_sigma == s) & mq).sum())
                       for s in n_loaded}

        opt = score_optimum(s_sc, q_sc, q_yl, sig_w, min_raw=1)
        opt_rel = score_optimum(s_sc, q_sc, q_yl, sig_w,
                                min_raw=args.min_raw_reliable)
        sb_err, sb_band = ((np.nan, [np.nan, np.nan]) if opt is None else
                           bootstrap_sb_opt(s_sc, q_sc, q_sigma_idx_all[mq],
                                            yield_per_sigma, sig_w, 1,
                                            args.n_boot, rng))
        R = {str(kl): (shape_info_sum(w_kl[kl], t_kl[kl], ms & uni)
                       / shape_tot[kl]) for kl in kls}
        neff = {str(kl): kish_n_eff(w_kl[kl][ms & uni]) for kl in (2.45, 5.0)
                if kl in kls}
        points.append({
            "name": name, "pt_scale": pt_scale, "ht_cut": ht_cut, "njet": njet,
            "btag_arm": bname, "btag_wp": bwp,
            "n_sig_raw": int(ms.sum()), "n_qcd_raw": int(mq.sum()),
            "S_nocut": float(ms.sum() * sig_w), "B_nocut": float(q_yl.sum()),
            "qcd_raw_per_sigma_bin": raw_per_bin,
            "opt": opt, "opt_reliable": opt_rel,
            "SB_opt_bootstrap_std": sb_err, "SB_opt_band_16_84": sb_band,
            "R_shape": R, "sig_neff": neff,
            "fisher_unreliable_beyond_kl1": any(v < 500 for v in neff.values()),
        })
        ovals = "none" if opt is None else (
            f"SB {opt['SB']:.4f}+/-{sb_err:.4f} @score>{opt['score_cut']:.2f} "
            f"(B on {opt['n_qcd_raw']} raw)")
        print(f"[dir2]  {name:32s} sig {ms.sum():6d} qcd {mq.sum():6d}  "
              f"{ovals}  R1 {R['1.0']:.4f}")

    # ---------------- gate evaluation ----------------
    base = next(p for p in points if p["name"] == "pt1.0_ht330_nj4_btag-wp")
    sb_base = base["opt"]["SB"]
    r_base = base["R_shape"]["1.0"]
    winners, acc_wins = [], []
    for p in points:
        o = p["opt_reliable"]
        if o is None or p["name"] == base["name"]:
            continue
        margin_ok = (o["SB"] - p["SB_opt_bootstrap_std"]) >= 1.5 * sb_base
        if (o["SB"] >= 1.5 * sb_base and o["n_qcd_raw"] >= args.min_raw_reliable
                and p["R_shape"]["1.0"] >= r_base and margin_ok):
            winners.append(p["name"])
        if (p["R_shape"]["1.0"] >= 2 * r_base
                and p["opt"] is not None and p["opt"]["SB"] >= sb_base / 1.5):
            acc_wins.append(p["name"])
    if winners:
        verdict = f"CO-DESIGN WINS (Stage-1 shortlist): {winners}"
    elif acc_wins:
        verdict = (f"ACCEPTANCE-SIDE WIN (R >= 2x baseline at comparable "
                   f"S/sqrtB): {acc_wins} — hand to the Fisher-vs-NRE arms; "
                   f"shortlist for Stage 2")
    else:
        verdict = ("NULL at Stage 1: no point beats baseline outside MC-stat "
                   "error at the FIXED full-sample observable. Per the "
                   "pre-registered pitfall this cannot reject loose "
                   "selections — Stage 2 (retrained observable) decides.")
    gate = {"baseline": {"name": base["name"], "SB_opt": sb_base,
                         "n_qcd_raw_at_opt": base["opt"]["n_qcd_raw"],
                         "R_kl1": r_base,
                         "external_baseline_H1": 0.0477},
            "winners": winners, "acceptance_side": acc_wins,
            "verdict": verdict}
    print(f"\n[dir2] baseline SB_opt = {sb_base:.4f} (H1 reference 0.0477), "
          f"R1 = {r_base:.4f}")
    print(f"[dir2] GATE (Stage-1): {verdict}")

    out = {"stage0": stage0, "gate": gate, "grid": points,
           "score_grid": SCORE_GRID.tolist(), "sig_w": sig_w,
           "wp_current": wp_current, "wp_loose": LOOSE_WP,
           "note": ("score is selection-correlated (full-sample training): "
                    "Stage 1 shortlists, never rejects, loose selections")}
    with open(f"{args.outdir}/results.json", "w") as f:
        json.dump(out, f, indent=1, default=float)

    # Pareto view: R(kl=1) vs SB_opt
    fig, ax = plt.subplots(figsize=(8, 5.6))
    for p in points:
        if p["opt"] is None:
            continue
        reliable = (p["opt_reliable"] is not None
                    and p["opt"]["n_qcd_raw"] >= args.min_raw_reliable)
        ax.errorbar(p["R_shape"]["1.0"], p["opt"]["SB"],
                    yerr=p["SB_opt_bootstrap_std"],
                    fmt="o" if reliable else "x",
                    color={"wp": "C0", "loose": "C1", "none": "C2"}[p["btag_arm"]],
                    ms=5, alpha=0.85)
    ax.errorbar(r_base, sb_base, yerr=base["SB_opt_bootstrap_std"], fmt="*",
                color="k", ms=16, label="baseline (trigger emulation)")
    ax.axhline(1.5 * sb_base, color="k", ls=":", alpha=0.6,
               label="gate: 1.5x baseline SB")
    ax.axvline(2 * r_base, color="k", ls="--", alpha=0.4,
               label="gate: 2x baseline R")
    for c, lab in (("C0", "btag WP current"), ("C1", "btag loose"),
                   ("C2", "btag none")):
        ax.plot([], [], "o", color=c, label=lab)
    ax.plot([], [], "kx", label="B on < 20 raw MC (unreliable)")
    ax.set_xlabel(r"R($\kappa_\lambda$=1): shape info retained")
    ax.set_ylabel(r"optimized $S/\sqrt{B}$ (L = 1000/fb)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Dir-2 Stage 1: selection grid, information vs significance")
    ax.legend(fontsize=7)
    fig.text(0.99, 0.01, "fixed full-sample observable; Stage-1 shortlist only",
             ha="right", fontsize=7, style="italic")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/pareto_R_vs_SB.png", dpi=140)
    print(f"[dir2] wrote {args.outdir}/results.json + figures")


if __name__ == "__main__":
    raise SystemExit(main())
