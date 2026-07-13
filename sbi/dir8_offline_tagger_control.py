"""
Direction 8 control study — is the b-tag <-> κλ-info anti-correlation physics
or a trained-model artifact?

Vault plan: Experiments/l1-scouting-sbi/jet-level-tagger-sbi-chain.md
("Control study (HPC)"). The laptop Stage-1 proto found corr(R, S/√B) = −0.50
scanning a min-b-tag cut with OUR trained jet-level tagger. Here the same scan
runs on the reference taggers stored in the extended cache:

  L1Ext_btagScore   L1puppiExtJetSC4 btagScore (−1 = untagged sentinel)
  Offline_PNet      offline Jet btagPNetB
  Offline_UParT     offline Jet btagUParTAK4B
  L1NG              L1puppiJetSC4NG bTagScore

Gate: anti-correlation persists across the offline taggers -> physics feature
(threshold-region jets intrinsically less b-taggable; "bin, don't cut"
generalizes). Doesn't persist -> artifact of our L1/trained models; better
tagging is a live lever. Record as dir8-offline-tagger-control-result.

Production upgrades over the proto: gen m_HH (not reco proxy) for the info
density; weighted-variance shape information (sbi/fisher_info.py, mean
subtracted) instead of the raw score^2 spectrum; per-tagger quantile cut grid
so every tagger scans its own score range.

Definitions (per tagger): universe = HH reco succeeded (finite reco_mhh,
mirrors the proto's di-Higgs-cache universe) AND >= 4 jets in that tagger's
collection. min-btag = min score over the 4 leading-pT jets (Higgs-candidate
proxy; secondary check: min over the 4 best-scored jets). S/√B uses honest
Convention-A yields: q_yield = sigma * 1000 * L / n_loaded(sigma), n_loaded
counted per unique sigma in this ambient (skip-trigger) cache where
kept == loaded. NOTE (P2/regime caveat): the scan runs on the ambient sample
exactly like the proto so the two are comparable; it is a control on the
proto's conclusion, not a lever ranking.

Usage: python sbi/dir8_offline_tagger_control.py --outdir autoresearch/dir8-tagger-control
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
                             shape_info_sum)

# cache field suffix -> (display name, collection, score branch)
TAGGERS = {
    "jet_btag": ("L1Ext_btagScore", "L1puppiExtJetSC4", "btagScore"),
    "ref_btagPNetB": ("Offline_PNet", "Jet", "btagPNetB"),
    "ref_btagUParTAK4B": ("Offline_UParT", "Jet", "btagUParTAK4B"),
    "l1ng_bTagScore": ("L1NG", "L1puppiJetSC4NG", "bTagScore"),
}
CAVEAT = "LO HEFT reweighting; ambient-regime scan (proto-comparable), honest yields"


def config_wp(cfg, collection, tagger):
    """Working point from the analysis config, if one is defined for exactly
    this (collection, score branch) pair."""
    for v in cfg.values():
        if (isinstance(v, dict) and v.get("collection_name") == collection
                and v.get("tagger_name") == tagger and "b_tag_cut" in v):
            return float(v["b_tag_cut"])
    return None


def min_btag(scores, leading_k=4):
    """Per-event min score over the 4 leading-pT jets (arrays are pT-sorted,
    NaN-padded). Rows with <4 jets come out NaN and fail every cut."""
    return scores[:, :leading_k].min(axis=1)


def min_of_best4(scores):
    """Secondary definition: min over the 4 best-scored jets (= 4th-highest
    score). NaN pads sink to -inf so they are never among the best 4; rows
    with <4 real jets come out -inf and are excluded by the caller."""
    y = np.where(np.isfinite(scores), scores, -np.inf)
    return np.partition(y, -4, axis=1)[:, -4]


def scan(cuts, s_mb, q_mb, w_by_kl, t_by_kl, sig_w, q_yield):
    """R(κλ) and S/√B across cuts on a fixed universe (arrays pre-masked)."""
    out = {"R": {k: [] for k in w_by_kl}, "SB": [], "S": [], "B": [],
           "n_qcd_raw": [], "B_neff": []}
    base = {k: shape_info_sum(w_by_kl[k], t_by_kl[k]) for k in w_by_kl}
    for c in cuts:
        ms = s_mb >= c
        mq = q_mb >= c
        for k in w_by_kl:
            sh = shape_info_sum(w_by_kl[k], t_by_kl[k], ms)
            out["R"][k].append(sh / base[k] if base[k] > 0 else np.nan)
        S = float(ms.sum()) * sig_w
        B = float(q_yield[mq].sum())
        out["S"].append(S)
        out["B"].append(B)
        out["SB"].append(S / np.sqrt(B) if B > 0 else 0.0)
        # weighted-MC-stats guards (handoff trap: one high-weight event can
        # fake an S/sqrtB peak): raw surviving MC count + Kish N_eff of B
        out["n_qcd_raw"].append(int(mq.sum()))
        out["B_neff"].append(kish_n_eff(q_yield[mq]))
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/event_level/nsbi_cache.npz")
    ap.add_argument("--config", default="hh-bbbb-obj-config.json")
    ap.add_argument("--kls", default="-1,0,1")
    ap.add_argument("--n-cuts", type=int, default=60)
    ap.add_argument("--outdir", default="autoresearch/dir8-tagger-control")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cfg = json.load(open(args.config))
    phys = cfg["physics"]
    L = phys["luminosity_fb"]
    sig_w = phys["signal_xsec_pb"] * 1000.0 * L / phys["n_gen_signal"]
    kls = [float(k) for k in args.kls.split(",")]

    d = np.load(args.cache)
    meta = json.loads(str(d["meta"]))
    assert meta.get("skip_trigger", True), \
        "control scan needs the ambient (skip-trigger) cache: kept == loaded"

    gen = d["sig_gen_mhh"].astype(np.float64)
    sig_reco = np.isfinite(d["sig_reco_mhh"]) & np.isfinite(gen)
    qcd_reco = np.isfinite(d["qcd_reco_mhh"])

    # honest per-event QCD yields (Convention A; ambient => kept == loaded)
    q_sigma = d["qcd_sigma"].astype(np.float64)
    n_loaded = {s: int((q_sigma == s).sum()) for s in np.unique(q_sigma)}
    q_yield_all = np.array([s * 1000.0 * L / n_loaded[s] for s in q_sigma])
    print(f"[dir8] QCD bins loaded: "
          f"{ {f'{s:.3g}': n for s, n in n_loaded.items()} }")

    results = {"luminosity_fb": L, "sig_w_per_event": sig_w, "kls": kls,
               "caveat": CAVEAT, "taggers": {}}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

    for ax, (field, (name, coll, score)) in zip(
            axes.ravel(), TAGGERS.items()):
        skey, qkey = f"sig_{field}", f"qcd_{field}"
        if skey not in d or qkey not in d:
            print(f"[dir8] {name}: {skey}/{qkey} missing from cache, skipped")
            ax.set_title(f"{name} (not in cache)")
            continue
        s_scores = d[skey]
        q_scores = d[qkey]
        s_mb_all = min_btag(s_scores)
        q_mb_all = min_btag(q_scores)
        su = sig_reco & np.isfinite(s_mb_all)   # >=4 jets in this collection
        qu = qcd_reco & np.isfinite(q_mb_all)
        s_mb = s_mb_all[su]
        q_mb = q_mb_all[qu]
        q_yield = q_yield_all[qu]
        genu = gen[su]

        w_by_kl = {kl: event_weight_w(genu, kl) for kl in kls}
        t_by_kl = {kl: event_score_t(genu, kl) for kl in kls}
        neff = {str(kl): kish_n_eff(w_by_kl[kl]) for kl in kls}

        cuts = np.quantile(s_mb, np.linspace(0.01, 0.99, args.n_cuts))
        cuts = np.unique(cuts)
        sc = scan(cuts, s_mb, q_mb, w_by_kl, t_by_kl, sig_w, q_yield)
        R1 = np.array(sc["R"][1.0])
        SB = np.array(sc["SB"])
        corr = float(np.corrcoef(R1, SB)[0, 1])
        iopt = int(np.argmax(SB))

        # robustness: same scan with the best-4-scored-jets definition
        s_mb2 = min_of_best4(s_scores)[su]
        q_mb2 = min_of_best4(q_scores)[qu]
        cuts2 = np.unique(np.quantile(s_mb2[np.isfinite(s_mb2)],
                                      np.linspace(0.01, 0.99, args.n_cuts)))
        sc2 = scan(cuts2, s_mb2, q_mb2, {1.0: w_by_kl[1.0]},
                   {1.0: t_by_kl[1.0]}, sig_w, q_yield)
        corr2 = float(np.corrcoef(np.array(sc2["R"][1.0]),
                                  np.array(sc2["SB"]))[0, 1])

        wp = config_wp(cfg, coll, score)
        wp_stats = None
        if wp is not None and cuts.min() <= wp <= cuts.max():
            iwp = int(np.argmin(np.abs(cuts - wp)))
            wp_stats = {"cut": wp, "R_kl1": float(R1[iwp]),
                        "SB": float(SB[iwp])}

        results["taggers"][name] = {
            "collection": coll, "score": score,
            "n_sig_universe": int(su.sum()), "n_qcd_universe": int(qu.sum()),
            "n_eff_kish": neff, "cuts": cuts.tolist(),
            "R": {str(k): v for k, v in sc["R"].items()},
            "S": sc["S"], "B": sc["B"], "SB": sc["SB"],
            "n_qcd_raw": sc["n_qcd_raw"], "B_neff": sc["B_neff"],
            "corr_R1_SB": corr, "corr_R1_SB_best4def": corr2,
            "SB_opt": {"cut": float(cuts[iopt]), "R_kl1": float(R1[iopt]),
                       "SB": float(SB[iopt]),
                       "n_qcd_raw": int(sc["n_qcd_raw"][iopt]),
                       "B_neff": float(sc["B_neff"][iopt])},
            "config_wp": wp_stats,
        }
        print(f"[dir8] {name:16s} corr(R1,S/sqrtB) = {corr:+.2f} "
              f"(best4 def: {corr2:+.2f})  S/sqrtB-opt cut {cuts[iopt]:.3f} "
              f"-> R1 {R1[iopt]:.3f}, S/sqrtB {SB[iopt]:.3g} "
              f"(B on {sc['n_qcd_raw'][iopt]} raw MC evts, "
              f"B_neff {sc['B_neff'][iopt]:.1f})  "
              f"[sig {su.sum()}, qcd {qu.sum()}]")

        for kl, ls in zip(kls, ("-.", "--", "-")):
            ax.plot(cuts, sc["R"][kl], "C0", ls=ls, lw=1.4,
                    label=rf"$R$ ($\kappa_\lambda$={kl:g})")
        axb = ax.twinx()
        axb.plot(cuts, SB, "C3", label=r"$S/\sqrt{B}$")
        axb.axvline(cuts[iopt], color="C3", ls=":", alpha=0.6)
        if wp is not None:
            ax.axvline(wp, color="k", ls=":", label=f"config WP {wp:.2f}")
        ax.set_title(f"{name}  corr(R1, S/sqrtB) = {corr:+.2f}")
        ax.set_xlabel("min-b-tag cut (4 leading-pT jets)")
        ax.set_ylabel(r"$R$ ($\kappa_\lambda$ shape info retained)", color="C0")
        axb.set_ylabel(r"$S/\sqrt{B}$", color="C3")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axb.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")

    corrs = {n: v["corr_R1_SB"] for n, v in results["taggers"].items()}
    offline = [v for n, v in corrs.items() if n.startswith("Offline")]
    persists = bool(offline) and all(c < -0.2 for c in offline)
    results["gate"] = {
        "corr_by_tagger": corrs,
        "anti_correlation_persists_offline": persists,
        "reading": ("PHYSICS FEATURE: anti-correlation persists in the offline "
                    "taggers -> 'bin b-tag, don't cut' generalizes; threshold-"
                    "region kl sensitivity is b-tag-limited, not model-limited"
                    if persists else
                    "MODEL ARTIFACT (or absent): offline taggers do not show "
                    "the anti-correlation -> improved tagging is a live lever"),
    }
    print(f"\n[dir8] GATE: corr by tagger {corrs}")
    print(f"[dir8] {results['gate']['reading']}")

    fig.suptitle("Dir-8 control: min-b-tag cut trade-off across reference taggers",
                 y=0.995)
    fig.text(0.99, 0.005, CAVEAT, ha="right", fontsize=7, style="italic")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/tagger_control_scan.png", dpi=140)
    with open(f"{args.outdir}/results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"[dir8] wrote {args.outdir}/results.json + tagger_control_scan.png")


if __name__ == "__main__":
    raise SystemExit(main())
