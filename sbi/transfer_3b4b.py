"""
Direction 3 — Scheme 2: 3b->4b shape-transfer closure on QCD MC.

Vault plan: data-driven-qcd-background-estimate.md, Scheme 2 (the
pre-registered fallback after Scheme 1's large trim-stable non-closure,
[[data-driven-qcd-background-estimate-scheme1-result]]).

Method (Run-2 style, binned ratio): in the LOW-score sideband, fit the
transfer T(m_HH, HT) = 4b/3b yield ratio in (6 m_HH quantile bins + 1
"no-reco" category) x (4 HT quantile bins); apply T to the 3b high-score
events to predict the 4b high-score (signal-region) yield and m_HH shape.
Honest validation: fit T on the even MC half, close on the odd half (and
swapped). Errors by Poisson bootstrap (T refit per resample); top-weight
trim; region Kish N_eff. Same accept/reject gate thresholds as Scheme 1.

Usage: python sbi/transfer_3b4b.py --outdir autoresearch/dir3-transfer-3b4b
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
from sbi.fisher_info import kish_n_eff
from sbi.selection_scan import LOOSE_WP, selection_mask

SCORE_BOUNDARIES = (0.5, 0.8)   # same pre-registered boundaries as Scheme 1
N_MHH_BINS = 6                  # + 1 "no-reco" category
N_HT_BINS = 4


def cell_index(mhh, ht, e_m, e_h):
    """Transfer-cell id: (m_HH quantile bin, or the no-reco category) x HT
    quantile bin."""
    mb = np.where(np.isfinite(mhh),
                  np.clip(np.digitize(mhh, e_m) - 1, 0, N_MHH_BINS - 1),
                  N_MHH_BINS)                       # no-reco category
    hb = np.clip(np.digitize(ht, e_h) - 1, 0, N_HT_BINS - 1)
    return mb * N_HT_BINS + hb


def closure(cidx, region, w, n_cells):
    """Fit T = 4b/3b per cell in the low-score sideband and predict the
    high-score 4b yield from the high-score 3b events.
    region: 0=3b-lo 1=4b-lo 2=3b-hi 3=4b-hi. Returns (pred, true, dropped)."""
    sums = np.bincount(cidx * 4 + region, weights=w, minlength=n_cells * 4) \
        .reshape(n_cells, 4)
    ok = sums[:, 0] > 0
    T = np.where(ok, sums[:, 1] / np.maximum(sums[:, 0], 1e-300), 0.0)
    pred = float((sums[:, 2] * T)[ok].sum())
    dropped = float(sums[~ok, 2].sum())
    return pred, float(sums[:, 3].sum()), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/event_level/nsbi_cache.npz")
    ap.add_argument("--config", default="hh-bbbb-obj-config.json")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--outdir", default="autoresearch/dir3-transfer-3b4b")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(20260713)

    cfg = json.load(open(args.config))
    wp = float(cfg["l1ext"]["b_tag_cut"])
    L = cfg["physics"]["luminosity_fb"]

    selections = [
        ("loosest_pt0.6_noht_looseWP", 0.6, 0.0, 4, LOOSE_WP),
        ("operating_looseWP", 1.0, 330.0, 4, LOOSE_WP),
    ]

    d = np.load(args.cache)
    assert json.loads(str(d["meta"])).get("skip_trigger", True)
    q_score = d["qcd_score"].astype(np.float64)
    q_ht = d["qcd_ht"].astype(np.float64)
    q_mhh = d["qcd_reco_mhh"].astype(np.float64)
    q_btag = np.nan_to_num(d["qcd_jet_btag"], nan=-np.inf)
    q_sigma = d["qcd_sigma"].astype(np.float64)
    n_loaded = {s: int((q_sigma == s).sum()) for s in np.unique(q_sigma)}
    q_yield = np.array([s * 1000.0 * L / n_loaded[s] for s in q_sigma])

    n_cells = (N_MHH_BINS + 1) * N_HT_BINS
    results = {"boundaries": list(SCORE_BOUNDARIES),
               "n_cells": n_cells, "selections": {}}

    for name, pts, htc, nj, bwp in selections:
        mq = selection_mask(d, "qcd", pts, htc, nj, bwp)
        nb = (q_btag[mq] > bwp).sum(1)
        sc, ht, mh, w = q_score[mq], q_ht[mq], q_mhh[mq], q_yield[mq]
        idx = np.flatnonzero(mq)
        cat = np.where(nb >= 4, 1, np.where(nb == 3, 0, -1))
        sel_out = {"n_qcd_raw": int(mq.sum()), "boundaries": {}}

        for bnd in SCORE_BOUNDARIES:
            inplane = cat >= 0
            region = np.where(sc >= bnd, 2, 0) + cat   # 0/1 lo, 2/3 hi
            # transfer cells from the 3b low-score sideband
            fitsel = inplane & (region == 0)
            e_m = np.quantile(mh[fitsel & np.isfinite(mh)],
                              np.linspace(0, 1, N_MHH_BINS + 1))
            e_h = np.quantile(ht[fitsel], np.linspace(0, 1, N_HT_BINS + 1))
            cidx = cell_index(mh, ht, e_m, e_h)

            ip = inplane
            pred, true, dropped = closure(cidx[ip], region[ip], w[ip], n_cells)
            nc = pred / true - 1 if true > 0 else np.nan

            # Poisson bootstrap (T refit per resample)
            ncs = []
            for _ in range(args.n_boot):
                p = rng.poisson(1.0, ip.sum())
                pb, tb, _ = closure(cidx[ip], region[ip], w[ip] * p, n_cells)
                ncs.append(pb / tb - 1 if tb > 0 else np.nan)
            sig_nc = float(np.nanstd(ncs))

            # even/odd honest halves: fit+predict entirely within each half
            halves = {}
            for hname, hm in (("even", idx % 2 == 0), ("odd", idx % 2 == 1)):
                hsel = ip & hm
                ph, th, _ = closure(cidx[hsel], region[hsel], w[hsel], n_cells)
                halves[hname] = float(ph / th - 1) if th > 0 else None
            # cross-half: T from even sideband, closure on odd (and swap)
            cross = {}
            for fit_h, test_h, lab in ((idx % 2 == 0, idx % 2 == 1, "fit_even_close_odd"),
                                       (idx % 2 == 1, idx % 2 == 0, "fit_odd_close_even")):
                fsel = ip & fit_h & (sc < bnd)
                tsel = ip & test_h
                sums_f = np.bincount(cidx[fsel] * 2 + cat[fsel],
                                     weights=w[fsel], minlength=n_cells * 2) \
                    .reshape(n_cells, 2)
                okc = sums_f[:, 0] > 0
                T = np.where(okc, sums_f[:, 1] / np.maximum(sums_f[:, 0], 1e-300), 0.0)
                hi3 = tsel & (region == 2)
                hi4 = tsel & (region == 3)
                predx = float((np.bincount(cidx[hi3], weights=w[hi3],
                                           minlength=n_cells) * T)[okc].sum())
                truex = float(w[hi4].sum())
                cross[lab] = float(predx / truex - 1) if truex > 0 else None

            # top-weight trim
            trims = {}
            for k in (1, 5):
                keep = np.ones(ip.sum(), bool)
                keep[np.argsort(w[ip])[-k:]] = False
                pt_, tt_, _ = closure(cidx[ip][keep], region[ip][keep],
                                      w[ip][keep], n_cells)
                trims[f"top{k}_removed"] = (float(pt_ / tt_ - 1)
                                            if tt_ > 0 else None)

            neff = {r: kish_n_eff(w[ip][region[ip] == r]) for r in range(4)}
            sel_out["boundaries"][str(bnd)] = {
                "pred_D": pred, "true_D": true, "nonclosure": float(nc),
                "nonclosure_err_boot": sig_nc,
                "dropped_3bhi_yield_fraction": (dropped / (pred + dropped)
                                                if pred + dropped > 0 else 0.0),
                "halves_nonclosure": halves, "cross_half": cross,
                "weight_trim_nonclosure": trims,
                "neff_regions_3blo_4blo_3bhi_4bhi": neff,
                "controls_neff_ge_500": bool(all(neff[r] >= 500
                                                 for r in (0, 1, 2))),
            }
            print(f"[dir3-s2] {name:28s} s*={bnd}: nc={nc:+.3f}+/-{sig_nc:.3f} "
                  f"halves {halves} cross {cross} "
                  f"neff {[f'{neff[r]:.0f}' for r in range(4)]}")
        results["selections"][name] = sel_out

    # gate: same thresholds as Scheme 1, loosest-first
    gate_blk, gate_sel, gate_bnd = None, None, None
    for name, *_ in selections:
        for bnd in SCORE_BOUNDARIES:
            blk = results["selections"][name]["boundaries"][str(bnd)]
            if blk["controls_neff_ge_500"]:
                gate_blk, gate_sel, gate_bnd = blk, name, bnd
                break
        if gate_blk:
            break
    if gate_blk is None:
        # fall back to best-populated point for a provisional reading
        blk = results["selections"][selections[0][0]]["boundaries"][
            str(SCORE_BOUNDARIES[0])]
        verdict = (
            "INCONCLUSIVE (controls N_eff < 500 everywhere, as Scheme 1). "
            f"Provisional at best-populated point: nc = "
            f"{blk['nonclosure']:+.3f} +/- {blk['nonclosure_err_boot']:.3f} "
            f"vs Scheme-1 ABCD -0.58 — transfer "
            f"{'REDUCES' if abs(blk['nonclosure']) < 0.4 else 'does NOT fix'} "
            "the non-closure. Formal accept/reject still needs stats (H4 "
            "sizing from the Scheme-1 note stands).")
    else:
        nc, snc = abs(gate_blk["nonclosure"]), gate_blk["nonclosure_err_boot"]
        if nc <= 0.30 and nc <= 2 * snc:
            verdict = f"METHOD ACCEPTED at {gate_sel} s*={gate_bnd} (|nc|={nc:.3f})"
        elif nc > 0.50 and nc > 2 * snc:
            verdict = (f"SCHEME 2 ALSO REJECTED at {gate_sel} s*={gate_bnd} — "
                       f"with Scheme 1 this triggers the pre-registered full "
                       f"rejection: H4 (more MC) is the only path to trusted B")
        else:
            verdict = f"INTERMEDIATE at {gate_sel} s*={gate_bnd} (|nc|={nc:.3f}+/-{snc:.3f})"
    results["gate"] = {"verdict": verdict}
    print(f"\n[dir3-s2] GATE: {verdict}")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    xi = 0
    labels = []
    for name, *_ in selections:
        for bnd in SCORE_BOUNDARIES:
            blk = results["selections"][name]["boundaries"][str(bnd)]
            ax.errorbar(xi, blk["nonclosure"], blk["nonclosure_err_boot"],
                        fmt="o", color="C0", label="transfer" if xi == 0 else None)
            for lab, v in blk["cross_half"].items():
                if v is not None:
                    ax.plot(xi + 0.15, v, "s", color="C2", ms=4,
                            label="cross-half" if xi == 0 else None)
            labels.append(f"{name}\ns*={bnd}")
            xi += 1
    ax.axhline(-0.58, color="C3", ls="--", alpha=0.6,
               label="Scheme-1 ABCD (loosest, s*=0.5)")
    for lvl in (0.3, -0.3, 0.5, -0.5):
        ax.axhline(lvl, color="k", ls=":", alpha=0.3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("non-closure (pred/true - 1)")
    ax.set_title("Dir-3 Scheme 2: 3b->4b transfer closure")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/transfer_nonclosure.png", dpi=140)
    with open(f"{args.outdir}/results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"[dir3-s2] wrote {args.outdir}/results.json + figure")


if __name__ == "__main__":
    raise SystemExit(main())
