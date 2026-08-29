"""
Direction 3 — data-driven QCD background estimate: ABCD closure on QCD MC.

Vault plan: Experiments/l1-scouting-sbi/data-driven-qcd-background-estimate.md
Scheme 1, plane 1: ABCD in (event-ParT score, b-tag multiplicity) — the L1
analogue of the offline 3b->4b method — prototyped on QCD MC as a stand-in
for scouted data. Scheme 2 (3b->4b shape transfer fit) is the pre-registered
fallback if this non-closure is large; it is NOT run here.

Blind-protocol rehearsal: region boundaries are FIXED here, in code, before
any D yield is computed —
  category axis: n b-tagged jets (> selection WP) == 3  vs  >= 4
  score axis:    boundaries 0.5 and 0.8 (both reported)
  regions:       A=(low,3b)  B=(high,3b)  C=(low,4b)  D=(high,4b, signal)
  prediction:    D_hat = B * C / A

Per (selection, boundary): weighted yields, raw counts and Kish N_eff per
region; non-closure D_hat/D - 1 with Poisson-bootstrap error; even/odd MC-half
consistency; weighted correlation audit (score vs n_btag, score vs HT);
SM-strength signal-injection shift; top-1/top-5 weight-trim check; and the
error-scaling extrapolation to scouting sideband sizes. Selections are shared
with the Dir-2 scan (sbi/selection_scan.selection_mask), loosest-first per the
plan's controls.

Deliverable beyond the gate: the predicted B template in m_HH (8 pooled-
quantile bins, the closure_kl_v2.py binning convention) with per-bin errors,
for the profiled re-closure (oom-systematics-extension wiring is follow-up).

Usage: python sbi/abcd_qcd_closure.py --outdir autoresearch/dir3-abcd-closure
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

SCORE_BOUNDARIES = (0.5, 0.8)         # pre-registered, fixed before results
NB_SIGNAL_CAT = 4                     # >= 4 tags = "4b"; exactly 3 = "3b"


def region_ids(score, nb, boundary):
    """0=A(low,3b) 1=B(high,3b) 2=C(low,4b) 3=D(high,4b); -1 = not in plane."""
    cat3 = nb == 3
    cat4 = nb >= NB_SIGNAL_CAT
    hi = score >= boundary
    rid = np.full(len(score), -1, np.int8)
    rid[~hi & cat3] = 0
    rid[hi & cat3] = 1
    rid[~hi & cat4] = 2
    rid[hi & cat4] = 3
    return rid


def yields(rid, w):
    return np.array([w[rid == r].sum() for r in range(4)])


def nonclosure(y):
    a, b, c, dd = y
    if a <= 0 or dd <= 0:
        return np.nan, np.nan
    pred = b * c / a
    return pred, pred / dd - 1.0


def weighted_corr(x, y, w):
    xm = np.average(x, weights=w)
    ym = np.average(y, weights=w)
    cov = np.average((x - xm) * (y - ym), weights=w)
    vx = np.average((x - xm) ** 2, weights=w)
    vy = np.average((y - ym) ** 2, weights=w)
    return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/event_level/nsbi_cache.npz")
    ap.add_argument("--config", default="hh-bbbb-obj-config.json")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--outdir", default="autoresearch/dir3-abcd-closure")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(20260713)

    cfg = json.load(open(args.config))
    wp = float(cfg["l1ext"]["b_tag_cut"])
    phys = cfg["physics"]
    L = phys["luminosity_fb"]
    sig_w = phys["signal_xsec_pb"] * 1000.0 * L / phys["n_gen_signal"]

    selections = [
        ("loosest_pt0.6_noht_looseWP", 0.6, 0.0, 4, LOOSE_WP),
        ("pt0.6_noht_wp", 0.6, 0.0, 4, wp),
        ("pt0.6_ht230_wp", 0.6, 230.0, 4, wp),
        ("pt0.8_ht280_wp", 0.8, 280.0, 4, wp),
        ("operating_looseWP", 1.0, 330.0, 4, LOOSE_WP),
        ("operating", 1.0, 330.0, 4, wp),
    ]

    d = np.load(args.cache)
    meta = json.loads(str(d["meta"]))
    assert meta.get("skip_trigger", True), "needs the ambient cache"
    q_score = d["qcd_score"].astype(np.float64)
    q_ht = d["qcd_ht"].astype(np.float64)
    q_mhh = d["qcd_reco_mhh"].astype(np.float64)
    q_btag = np.nan_to_num(d["qcd_jet_btag"], nan=-np.inf)
    q_sigma = d["qcd_sigma"].astype(np.float64)
    n_loaded = {s: int((q_sigma == s).sum()) for s in np.unique(q_sigma)}
    q_yield = np.array([s * 1000.0 * L / n_loaded[s] for s in q_sigma])
    s_score = d["sig_score"].astype(np.float64)
    s_btag = np.nan_to_num(d["sig_jet_btag"], nan=-np.inf)

    results = {"boundaries": list(SCORE_BOUNDARIES), "sig_w": sig_w,
               "selections": {}}
    print(f"[dir3] ABCD plane (score, n_btag): boundaries {SCORE_BOUNDARIES}, "
          f"3b vs >={NB_SIGNAL_CAT}b; blind protocol: boundaries fixed in code")

    for name, pts, htc, nj, bwp in selections:
        mq = selection_mask(d, "qcd", pts, htc, nj, bwp)
        ms = selection_mask(d, "sig", pts, htc, nj, bwp)
        nb_q = (q_btag[mq] > bwp).sum(1)
        nb_s = (s_btag[ms] > bwp).sum(1)
        sc = q_score[mq]
        w = q_yield[mq]
        idx = np.flatnonzero(mq)
        corr_tag = weighted_corr(sc, nb_q.astype(float), w)
        corr_ht = weighted_corr(sc, q_ht[mq], w)
        sel_out = {"pt_scale": pts, "ht_cut": htc, "btag_wp": bwp,
                   "n_qcd_raw": int(mq.sum()), "n_sig_raw": int(ms.sum()),
                   "corr_score_nbtag": corr_tag, "corr_score_ht": corr_ht,
                   "boundaries": {}}

        for bnd in SCORE_BOUNDARIES:
            rid = region_ids(sc, nb_q, bnd)
            y = yields(rid, w)
            raw = np.array([(rid == r).sum() for r in range(4)])
            neff = np.array([kish_n_eff(w[rid == r]) for r in range(4)])
            pred, nc = nonclosure(y)

            # Poisson bootstrap on the QCD events -> sigma(nonclosure), sigma(pred)
            ncs, preds = [], []
            rid1 = (rid + 1).astype(np.int64)   # -1..3 -> 0..4 for bincount
            for _ in range(args.n_boot):
                p = rng.poisson(1.0, len(w))
                yb = np.bincount(rid1, weights=w * p, minlength=5)[1:]
                pb, nb_ = nonclosure(yb)
                preds.append(pb)
                ncs.append(nb_)
            sig_nc = float(np.nanstd(ncs))
            sig_pred = float(np.nanstd(preds))
            # direct-MC error on D (weighted Poisson): sqrt(sum w^2)
            sig_direct = float(np.sqrt((w[rid == 3] ** 2).sum()))

            # even/odd half consistency
            halves = {}
            for hname, hmask in (("even", idx % 2 == 0), ("odd", idx % 2 == 1)):
                yh = yields(rid[hmask], w[hmask])
                _, nch = nonclosure(yh)
                halves[hname] = float(nch) if np.isfinite(nch) else None

            # SM-strength signal injection
            rid_s = region_ids(s_score[ms], nb_s, bnd)
            ys = np.array([(rid_s == r).sum() * sig_w for r in range(4)])
            pred_cont, _ = nonclosure(y + ys)
            inj_shift = ((pred_cont - pred) / sig_pred
                         if np.isfinite(pred_cont) and sig_pred > 0 else np.nan)

            # top-weight trim check
            trims = {}
            for k in (1, 5):
                keep = np.argsort(w)[:-k] if len(w) > k else np.array([], int)
                _, nct = nonclosure(yields(rid[keep], w[keep]))
                trims[f"top{k}_removed"] = float(nct) if np.isfinite(nct) else None

            # transferability: sigma_B/B at sideband sizes (Kish-N scaling)
            rel2 = sum(1.0 / n for n in neff[:3] if n > 0)
            n_ctrl_raw = int(raw[:3].sum())
            transfer = {}
            for n_side in (1e6, 1e9):
                scale = n_ctrl_raw / n_side if n_side > 0 else np.nan
                transfer[f"sigma_B_over_B_at_{n_side:.0e}"] = float(
                    np.sqrt(rel2 * scale)) if rel2 > 0 else np.nan

            sel_out["boundaries"][str(bnd)] = {
                "yields_ABCD": y.tolist(), "raw_ABCD": raw.tolist(),
                "neff_ABCD": neff.tolist(),
                "pred_D": float(pred) if np.isfinite(pred) else None,
                "true_D": float(y[3]),
                "nonclosure": float(nc) if np.isfinite(nc) else None,
                "nonclosure_err_boot": sig_nc,
                "pred_err_boot": sig_pred, "direct_MC_err_D": sig_direct,
                "pred_beats_direct": bool(sig_pred < sig_direct),
                "halves_nonclosure": halves,
                "signal_injection_shift_sigma": (float(inj_shift)
                                                 if np.isfinite(inj_shift)
                                                 else None),
                "weight_trim_nonclosure": trims,
                "controls_neff_ge_500": bool((neff[:3] >= 500).all()),
                "transferability": transfer,
            }
            ncr = "nan" if not np.isfinite(nc) else f"{nc:+.3f}+/-{sig_nc:.3f}"
            print(f"[dir3] {name:28s} s*={bnd}: nc={ncr}  "
                  f"raw A/B/C/D {raw.tolist()}  neff "
                  f"{[f'{n:.0f}' for n in neff]}  corr(score,nb)={corr_tag:+.3f}")
        results["selections"][name] = sel_out

    # ---- gate evaluation on the loosest selection with valid controls ----
    gate_sel, gate_bnd, gate_blk = None, None, None
    for name, _, _, _, _ in selections:      # loosest-first order
        for bnd in SCORE_BOUNDARIES:
            blk = results["selections"][name]["boundaries"][str(bnd)]
            if blk["controls_neff_ge_500"] and blk["nonclosure"] is not None:
                gate_sel, gate_bnd, gate_blk = name, bnd, blk
                break
        if gate_sel:
            break
    if gate_blk is None:
        verdict = ("INCONCLUSIVE: no (selection, boundary) has all three "
                   "control regions at Kish N_eff >= 500 — MC stats cannot "
                   "distinguish 30% from 50% non-closure anywhere. Deliverable "
                   "= the error budget (transferability numbers) sizing H4.")
    else:
        nc, snc = abs(gate_blk["nonclosure"]), gate_blk["nonclosure_err_boot"]
        trim_ok = all(t is not None and abs(t - gate_blk["nonclosure"]) < max(2 * snc, 0.1)
                      for t in gate_blk["weight_trim_nonclosure"].values())
        if nc <= 0.30 and nc <= 2 * snc and gate_blk["pred_beats_direct"] and trim_ok:
            verdict = (f"METHOD ACCEPTED at {gate_sel} (s*={gate_bnd}): "
                       f"|nc| = {nc:.3f} <= 30%, within 2 sigma ({snc:.3f}), "
                       f"prediction error beats direct MC, trim-stable. Adopt "
                       f"data-driven template; demote H4 to method validation.")
        elif nc > 0.50 and nc > 2 * snc:
            verdict = (f"SCHEME-1 REJECTED at {gate_sel} (s*={gate_bnd}): "
                       f"|nc| = {nc:.3f} > 50% and > 2x its MC error — run "
                       f"Scheme 2 (3b->4b shape transfer) before the final "
                       f"method verdict; record corr(score, n_btag).")
        else:
            verdict = (f"INTERMEDIATE at {gate_sel} (s*={gate_bnd}): |nc| = "
                       f"{nc:.3f} +/- {snc:.3f}, trim_ok={trim_ok} — neither "
                       f"accept (30%) nor reject (50%) cleanly; report error "
                       f"budget and consider Scheme 2.")
    results["gate"] = {"selection": gate_sel, "boundary": gate_bnd,
                       "verdict": verdict}
    print(f"\n[dir3] GATE: {verdict}")

    # ---- B template in m_HH for the gate point (closure_kl_v2 convention) ----
    if gate_blk is not None:
        name = gate_sel
        srow = next(s for s in selections if s[0] == name)
        mq = selection_mask(d, "qcd", srow[1], srow[2], srow[3], srow[4])
        sc, w = q_score[mq], q_yield[mq]
        nb_q = (q_btag[mq] > srow[4]).sum(1)
        mh = q_mhh[mq]
        fin = np.isfinite(mh)
        rid = region_ids(sc, nb_q, gate_bnd)
        e_m = np.quantile(mh[fin], np.linspace(0, 1, 9))   # 8 pooled-quantile bins
        tf = (w[(rid == 1)].sum() / w[(rid == 0)].sum())   # global B/A transfer
        hC, _ = np.histogram(mh[fin & (rid == 2)], e_m,
                             weights=w[fin & (rid == 2)])
        hD, _ = np.histogram(mh[fin & (rid == 3)], e_m,
                             weights=w[fin & (rid == 3)])
        pred_bins = hC * tf
        boots = np.zeros((args.n_boot, 8))
        for i in range(args.n_boot):
            p = rng.poisson(1.0, len(w))
            wb = w * p
            tfb = wb[rid == 1].sum() / max(wb[rid == 0].sum(), 1e-300)
            hCb, _ = np.histogram(mh[fin & (rid == 2)], e_m,
                                  weights=wb[fin & (rid == 2)])
            boots[i] = hCb * tfb
        results["template"] = {
            "selection": name, "boundary": gate_bnd,
            "mhh_edges": e_m.tolist(), "pred_D_mhh": pred_bins.tolist(),
            "pred_err_mhh": np.std(boots, 0).tolist(),
            "true_D_mhh": hD.tolist(),
            "note": ("global B/A transfer x C-region m_HH shape; wire into "
                     "closure_kl_v2 via the Dir-5 nuisance machinery")}
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ctr = 0.5 * (e_m[:-1] + e_m[1:])
        ax.errorbar(ctr, pred_bins, np.std(boots, 0), fmt="o-", color="C0",
                    label=r"predicted $\hat{D}$ (C x B/A)")
        ax.step(ctr, hD, where="mid", color="C3", label="true D (direct MC)")
        ax.set_xlabel(r"reco $m_{HH}$ [GeV]")
        ax.set_ylabel(f"expected QCD events (L = {L}/fb)")
        ax.set_title(f"Dir-3 B template closure: {name}, score* = {gate_bnd}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{args.outdir}/template_closure.png", dpi=140)

    # non-closure summary figure
    fig, ax = plt.subplots(figsize=(9, 5))
    xt = []
    for i, (name, *_rest) in enumerate(selections):
        for j, bnd in enumerate(SCORE_BOUNDARIES):
            blk = results["selections"][name]["boundaries"][str(bnd)]
            if blk["nonclosure"] is None:
                continue
            ok = blk["controls_neff_ge_500"]
            ax.errorbar(i + 0.15 * j, blk["nonclosure"],
                        blk["nonclosure_err_boot"],
                        fmt="o" if ok else "x", color=f"C{j}", ms=6)
        xt.append(name)
    for lvl, ls in ((0.3, ":"), (0.5, "--"), (-0.3, ":"), (-0.5, "--")):
        ax.axhline(lvl, color="k", ls=ls, alpha=0.4)
    ax.axhline(0, color="k", lw=0.8)
    for j, bnd in enumerate(SCORE_BOUNDARIES):
        ax.plot([], [], "o", color=f"C{j}", label=f"score* = {bnd}")
    ax.plot([], [], "kx", label="controls N_eff < 500")
    ax.set_xticks(range(len(xt)))
    ax.set_xticklabels(xt, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("non-closure  (pred/true - 1)")
    ax.set_title("Dir-3 ABCD closure across selections (loosest first)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/nonclosure_summary.png", dpi=140)

    with open(f"{args.outdir}/results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"[dir3] wrote {args.outdir}/results.json + figures")


if __name__ == "__main__":
    raise SystemExit(main())
