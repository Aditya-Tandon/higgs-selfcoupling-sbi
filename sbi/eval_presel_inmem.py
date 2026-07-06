"""
Task #12: honest PF-vs-PUPPI discrimination lever test in the PRESELECTED regime.

Zero-shot eval of the full-sample-trained event classifier on the small preselected
(trigger-passing) event dataset, loaded fully in memory (~5k events — no memmap/OOM).

Methodology mirrors the H3 PUPPI baseline (sbi/eval_finetune.py) exactly so the PF number is
directly comparable to the PUPPI preselected ceiling (xsec-weighted AUC 0.840):
  * seed-42 stratified 80/20 split via sklearn train_test_split (same as stratified_split);
  * score on the held-out val split;
  * report unweighted AUC and xsec-weighted AUC (within-QCD weighting by qcd_weights).

Shared caveat (identical on both PF and PUPPI sides, so the DIFFERENCE is fair): the full model
trained on 80% of the full sample, and most preselected events fall in that train split, so the
absolute presel-val AUC is optimistic. The PUPPI 0.840 baseline has the same structure.

Usage:
  python sbi/eval_presel_inmem.py \
      --npz data/event_level/event_hh4b_qcd_presel_pf.npz \
      --config config_event_pf.json \
      --ckpt best_event_model_j01wrf4e.pth
"""
import argparse
import json
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
from sbi.build_nsbi_cache import load_model


def score(model, X, mask, device, bs=32):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).float().to(device)
            mb = torch.from_numpy(np.ascontiguousarray(mask[i:i + bs])).to(device)
            logit = model(xb, particle_mask=mb)["classification"].squeeze(-1)
            out.append(torch.sigmoid(logit).float().cpu().numpy())
    return np.concatenate(out)


def report(tag, y, s, w):
    auc_u = roc_auc_score(y, s)
    auc_w = roc_auc_score(y, s, sample_weight=w)
    print(f"{tag}: AUC unweighted={auc_u:.4f}  xsec-weighted={auc_w:.4f}  "
          f"(sig med {np.median(s[y == 1]):.3f}, qcd med {np.median(s[y == 0]):.3f})",
          flush=True)
    return auc_u, auc_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/event_level/event_hh4b_qcd_presel_pf.npz")
    ap.add_argument("--config", default="config_event_pf.json")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs", type=int, default=32,
                    help="batch size (small: 512-const ParT attention is O(N^2) in memory)")
    ap.add_argument("--full", action="store_true",
                    help="also report on the full preselected set (not just val split)")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = np.load(args.npz)
    X, mask = z["x"], z["mask"]
    y = z["y"].reshape(-1).astype(int)
    qcd_w = z["qcd_weights"].reshape(-1).astype(float)
    print(f"loaded {len(y)} events (sig {int((y == 1).sum())}, qcd {int((y == 0).sum())}) "
          f"x={X.shape} from {args.npz}", flush=True)

    idx = np.arange(len(y))
    _, val_idx = train_test_split(idx, test_size=args.val_split, stratify=y,
                                  random_state=args.seed)
    val_idx = np.sort(val_idx)
    print(f"val split: {len(val_idx)} events "
          f"(sig {int((y[val_idx] == 1).sum())}, qcd {int((y[val_idx] == 0).sum())})",
          flush=True)

    m = load_model(args.ckpt, cfg, device)

    def xsec_weights(yv, wv):
        w = np.ones(len(yv))
        qm = yv == 0
        if wv[qm].sum() > 0:
            w[qm] = wv[qm] / wv[qm].sum() * qm.sum()
        return w

    yv, qv = y[val_idx], qcd_w[val_idx]
    sv = score(m, X[val_idx], mask[val_idx], device, bs=args.bs)
    report("VAL  (held-out, H3-comparable)", yv, sv, xsec_weights(yv, qv))

    if args.full:
        sf = score(m, X, mask, device, bs=args.bs)
        report("FULL (all preselected)     ", y, sf, xsec_weights(y, qcd_w))


if __name__ == "__main__":
    main()
