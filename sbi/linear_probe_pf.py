"""
Task #12 belt-and-suspenders: frozen-backbone LINEAR PROBE on the preselected PF phase space.

Freezes the full-sample-trained PF event ParT (best_event_model_j01wrf4e.pth, full-val AUC 0.9775),
extracts its frozen CLS embedding (the 128-d vector fed to the classification head), and fits a
logistic regression on top. This isolates the question the fine-tune cannot fully answer:

  "Is there ANY linearly-accessible HH-vs-QCD signal in the frozen PF features that the head is
   failing to exploit?"

If even an optimally-regularized linear probe on the best backbone's features cannot beat the
~0.69 xsec-weighted AUC of zero-shot / fine-tuned PF, the ~0.84 PUPPI ceiling is unreachable with
PF inputs and the input-limited conclusion is unimpeachable.

Same seed-42 stratified val split (998 events) and xsec-weighted AUC metric as eval_presel_inmem.py
and the PUPPI 0.840 baseline, so the number is directly comparable. Probe trained with balanced
class weights (mirrors the fine-tune's pos_weight=0.446); a C-sweep guards against a mis-regularized
(under-powered) probe.

Usage:
  python sbi/linear_probe_pf.py --npz data/event_level/event_hh4b_qcd_presel_pf.npz \
      --config config_event_pf.json --ckpt best_event_model_j01wrf4e.pth
"""
import argparse
import json
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from sbi.build_nsbi_cache import load_model


def extract_embeddings(model, X, mask, device, bs=32):
    """Run the frozen model and capture the input to model.head (the CLS embedding)."""
    feats = []
    captured = {}

    def hook(_module, inp, _out):
        captured["emb"] = inp[0].detach().float().cpu().numpy()

    h = model.head.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).float().to(device)
            mb = torch.from_numpy(np.ascontiguousarray(mask[i:i + bs])).to(device)
            model(xb, particle_mask=mb)
            feats.append(captured["emb"])
    h.remove()
    return np.concatenate(feats)


def xsec_weights(yv, wv):
    w = np.ones(len(yv))
    qm = yv == 0
    if wv[qm].sum() > 0:
        w[qm] = wv[qm] / wv[qm].sum() * qm.sum()
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/event_level/event_hh4b_qcd_presel_pf.npz")
    ap.add_argument("--config", default="config_event_pf.json")
    ap.add_argument("--ckpt", default="best_event_model_j01wrf4e.pth")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = np.load(args.npz)
    X, mask = z["x"], z["mask"]
    y = z["y"].reshape(-1).astype(int)
    qcd_w = z["qcd_weights"].reshape(-1).astype(float)
    print(f"loaded {len(y)} events (sig {int((y==1).sum())}, qcd {int((y==0).sum())}) "
          f"x={X.shape}", flush=True)

    m = load_model(args.ckpt, cfg, device)
    print(f"extracting frozen CLS embeddings from {args.ckpt} ...", flush=True)
    emb = extract_embeddings(m, X, mask, device, bs=args.bs)
    print(f"embeddings: {emb.shape}", flush=True)

    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=args.val_split, stratify=y,
                              random_state=args.seed)
    tr, va = np.sort(tr), np.sort(va)
    print(f"train {len(tr)} (qcd {int((y[tr]==0).sum())}), "
          f"val {len(va)} (qcd {int((y[va]==0).sum())})", flush=True)

    scaler = StandardScaler().fit(emb[tr])
    Ztr, Zva = scaler.transform(emb[tr]), scaler.transform(emb[va])
    wva = xsec_weights(y[va], qcd_w[va])
    wtr = xsec_weights(y[tr], qcd_w[tr])

    print("\nC        train_AUC(unw)  val_AUC(unw)  val_AUC(xsec-wtd)", flush=True)
    best = (-1, None)
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        clf.fit(Ztr, y[tr])
        ptr = clf.predict_proba(Ztr)[:, 1]
        pva = clf.predict_proba(Zva)[:, 1]
        a_tr = roc_auc_score(y[tr], ptr, sample_weight=wtr)
        a_va_u = roc_auc_score(y[va], pva)
        a_va_w = roc_auc_score(y[va], pva, sample_weight=wva)
        print(f"{C:<8g} {a_tr:.4f}          {a_va_u:.4f}        {a_va_w:.4f}", flush=True)
        if a_va_w > best[0]:
            best = (a_va_w, C)

    print(f"\nBEST linear-probe xsec-weighted val AUC = {best[0]:.4f} (C={best[1]})", flush=True)
    print(f"compare: PF zero-shot 0.683, PF fine-tuned 0.687, PUPPI ceiling 0.840", flush=True)


if __name__ == "__main__":
    main()
