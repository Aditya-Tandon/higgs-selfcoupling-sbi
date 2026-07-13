"""
Evaluate the H3 fine-tuned event classifier vs the original, on the PRESELECTED phase space.

Fair comparison: both models scored on the SAME held-out val split (stratified_split seed 42,
val_split 0.2 — the split used during fine-tuning). Reports unweighted and xsec-weighted AUC.
The original model was trained on the full (un-preselected) dataset, so any gain here is the
benefit of focusing capacity on the hard, trigger-passing background.
"""
import argparse
import glob
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from data_pipeline.datasets import StratifiedJetDataset
from data_pipeline.splitting import stratified_split
from sbi.build_nsbi_cache import load_model


def score_all(model, X_mm, mask_mm, idx, device, bs=256):
    """Score the events at absolute dataset positions `idx`, reading rows from the
    memmaps per batch (avoids materializing the whole val subset — ~13.5 GB for PF)."""
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            xb = torch.from_numpy(np.ascontiguousarray(X_mm[j])).float().to(device)
            mb = torch.from_numpy(np.ascontiguousarray(mask_mm[j])).to(device)
            logit = model(xb, particle_mask=mb)["classification"].squeeze(-1)
            out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/event_level/presel_flat_npy/")
    ap.add_argument("--config", default="config_event_finetune.json")
    ap.add_argument("--old", default="", help="baseline checkpoint to also score (optional); "
                    "omit when the two models use different inputs (e.g. PF vs PUPPI)")
    ap.add_argument("--new", required=True, help="fine-tuned checkpoint .pth")
    ap.add_argument("--subsample", type=int, default=0,
                    help="score only this many val events (fixed seed 42); 0 = full val split")
    args = ap.parse_args()
    import json
    cfg = json.load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = StratifiedJetDataset(args.data)
    _, _, _, val_idx, _ = stratified_split(ds, 0.2, num_classes=1, random_state=42)
    val_idx = np.asarray(val_idx)
    if args.subsample and args.subsample < len(val_idx):
        rng = np.random.default_rng(42)
        val_idx = np.sort(rng.choice(val_idx, size=args.subsample, replace=False))
    # keep flat arrays as memmaps and index per batch inside score_all (no full materialization)
    xp = glob.glob(args.data + "/*_x.npy")[0]
    mp = glob.glob(args.data + "/*_mask.npy")[0]
    X_mm = np.load(xp, mmap_mode="r")
    mask_mm = np.load(mp, mmap_mode="r")
    meta = np.load(glob.glob(args.data + "/*_meta.npz")[0], allow_pickle=True)
    y = meta["y"].reshape(-1)[val_idx]
    qcd_w = meta["qcd_weights"][val_idx]
    print(f"val events: {len(y)} (signal {int((y==1).sum())}, qcd {int((y==0).sum())})")

    # xsec-weighted sample weights: within-class normalized so classes weigh equally,
    # but QCD events weighted by cross-section (the physically meaningful within-QCD weighting)
    w = np.ones(len(y))
    qm = y == 0
    if qcd_w[qm].sum() > 0:
        w[qm] = qcd_w[qm] / qcd_w[qm].sum() * qm.sum()

    ckpts = [("NEW", args.new)]
    if args.old:
        ckpts.insert(0, ("OLD (baseline)", args.old))
    for tag, ckpt in ckpts:
        m = load_model(ckpt, cfg, device)
        s = score_all(m, X_mm, mask_mm, val_idx, device)
        auc_u = roc_auc_score(y, s)
        auc_w = roc_auc_score(y, s, sample_weight=w)
        print(f"{tag:32s}: AUC unweighted={auc_u:.4f}  xsec-weighted={auc_w:.4f}  "
              f"(sig med {np.median(s[y==1]):.3f}, qcd med {np.median(s[y==0]):.3f})")


if __name__ == "__main__":
    main()
