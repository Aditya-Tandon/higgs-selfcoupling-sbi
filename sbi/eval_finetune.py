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


def score_all(model, X, mask, device, bs=512):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).float().to(device)
            mb = torch.from_numpy(mask[i:i + bs]).to(device)
            logit = model(xb, particle_mask=mb)["classification"].squeeze(-1)
            out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/event_level/presel_flat_npy/")
    ap.add_argument("--config", default="config_event_finetune.json")
    ap.add_argument("--old", default="best_event_model_61z973dk.pth")
    ap.add_argument("--new", required=True, help="fine-tuned checkpoint .pth")
    args = ap.parse_args()
    import json
    cfg = json.load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = StratifiedJetDataset(args.data)
    _, _, _, val_idx, _ = stratified_split(ds, 0.2, num_classes=1, random_state=42)
    val_idx = np.asarray(val_idx)
    # load flat arrays directly (dataset stores tensors/memmaps not directly np-indexable)
    xp = glob.glob(args.data + "/*_x.npy")[0]
    mp = glob.glob(args.data + "/*_mask.npy")[0]
    X = np.load(xp, mmap_mode="r")[val_idx]
    mask = np.load(mp, mmap_mode="r")[val_idx]
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

    for tag, ckpt in [("OLD (full-trained)", args.old), ("NEW (preselected fine-tune)", args.new)]:
        m = load_model(ckpt, cfg, device)
        s = score_all(m, X, mask, device)
        auc_u = roc_auc_score(y, s)
        auc_w = roc_auc_score(y, s, sample_weight=w)
        print(f"{tag:32s}: AUC unweighted={auc_u:.4f}  xsec-weighted={auc_w:.4f}  "
              f"(sig med {np.median(s[y==1]):.3f}, qcd med {np.median(s[y==0]):.3f})")


if __name__ == "__main__":
    main()
