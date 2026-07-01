"""
Build a self-contained, row-aligned NSBI cache for iteration-2 κλ inference.

For every event it stores the *real* iteration-2 observables + everything the
extended likelihood needs, all aligned by construction (no cross-dataset joins):

  signal (HH->4b):  score, reco_mhh, reco_lead_m, reco_sub_m, gen_mhh, cos_star
  QCD:              score, reco_mhh, reco_lead_m, reco_sub_m, qcd_sigma (raw σ_bin)

- constituents come from the L1ExtPuppi collection via load_event_level_data (same
  path as make_event_dataset.py), features via extract_event_features;
- score = sigmoid of the event-level ParticleTransformer (local checkpoint);
- reco_mhh via evaluation.dihiggs.reconstruct_dihiggs_from_constituents;
- gen (m_HH, cos θ*) for signal from the SAME sorted ROOT file list (uproot),
  so signal rows line up with the constituents 1:1.

GPU recommended (model forward); clustering runs on CPU. Cap QCD with --qcd-max-per-bin.
Usage: python sbi/build_nsbi_cache.py --out data/event_level/nsbi_cache.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import sys

import numpy as np
import awkward as ak
import torch
import uproot
from tqdm import tqdm

sys.path.insert(0, ".")
from data_pipeline.root_loading import load_event_level_data
from data_pipeline.make_event_dataset import (extract_event_features,
                                              passes_trigger_emulation)
from evaluation.dihiggs import reconstruct_dihiggs_from_constituents
from model.parT import ParticleTransformer
from sbi.kl_reweight import gen_higgs_kinematics

GEN_BRANCHES = ["GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass",
                "GenPart_pdgId", "GenPart_statusFlags"]


def local_glob(config_path, data_root="data"):
    """Remap a config file_pattern (originally Mac paths) to the HPC data/ tree:
    '/Users/.../data/hh4b/data_*.root' -> 'data/hh4b/data_*.root'."""
    import os
    return os.path.join(data_root, os.path.basename(os.path.dirname(config_path)),
                        os.path.basename(config_path))


def load_model(ckpt_path, cfg, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = ParticleTransformer(
        input_dim=cfg["input_dim"], embed_dim=cfg["model"]["embed_dim"],
        num_pairwise_feat=cfg["model"].get("num_pairwise_feat", 7),
        num_heads=cfg["model"]["num_heads"], num_layers=cfg["model"]["num_layers"],
        num_cls_layers=cfg["model"]["num_cls_layers"], dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
        use_batch_norm=cfg["model"].get("use_batch_norm", True),
        pt_regression=cfg["model"].get("pt_regression", False),
        quantile_regression=cfg["model"].get("quantile_regression", False),
    ).to(device)
    m.load_state_dict(ckpt.get("model_state_dict", ckpt))
    m.eval()
    print(f"[cache] model loaded (epoch {ckpt.get('epoch')}, "
          f"val_auc {ckpt.get('val_auc')})", flush=True)
    return m


def extract_features_block(puppi_events, num_constituents):
    n = len(puppi_events)
    X = np.zeros((n, num_constituents, 17), dtype=np.float32)
    M = np.zeros((n, num_constituents), dtype=bool)
    for i in tqdm(range(n), desc="  features"):
        x, m = extract_event_features(puppi_events[i], num_constituents)
        X[i], M[i] = x, m
    return X, M


@torch.no_grad()
def score_block(model, X, M, device, batch=1024):
    out = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        mb = torch.from_numpy(M[i:i + batch]).to(device)
        o = model(xb, particle_mask=mb)
        cls = o["classification"] if isinstance(o, dict) else o
        out[i:i + batch] = torch.sigmoid(cls).squeeze(-1).cpu().numpy()
    return out


def reco_block(X, M):
    """Return per-event reco (hh_m, lead_m, sub_m); NaN where <4 jets."""
    r = reconstruct_dihiggs_from_constituents(X, masks=M, top_k=4, jet_R=0.4,
                                              min_jet_pt=25.0)
    n = len(X)
    hh = np.full(n, np.nan, np.float32)
    lead = np.full(n, np.nan, np.float32)
    sub = np.full(n, np.nan, np.float32)
    idx = r["event_indices"]
    if len(idx):
        hh[idx] = r["hh_m"]
        lead[idx] = r["lead_m"]
        sub[idx] = r["sub_m"]
    return hh, lead, sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="hh-bbbb-obj-config.json")
    ap.add_argument("--event-config", default="config_event.json")
    ap.add_argument("--ckpt", default="best_event_model_61z973dk.pth")
    ap.add_argument("--num-constituents", type=int, default=128)
    ap.add_argument("--skip-trigger", action="store_true", default=True)
    ap.add_argument("--apply-trigger", dest="skip_trigger", action="store_false")
    ap.add_argument("--max-signal", type=int, default=None)
    ap.add_argument("--qcd-max-per-bin", type=int, default=40000)
    ap.add_argument("--out", default="data/event_level/nsbi_cache.npz")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cache] device={device}", flush=True)
    cfg = json.load(open(args.config))
    ecfg = json.load(open(args.event_config))
    model = load_model(args.ckpt, ecfg, device)

    puppi_coll = cfg["l1extpuppi"]["collection_name"]      # L1ExtPuppi
    jet_coll = cfg["l1ext"]["collection_name"]             # L1puppiExtJetSC4
    jet_tag = cfg["l1ext"]["tagger_name"]                  # btagScore
    btag_wp = cfg["l1ext"]["b_tag_cut"]

    # ---------------- signal ----------------
    # Use the SAME glob-string pattern + tree + entry_stop for both the constituent
    # loader and the gen load so uproot expands files identically -> rows aligned.
    print("\n[cache] === signal HH->4b ===", flush=True)
    pattern = local_glob(cfg["file_pattern"])
    print(f"[cache] signal glob: {pattern}", flush=True)
    sig_puppi, sig_jets, n_sig = load_event_level_data(
        file_pattern=pattern, tree_name=cfg["tree_name"],
        puppi_collection=puppi_coll, jet_collection=jet_coll,
        jet_tagger_field=jet_tag, max_events=args.max_signal)
    gen = uproot.concatenate(f"{pattern}:{cfg['tree_name']}", GEN_BRANCHES,
                             library="ak", entry_stop=args.max_signal)
    assert len(gen) == n_sig, f"gen/puppi misalign {len(gen)} vs {n_sig}"
    gen_mhh, gen_cos = gen_higgs_kinematics(
        gen["GenPart_pt"], gen["GenPart_eta"], gen["GenPart_phi"],
        gen["GenPart_mass"], gen["GenPart_pdgId"], gen["GenPart_statusFlags"])

    sig_mask = (np.ones(n_sig, bool) if args.skip_trigger else
                passes_trigger_emulation(sig_jets, btag_wp=btag_wp, btag_field=jet_tag))
    print(f"[cache] signal kept {sig_mask.sum()}/{n_sig}", flush=True)
    sig_puppi = sig_puppi[sig_mask]
    gen_mhh, gen_cos = gen_mhh[sig_mask], gen_cos[sig_mask]

    Xs, Ms = extract_features_block(sig_puppi, args.num_constituents)
    sig_score = score_block(model, Xs, Ms, device)
    sig_hh, sig_lead, sig_sub = reco_block(Xs, Ms)

    # ---------------- QCD ----------------
    print("\n[cache] === QCD ===", flush=True)
    q_score, q_hh, q_lead, q_sub, q_sigma = [], [], [], [], []
    for name, b in cfg["QCD_background"].items():
        qp, qj, nq = load_event_level_data(
            file_pattern=local_glob(b["file_pattern"]),
            tree_name=b["tree_name"], puppi_collection=puppi_coll,
            jet_collection=jet_coll, jet_tagger_field=jet_tag,
            max_events=args.qcd_max_per_bin)
        if nq == 0:
            print(f"[cache]  {name}: 0 events, skip", flush=True)
            continue
        qmask = (np.ones(nq, bool) if args.skip_trigger else
                 passes_trigger_emulation(qj, btag_wp=btag_wp, btag_field=jet_tag))
        qp = qp[qmask]
        print(f"[cache]  {name}: kept {qmask.sum()}/{nq}", flush=True)
        Xq, Mq = extract_features_block(qp, args.num_constituents)
        q_score.append(score_block(model, Xq, Mq, device))
        hh, lead, sub = reco_block(Xq, Mq)
        q_hh.append(hh); q_lead.append(lead); q_sub.append(sub)
        q_sigma.append(np.full(len(hh), float(b["weight"]), np.float32))

    out = dict(
        sig_score=sig_score, sig_reco_mhh=sig_hh, sig_lead_m=sig_lead,
        sig_sub_m=sig_sub, sig_gen_mhh=gen_mhh.astype(np.float32),
        sig_cos_star=gen_cos.astype(np.float32),
        qcd_score=np.concatenate(q_score), qcd_reco_mhh=np.concatenate(q_hh),
        qcd_lead_m=np.concatenate(q_lead), qcd_sub_m=np.concatenate(q_sub),
        qcd_sigma=np.concatenate(q_sigma),
        meta=json.dumps(dict(puppi_coll=puppi_coll, skip_trigger=args.skip_trigger,
                             n_sig=int(sig_mask.sum()), ckpt=args.ckpt)),
    )
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"\n[cache] saved {args.out}", flush=True)
    print(f"[cache] signal: {len(sig_score)} ({np.isfinite(sig_hh).sum()} with reco); "
          f"QCD: {len(out['qcd_score'])} ({np.isfinite(out['qcd_reco_mhh']).sum()} reco)",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
