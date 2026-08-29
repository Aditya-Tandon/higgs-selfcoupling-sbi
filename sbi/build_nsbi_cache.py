"""
Build a self-contained, row-aligned NSBI cache for iteration-2 κλ inference.

For every event it stores the *real* iteration-2 observables + everything the
extended likelihood needs, all aligned by construction (no cross-dataset joins):

  signal (HH->4b):  score, reco_mhh, reco_lead_m, reco_sub_m, gen_mhh, cos_star
  QCD:              score, reco_mhh, reco_lead_m, reco_sub_m, qcd_sigma (raw σ_bin)

Extended schema (2026-07-12 rebuild — unblocks Dirs 1/2/8 + the Dir-8 control):
  {sig,qcd}_jet_pt / _jet_eta / _jet_btag   leading TOP_K l1ext jets by pT,
                                            NaN-padded (N, k) float32
  {sig,qcd}_ht / _n_jets                    over ALL l1ext jets (= trigger HT)
  {sig,qcd}_ref_{pt,eta,btagPNetB,btagUParTAK4B,ht,n_jets}   offline Jet coll.
  {sig,qcd}_l1ng_{pt,eta,bTagScore,ht,n_jets}                L1puppiJetSC4NG
  (reference collections stored only where the branches exist in the files)

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
import time

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

_T0 = time.time()


def stamp(msg):
    print(f"[cache][t+{time.time() - _T0:7.1f}s] {msg}", flush=True)


# Per-event jet info stored in the cache (leading TOP_K jets by pT, NaN-padded).
# NOTE: btagScore uses -1.0 as an "untagged / no score" sentinel — kept raw.
# Primary collection (l1ext) feeds the (jet-pT, HT, n-jet, b-tag-WP) selection
# grid; reference collections feed the Dir-8 offline-tagger control study.
TOP_K = 10
REF_COLLECTIONS = {
    # short name -> (collection, [score branches])
    "ref":  ("Jet", ["btagPNetB", "btagUParTAK4B"]),      # offline taggers
    "l1ng": ("L1puppiJetSC4NG", ["bTagScore"]),           # L1 next-gen tagger
}


def topk_from_jets(jets, fields, k=TOP_K):
    """Per-event leading-k jets by pT for each field, NaN-padded (N, k) float32,
    plus exact all-jet HT and n_jets (computed before truncation)."""
    idx = ak.argsort(jets.pt, ascending=False)
    s = jets[idx]
    out = {
        "ht": ak.to_numpy(ak.sum(jets.pt, axis=1)).astype(np.float32),
        "n_jets": ak.to_numpy(ak.num(jets.pt, axis=1)).astype(np.int16),
    }
    for f in fields:
        padded = ak.pad_none(s[f], k, axis=1, clip=True)
        out[f] = ak.to_numpy(ak.fill_none(padded, np.nan)).astype(np.float32)
    return out


def load_ref_jet_blocks(pattern, tree_name, entry_stop, n_expected):
    """Load reference-tagger jet collections (offline PNet/UParT, L1NG) row-aligned
    with the main load (same glob + entry_stop trick as the gen load). Collections
    whose branches are absent from the files are skipped. Returns
    {short: {field: (N, k) array, 'ht': ..., 'n_jets': ...}}."""
    import os
    first = sorted(glob.glob(pattern))[0]
    available = set(uproot.open(first)[tree_name].keys())
    blocks = {}
    for short, (coll, scores) in REF_COLLECTIONS.items():
        branches = [f"{coll}_{f}" for f in ["pt", "eta"] + scores]
        missing = [b for b in branches if b not in available]
        if missing:
            print(f"[cache]  ref '{short}' ({coll}): missing {missing}, skipped",
                  flush=True)
            continue
        ev = uproot.concatenate(f"{pattern}:{tree_name}", branches, library="ak",
                                entry_stop=entry_stop)
        assert len(ev) == n_expected, \
            f"ref '{short}' misalign {len(ev)} vs {n_expected}"
        jets = ak.zip({f.replace(f"{coll}_", ""): ev[f] for f in branches})
        blocks[short] = topk_from_jets(jets, ["pt", "eta"] + scores)
        stamp(f"ref '{short}' ({coll}) loaded ({n_expected} events)")
    return blocks


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
    stamp("signal constituents+jets loaded")
    gen = uproot.concatenate(f"{pattern}:{cfg['tree_name']}", GEN_BRANCHES,
                             library="ak", entry_stop=args.max_signal)
    stamp("signal gen branches loaded")
    assert len(gen) == n_sig, f"gen/puppi misalign {len(gen)} vs {n_sig}"
    gen_mhh, gen_cos = gen_higgs_kinematics(
        gen["GenPart_pt"], gen["GenPart_eta"], gen["GenPart_phi"],
        gen["GenPart_mass"], gen["GenPart_pdgId"], gen["GenPart_statusFlags"])

    sig_mask = (np.ones(n_sig, bool) if args.skip_trigger else
                passes_trigger_emulation(sig_jets, btag_wp=btag_wp, btag_field=jet_tag))
    print(f"[cache] signal kept {sig_mask.sum()}/{n_sig}", flush=True)
    sig_puppi = sig_puppi[sig_mask]
    gen_mhh, gen_cos = gen_mhh[sig_mask], gen_cos[sig_mask]

    # per-jet info: primary l1ext collection (already in memory) + reference taggers
    sig_l1ext = {k: v[sig_mask] for k, v in
                 topk_from_jets(sig_jets, ["pt", "eta", jet_tag]).items()}
    sig_refs = {short: {k: v[sig_mask] for k, v in blk.items()}
                for short, blk in
                load_ref_jet_blocks(pattern, cfg["tree_name"], args.max_signal,
                                    n_sig).items()}

    Xs, Ms = extract_features_block(sig_puppi, args.num_constituents)
    stamp("signal features extracted")
    sig_score = score_block(model, Xs, Ms, device)
    stamp("signal scored")
    sig_hh, sig_lead, sig_sub = reco_block(Xs, Ms)
    stamp("signal reco done")

    # ---------------- QCD ----------------
    print("\n[cache] === QCD ===", flush=True)
    q_score, q_hh, q_lead, q_sub, q_sigma = [], [], [], [], []
    q_l1ext, q_refs, n_bins_done = [], [], 0
    for name, b in cfg["QCD_background"].items():
        qpat = local_glob(b["file_pattern"])
        qp, qj, nq = load_event_level_data(
            file_pattern=qpat,
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
        stamp(f"{name}: constituents+jets loaded")
        q_l1ext.append({k: v[qmask] for k, v in
                        topk_from_jets(qj, ["pt", "eta", jet_tag]).items()})
        q_refs.append({short: {k: v[qmask] for k, v in blk.items()}
                       for short, blk in
                       load_ref_jet_blocks(qpat, b["tree_name"],
                                           args.qcd_max_per_bin, nq).items()})
        n_bins_done += 1
        Xq, Mq = extract_features_block(qp, args.num_constituents)
        q_score.append(score_block(model, Xq, Mq, device))
        stamp(f"{name}: scored")
        hh, lead, sub = reco_block(Xq, Mq)
        stamp(f"{name}: reco done")
        q_hh.append(hh); q_lead.append(lead); q_sub.append(sub)
        q_sigma.append(np.full(len(hh), float(b["weight"]), np.float32))

    # concatenate per-bin jet blocks; keep a ref collection only if every bin has it
    qcd_l1ext = {k: np.concatenate([d[k] for d in q_l1ext]) for k in q_l1ext[0]}
    qcd_refs = {}
    for short in REF_COLLECTIONS:
        if all(short in d for d in q_refs):
            qcd_refs[short] = {k: np.concatenate([d[short][k] for d in q_refs])
                               for k in q_refs[0][short]}
        elif any(short in d for d in q_refs):
            print(f"[cache] WARNING: ref '{short}' present in only some QCD bins "
                  f"-> dropped from QCD side", flush=True)

    out = dict(
        sig_score=sig_score, sig_reco_mhh=sig_hh, sig_lead_m=sig_lead,
        sig_sub_m=sig_sub, sig_gen_mhh=gen_mhh.astype(np.float32),
        sig_cos_star=gen_cos.astype(np.float32),
        qcd_score=np.concatenate(q_score), qcd_reco_mhh=np.concatenate(q_hh),
        qcd_lead_m=np.concatenate(q_lead), qcd_sub_m=np.concatenate(q_sub),
        qcd_sigma=np.concatenate(q_sigma),
        # primary (l1ext) per-jet info: leading TOP_K by pT, NaN-padded (N, k);
        # ht/n_jets computed over ALL jets (matches trigger emulation)
        sig_jet_pt=sig_l1ext["pt"], sig_jet_eta=sig_l1ext["eta"],
        sig_jet_btag=sig_l1ext[jet_tag],
        sig_ht=sig_l1ext["ht"], sig_n_jets=sig_l1ext["n_jets"],
        qcd_jet_pt=qcd_l1ext["pt"], qcd_jet_eta=qcd_l1ext["eta"],
        qcd_jet_btag=qcd_l1ext[jet_tag],
        qcd_ht=qcd_l1ext["ht"], qcd_n_jets=qcd_l1ext["n_jets"],
        meta=json.dumps(dict(puppi_coll=puppi_coll, skip_trigger=args.skip_trigger,
                             n_sig=int(sig_mask.sum()), ckpt=args.ckpt,
                             top_k=TOP_K, jet_coll=jet_coll, jet_tag=jet_tag,
                             sig_refs=sorted(sig_refs), qcd_refs=sorted(qcd_refs))),
    )
    # reference-tagger collections (Dir-8 offline-tagger control):
    # {sig,qcd}_{ref,l1ng}_{pt,eta,<score>,ht,n_jets}
    for prefix, blocks in (("sig", sig_refs), ("qcd", qcd_refs)):
        for short, blk in blocks.items():
            for field, arr in blk.items():
                out[f"{prefix}_{short}_{field}"] = arr
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"\n[cache] saved {args.out}", flush=True)
    print(f"[cache] signal: {len(sig_score)} ({np.isfinite(sig_hh).sum()} with reco); "
          f"QCD: {len(out['qcd_score'])} ({np.isfinite(out['qcd_reco_mhh']).sum()} reco)",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
