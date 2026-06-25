#!/usr/bin/env python
"""
multiplicity_stats.py
=====================
Compute jet and particle-candidate multiplicities for signal (HH→bbbb)
and QCD samples, using L1 PF and PUPPI collections.

Reports
-------
1. Mean / median / std of candidate (particle) multiplicity per event
   — for signal and each QCD pT bin, separately for PF and PUPPI.
2. Mean / median / std of AK4 jet multiplicity per event after clustering
   — for both collections, signal and QCD.
3. Same jet multiplicity restricted to gen-b-matched jets (cross-matching),
   to show how many of the clustered jets survive matching.

Uses:
    data_pipeline.root_loading.load_and_prepare_data
    data_pipeline.root_loading.select_gen_b_quarks_from_higgs
    data_pipeline.root_loading.select_gen_b_quarks_by_status
    data_pipeline.make_particle_dataset.cluster_candidates
    evaluation.jet_matching.get_purity_mask_cross_matched
"""

import json
import sys
import os
import gc
import numpy as np
import awkward as ak

# ── project imports ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.root_loading import (
    load_and_prepare_data,
    select_gen_b_quarks_from_higgs,
    select_gen_b_quarks_by_status,
    apply_custom_cuts,
)
from data_pipeline.make_particle_dataset import cluster_candidates
from evaluation.jet_matching import get_purity_mask_cross_matched

# ── config ───────────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "hh-bbbb-obj-config.json")
with open(CONFIG_PATH) as f:
    config = json.load(f)

# Collections to compare
COLLECTIONS = {
    "PF":    "l1extpf",
    "PUPPI": "l1extpuppi",
}

# How many signal / QCD events to scan (None = all available)
MAX_SIGNAL_EVENTS = None
MAX_QCD_EVENTS_PER_BIN = None   # per pT bin


# ── helpers ──────────────────────────────────────────────────
def describe(arr, label=""):
    """Print mean / median / std / min / max for a 1-D array."""
    if len(arr) == 0:
        print(f"  {label:.<40s} NO DATA")
        return
    print(
        f"  {label:.<40s} "
        f"mean={np.mean(arr):8.2f}  "
        f"median={np.median(arr):6.1f}  "
        f"std={np.std(arr):7.2f}  "
        f"min={np.min(arr):4.0f}  "
        f"max={np.max(arr):5.0f}  "
        f"N_events={len(arr)}"
    )


def load_events(file_pattern, tree_name, collections_to_load, max_events=None,
                entry_start=None, cfg=None):
    """Thin wrapper around load_and_prepare_data."""
    return load_and_prepare_data(
        file_pattern,
        tree_name,
        collections_to_load,
        max_events=max_events,
        correct_pt=False,          # no pT correction needed for counting
        CONFIG=cfg or config,
        filter_branches=True,
        entry_start=entry_start,
    )


# ── per-sample analysis ─────────────────────────────────────
def analyse_sample(events, cfg, collection_key, gen_b_quarks=None, label=""):
    """
    For one sample (signal or a QCD bin) and one collection:
      • candidate multiplicity per event
      • jet multiplicity per event after AK4 clustering
      • matched-jet multiplicity per event (if gen_b_quarks provided)

    Returns dict of 1-D numpy arrays.
    """
    coll_name = cfg[collection_key]["collection_name"]

    # ---- candidate multiplicity (raw particles before clustering) ----
    candidates = events[coll_name]
    # apply the same kinematic cuts that cluster_candidates uses internally
    candidates_cut = apply_custom_cuts(candidates, cfg, collection_key,
                                       kinematic_only=True, return_jets=True)
    n_cands_per_event = ak.to_numpy(ak.num(candidates_cut, axis=1))

    # ---- cluster ----
    jets = cluster_candidates(events, cfg, collection_key, dist_param=0.4)
    n_jets_per_event = ak.to_numpy(ak.num(jets, axis=1))

    # ---- constituent multiplicity per jet ----
    n_const_per_jet = ak.to_numpy(
        ak.flatten(ak.num(jets.constituents, axis=2), axis=None)
    )

    # ---- cross-matched jets (only if gen truth available) ----
    n_matched_per_event = None
    if gen_b_quarks is not None:
        match_mask, _ = get_purity_mask_cross_matched(
            gen_b_quarks, jets, CONFIG=cfg, return_gen_idx=True
        )
        n_matched_per_event = ak.to_numpy(ak.sum(match_mask, axis=1))

    del jets, candidates, candidates_cut
    gc.collect()

    return {
        "n_cands":   n_cands_per_event,
        "n_jets":    n_jets_per_event,
        "n_const":   n_const_per_jet,
        "n_matched": n_matched_per_event,
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    sep = "=" * 90

    # All collection names we need to load from ROOT
    all_coll_names = [config[k]["collection_name"] for k in COLLECTIONS.values()]
    # Also need GenPart for matching
    all_coll_names.append(config["gen"]["collection_name"])
    all_coll_names = list(set(all_coll_names))

    # ─────────────────────────────────────────────────────────
    # 1.  SIGNAL (HH → bbbb)
    # ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  SIGNAL  (HH → bbbb)")
    print(sep)

    sig_events = load_events(
        config["file_pattern"],
        config["tree_name"],
        all_coll_names,
        max_events=MAX_SIGNAL_EVENTS,
        cfg=config,
    )
    n_sig = len(sig_events)
    print(f"  Loaded {n_sig} signal events\n")

    gen_b_sig = select_gen_b_quarks_from_higgs(sig_events)

    for coll_label, coll_key in COLLECTIONS.items():
        print(f"  --- {coll_label} ({config[coll_key]['collection_name']}) ---")
        res = analyse_sample(sig_events, config, coll_key,
                             gen_b_quarks=gen_b_sig, label=f"signal-{coll_label}")

        describe(res["n_cands"],  f"Candidates/event ({coll_label})")
        describe(res["n_jets"],   f"AK4 jets/event ({coll_label})")
        describe(res["n_const"],  f"Constituents/jet ({coll_label})")
        if res["n_matched"] is not None:
            describe(res["n_matched"], f"Gen-b matched jets/event ({coll_label})")
        print()

    del sig_events, gen_b_sig
    gc.collect()

    # ─────────────────────────────────────────────────────────
    # 2.  QCD BACKGROUND  (per pT bin)
    # ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  QCD BACKGROUND")
    print(sep)

    qcd_config = config["QCD_background"]

    # Accumulators across all QCD bins (for a grand summary)
    qcd_accum = {k: {m: [] for m in ["n_cands", "n_jets", "n_const", "n_matched"]}
                 for k in COLLECTIONS}

    for bin_name, bin_cfg in qcd_config.items():
        print(f"\n  ── {bin_name}  (σ = {bin_cfg['weight']:.3e} pb) ──")

        qcd_cfg = dict(config)
        qcd_cfg["file_pattern"] = bin_cfg["file_pattern"]
        qcd_cfg["tree_name"] = bin_cfg["tree_name"]

        max_ev = bin_cfg.get("max_events") or MAX_QCD_EVENTS_PER_BIN
        try:
            qcd_events = load_events(
                bin_cfg["file_pattern"],
                bin_cfg["tree_name"],
                all_coll_names,
                max_events=max_ev,
                cfg=qcd_cfg,
            )
        except Exception as e:
            print(f"    ⚠ Could not load: {e}")
            continue

        n_qcd = len(qcd_events)
        if n_qcd == 0:
            print("    (empty)")
            continue
        print(f"    Loaded {n_qcd} events")

        # QCD gen-b quarks (from gluon splitting etc.) for cross-matching
        gen_b_qcd = select_gen_b_quarks_by_status(qcd_events, config=qcd_cfg)

        for coll_label, coll_key in COLLECTIONS.items():
            res = analyse_sample(qcd_events, qcd_cfg, coll_key,
                                 gen_b_quarks=gen_b_qcd,
                                 label=f"{bin_name}-{coll_label}")

            describe(res["n_cands"], f"  Candidates/event ({coll_label})")
            describe(res["n_jets"],  f"  AK4 jets/event ({coll_label})")
            describe(res["n_const"], f"  Constituents/jet ({coll_label})")
            if res["n_matched"] is not None:
                describe(res["n_matched"], f"  Gen-b matched jets/event ({coll_label})")

            for m in ["n_cands", "n_jets", "n_const", "n_matched"]:
                if res[m] is not None:
                    qcd_accum[coll_label][m].append(res[m])

        del qcd_events, gen_b_qcd
        gc.collect()

    # ─────────────────────────────────────────────────────────
    # 3.  QCD GRAND SUMMARY (all bins concatenated)
    # ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  QCD — ALL BINS COMBINED")
    print(sep)
    for coll_label in COLLECTIONS:
        print(f"\n  --- {coll_label} ---")
        for m in ["n_cands", "n_jets", "n_const", "n_matched"]:
            arrs = qcd_accum[coll_label][m]
            if arrs:
                combined = np.concatenate(arrs)
                nice = {"n_cands": "Candidates/event",
                        "n_jets": "AK4 jets/event",
                        "n_const": "Constituents/jet",
                        "n_matched": "Gen-b matched jets/event"}[m]
                describe(combined, f"{nice} ({coll_label})")

    print(f"\n{sep}")
    print("  Done.")
    print(sep)


if __name__ == "__main__":
    main()
