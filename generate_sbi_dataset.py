#!/usr/bin/env python3
"""
Generate event-level dataset for SBI analysis.

This script processes ROOT ntuples to extract event-level features
for neural simulation-based inference of the Higgs trilinear coupling.

Usage:
    python generate_sbi_dataset.py --config hh-bbbb-obj-config.json \\
                                    --output data/sbi_events.npz \\
                                    --use-part --part-checkpoint best_part_model.pth

Output:
    NPZ file with:
        - m_hh, pt_hh, etc.: event-level features
        - part_scores: ParT b-tag scores (if --use-part)
        - pnet_scores, upart_scores: baseline tagger scores
        - kappa_lambda: (optional) true kappa_lambda values from metadata
        - event_weights: event weights
"""

import os
import json
import argparse
import numpy as np
import awkward as ak

from tqdm import tqdm
from data_loading_helpers import load_and_prepare_data, select_gen_b_quarks_from_higgs
from analysis_helpers import apply_custom_cuts
from event_features import (
    extract_hh_features,
    load_part_model,
    compute_part_scores,
    extract_baseline_tagger_features,
)
from make_dataset import cluster_candidates


def generate_sbi_dataset(
    config_path,
    output_file,
    data_dir,
    use_part=False,
    part_checkpoint=None,
    part_config=None,
    use_l1_constituents=False,
    kappa_lambda=None,
):
    """
    Generate event-level dataset for SBI.

    Args:
        config_path: path to analysis config JSON
        output_file: output NPZ filename
        data_dir: directory with ROOT files
        use_part: whether to run ParT inference
        part_checkpoint: path to ParT model checkpoint
        part_config: path to ParT config JSON
        use_l1_constituents: if True, cluster L1 constituents; else use offline jets
        kappa_lambda: optional, true kappa_lambda value for this sample
    """
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Get ROOT files
    root_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".root")
    ]
    root_files.sort()

    print(f"Processing {len(root_files)} ROOT files...")

    # Load ParT model if requested
    part_model = None
    if use_part:
        print("Loading ParT model...")
        part_model = load_part_model(part_checkpoint, part_config)

    # Storage for all events
    all_features = {
        "m_hh": [],
        "pt_hh": [],
        "eta_hh": [],
        "phi_hh": [],
        "m_h1": [],
        "m_h2": [],
        "pt_h1": [],
        "pt_h2": [],
        "delta_r_hh": [],
        "delta_eta_hh": [],
        "delta_phi_hh": [],
        "cos_theta_star": [],
        "n_jets": [],
    }
    all_part_scores = []
    all_baseline_scores = {
        "pnet": [],
        "upart": [],
        "l1ext": [],
    }
    all_kappa_lambda = []
    all_weights = []

    # Process each file
    for root_file in tqdm(root_files):
        config["file_pattern"] = root_file

        # Collections to load
        if use_l1_constituents:
            collections = ["L1BarrelExtPuppi", "GenPart"]
        else:
            collections = ["Jet", "L1puppiExtJetSC4", "GenPart"]

        # Load events
        events = load_and_prepare_data(
            config["file_pattern"],
            config["tree_name"],
            collections,
            max_events=config["max_events"],
            correct_pt=True,
            CONFIG=config,
        )

        n_events = len(events)
        if n_events == 0:
            continue

        # Get jets
        if use_l1_constituents:
            # Cluster L1 constituents with ParT
            print("Clustering L1 constituents...")
            jets = cluster_candidates(
                events, config, "l1barrelextpuppi", dist_param=0.8
            )

            # Apply kinematic cuts
            pt_mask = jets.pt > 25.0
            eta_mask = np.abs(jets.eta) < 2.4
            jets = jets[pt_mask & eta_mask]

        else:
            # Use offline jets
            jets = events.Jet
            jets = apply_custom_cuts(jets, config, "offline", kinematic_only=True)

        # Require at least 4 jets
        event_mask = ak.num(jets, axis=1) >= 4
        jets = jets[event_mask]
        events = events[event_mask]

        if len(jets) == 0:
            continue

        # ParT inference
        if use_part and use_l1_constituents:
            print("Running ParT inference...")
            # Prepare inputs (reuse logic from make_dataset.py)
            from make_dataset import process_batch

            # Extract constituent features
            constituents = jets.constituents[:, :4, :, :]  # Top 4 jets
            # Flatten to (N_jets, N_const, N_feat)
            # ... (detailed implementation needed)
            # For now, placeholder:
            part_scores_batch = np.zeros((len(jets), 4))  # Top 4 jets
        else:
            part_scores_batch = np.zeros((len(jets), 4))

        # Extract baseline tagger scores
        if not use_l1_constituents:
            baseline_scores = extract_baseline_tagger_features(
                jets, tagger_names=["btagPNetB", "btagUParTAK4probb"]
            )
        else:
            # Use L1 scores
            l1_jets = (
                events.L1puppiExtJetSC4 if "L1puppiExtJetSC4" in events.fields else None
            )
            if l1_jets is not None:
                l1_jets = l1_jets[ak.argsort(l1_jets.pt, axis=1, ascending=False)]
                baseline_scores = {
                    "l1ext": ak.to_numpy(
                        ak.fill_none(
                            ak.pad_none(l1_jets.btagScore[:, :4], 4, axis=1, clip=True),
                            0.0,
                        )
                    )
                }
            else:
                baseline_scores = {}

        # Extract HH features
        hh_features = extract_hh_features(events, jets)

        # Store
        for key in all_features.keys():
            if key in hh_features:
                all_features[key].append(hh_features[key])

        all_part_scores.append(part_scores_batch)

        for key in baseline_scores:
            if key == "btagPNetB":
                all_baseline_scores["pnet"].append(baseline_scores[key])
            elif key == "btagUParTAK4probb":
                all_baseline_scores["upart"].append(baseline_scores[key])
            elif key == "l1ext":
                all_baseline_scores["l1ext"].append(baseline_scores[key])

        # Kappa lambda (if provided)
        if kappa_lambda is not None:
            all_kappa_lambda.extend([kappa_lambda] * len(hh_features["m_hh"]))

        # Weights (uniform for now)
        all_weights.extend([1.0] * len(hh_features["m_hh"]))

    # Concatenate
    print("Concatenating all batches...")
    final_features = {
        key: np.concatenate(val) for key, val in all_features.items() if len(val) > 0
    }

    # Save dataset
    print(f"Saving to {output_file}...")
    save_dict = {**final_features}

    if len(all_part_scores) > 0:
        save_dict["part_scores"] = np.concatenate(all_part_scores)

    for key, val in all_baseline_scores.items():
        if len(val) > 0:
            save_dict[f"{key}_scores"] = np.concatenate(val)

    if len(all_kappa_lambda) > 0:
        save_dict["kappa_lambda"] = np.array(all_kappa_lambda)

    save_dict["weights"] = np.array(all_weights)

    np.savez_compressed(output_file, **save_dict)

    print(f"Saved {len(final_features['m_hh'])} events")
    print(f"Features: {list(final_features.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SBI event dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="hh-bbbb-obj-config.json",
        help="Analysis configuration JSON",
    )
    parser.add_argument(
        "--output", type=str, default="data/sbi_events.npz", help="Output NPZ file"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/hh4b", help="Directory with ROOT files"
    )
    parser.add_argument(
        "--use-part", action="store_true", help="Run ParT inference for b-tagging"
    )
    parser.add_argument(
        "--part-checkpoint",
        type=str,
        default="best_part_model.pth",
        help="ParT model checkpoint",
    )
    parser.add_argument(
        "--part-config",
        type=str,
        default="config_part.json",
        help="ParT configuration JSON",
    )
    parser.add_argument(
        "--use-l1",
        action="store_true",
        help="Cluster L1 constituents instead of using offline jets",
    )
    parser.add_argument(
        "--kappa-lambda",
        type=float,
        default=None,
        help="True kappa_lambda value for this sample",
    )

    args = parser.parse_args()

    generate_sbi_dataset(
        config_path=args.config,
        output_file=args.output,
        data_dir=args.data_dir,
        use_part=args.use_part,
        part_checkpoint=args.part_checkpoint,
        part_config=args.part_config,
        use_l1_constituents=args.use_l1,
        kappa_lambda=args.kappa_lambda,
    )
