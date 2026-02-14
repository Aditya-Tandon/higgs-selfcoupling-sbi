#!/usr/bin/env python3
"""
Compare m_HH resolution across different tagger configurations.

Loads offline (PNet), L1NG, L1ext, and ParT jet collections,
reconstructs di-Higgs via D_HH pairing for each, and produces
comparison plots.

For ParT: clusters L1BarrelExtPuppi constituents into jets, runs
ParT inference to get per-jet b-tag scores, selects top 4 b-tagged
jets, then pairs via D_HH.

Run with:
    conda run -n hep-root-ml python run_tagger_comparison.py
    conda run -n hep-root-ml python run_tagger_comparison.py --skip-part
"""

import os
import json
import argparse
import numpy as np
import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_loading_helpers import load_and_prepare_data, apply_custom_cuts
from event_features import pair_jets_to_higgs, extract_hh_features

sns.set_style("whitegrid")
PLOT_DIR = "Updates/sbi-test/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# ---- helpers ----------------------------------------------------------------


def load_jets_for_collection(config, key, events):
    """Return kinematically-cut jets for a given config key."""
    coll = config[key]["collection_name"]
    if coll not in events.fields:
        print(f"  [SKIP] Collection '{coll}' not found in data")
        return None

    jets = events[coll]
    jets = apply_custom_cuts(jets, config, key, kinematic_only=True)
    return jets


def features_for_jets(events, jets, label):
    """Pair jets, extract features, return dict with metadata."""
    mask = ak.num(jets, axis=1) >= 4
    sel_jets = jets[mask]
    sel_events = events[mask]
    n_pass = int(ak.sum(mask))
    print(f"  [{label}] Events with >= 4 jets: {n_pass}")
    if n_pass == 0:
        return None
    feats = extract_hh_features(sel_events, sel_jets)
    feats["_label"] = label
    feats["_n_pass"] = n_pass
    return feats


def run_part_inference(
    events,
    config,
    part_checkpoint,
    part_config_path,
    n_constituents=16,
    cluster_dist_param=0.4,
):
    """
    Run ParT inference on L1BarrelExtPuppi constituents.

    Pipeline:
      1. Cluster L1BarrelExtPuppi into jets using anti-kt
      2. Prepare constituent features (17 dims per particle)
      3. Run ParT forward pass to get per-jet b-tag scores
      4. Sort jets by ParT score (descending, best b-jets first)
      5. Select top 4 b-tagged jets per event
      6. Return jets with vector field for D_HH pairing

    Returns:
        jets_top4: awkward array of top 4 b-tagged jets per event
        n_events_with_4: number of events with at least 4 jets
    """
    import torch
    from make_dataset import cluster_candidates
    from data_loading_helpers import one_hot_encode_l1_puppi
    from event_features import load_part_model

    print("  [ParT] Loading model...")
    part_model = load_part_model(part_checkpoint, part_config_path)

    # Use MPS if available, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    part_model = part_model.to(device)
    print(f"  [ParT] Using device: {device}")

    # 1. Cluster L1 constituents into jets
    print("  [ParT] Clustering L1BarrelExtPuppi constituents (anti-kt R=0.4)...")
    clustered_jets = cluster_candidates(
        events,
        config,
        "l1barrelextpuppi",
        dist_param=cluster_dist_param,
    )
    n_jets_total = int(ak.sum(ak.num(clustered_jets, axis=1)))
    print(f"  [ParT] Clustered {n_jets_total} jets")

    # Sort jets by pT
    sorted_indices = ak.argsort(clustered_jets.pt, axis=1, ascending=False)
    clustered_jets = clustered_jets[sorted_indices]

    # 2. Prepare constituent features
    print("  [ParT] Preparing constituent features...")
    matched_cands = clustered_jets.constituents

    # Sort constituents by pT
    const_pt_sort = ak.argsort(matched_cands.pt, axis=2, ascending=False)
    matched_cands = matched_cands[const_pt_sort]

    j_pt = clustered_jets.pt[:, :, None]
    j_eta = clustered_jets.eta[:, :, None]
    j_phi = clustered_jets.phi[:, :, None]

    m_pt = matched_cands.vector.pt
    m_eta = matched_cands.vector.eta
    m_phi = matched_cands.vector.phi
    m_mass = matched_cands.vector.mass
    m_dxy = matched_cands.dxy
    m_z0 = matched_cands.z0
    m_charge = matched_cands.charge

    log_pt_rel = np.log(np.maximum(m_pt, 1e-3) / np.maximum(j_pt, 1e-3))
    deta = m_eta - j_eta
    dphi = m_phi - j_phi
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    m_w = matched_cands.puppiWeight
    log_dr = np.log(np.maximum(np.sqrt(deta**2 + dphi**2), 1e-3))
    m_id = matched_cands.id

    def pad_and_fill(arr, target=n_constituents):
        return ak.fill_none(ak.pad_none(arr, target, axis=2, clip=True), 0.0)

    feature_list = [
        pad_and_fill(m_mass),
        pad_and_fill(m_pt),
        pad_and_fill(m_eta),
        pad_and_fill(m_phi),
        pad_and_fill(m_dxy),
        pad_and_fill(m_z0),
        pad_and_fill(m_charge),
        pad_and_fill(log_pt_rel),
        pad_and_fill(deta),
        pad_and_fill(dphi),
        pad_and_fill(m_w),
        pad_and_fill(log_dr),
        pad_and_fill(m_id),
    ]

    n_jets_per_event = ak.num(clustered_jets, axis=1)

    x_ini = np.stack(
        [ak.to_numpy(ak.flatten(f, axis=1)) for f in feature_list],
        axis=-1,
    )
    flat_ids = x_ini[..., -1]
    one_hot_ids = one_hot_encode_l1_puppi(flat_ids, n_classes=5)
    X = np.concatenate([x_ini[..., :-1], one_hot_ids], axis=-1)

    # Particle mask
    n_actual = ak.num(matched_cands, axis=2)
    n_actual_flat = ak.to_numpy(ak.flatten(n_actual, axis=1))
    particle_mask = np.zeros((X.shape[0], n_constituents), dtype=bool)
    for i in range(X.shape[0]):
        n_real = min(n_actual_flat[i], n_constituents)
        particle_mask[i, :n_real] = True

    # 3. Run ParT inference
    print(f"  [ParT] Running inference on {len(X)} jets...")
    part_model.eval()
    batch_size = 512
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = torch.tensor(
                X[i : i + batch_size],
                dtype=torch.float32,
                device=device,
            )
            batch_mask = torch.tensor(
                particle_mask[i : i + batch_size],
                dtype=torch.bool,
                device=device,
            )
            logits = part_model(batch_x, batch_mask)
            scores = torch.sigmoid(logits).cpu().numpy().flatten()
            all_scores.append(scores)

    flat_scores = np.concatenate(all_scores)
    print(
        f"  [ParT] Scores: mean={flat_scores.mean():.4f}, "
        f"std={flat_scores.std():.4f}"
    )

    # 4. Unflatten scores back to (events, jets)
    scores_ak = ak.unflatten(flat_scores, n_jets_per_event)

    # 5. Sort jets by ParT b-tag score (descending)
    score_sort = ak.argsort(scores_ak, axis=1, ascending=False)
    sorted_jets = clustered_jets[score_sort]

    # 6. Require >= 4 jets, take top 4
    has_4 = ak.num(sorted_jets, axis=1) >= 4
    n_pass = int(ak.sum(has_4))
    print(f"  [ParT] Events with >= 4 jets: {n_pass}")

    return sorted_jets, has_4


# ---- plotting ---------------------------------------------------------------


def plot_mhh_comparison(results, tag="all"):
    """Compare m_HH distributions across taggers."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colours = plt.cm.Set2.colors

    # ---- panel 0: overlaid m_HH distributions ----
    ax = axes[0]
    bins = np.linspace(200, 1200, 60)
    for i, r in enumerate(results):
        m = r["m_hh"]
        in_range = (m >= 200) & (m <= 1200)
        ax.hist(
            m[in_range],
            bins=bins,
            alpha=0.55,
            label=r["_label"],
            edgecolor="black",
            linewidth=0.5,
            color=colours[i % len(colours)],
        )
    ax.set_xlabel("$m_{HH}$ [GeV]")
    ax.set_ylabel("Events")
    ax.set_title("Di-Higgs Mass Distributions")
    ax.legend()

    # ---- panel 1: normalised comparison ----
    ax = axes[1]
    for i, r in enumerate(results):
        m = r["m_hh"]
        in_range = (m >= 200) & (m <= 1200)
        ax.hist(
            m[in_range],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=r["_label"],
            color=colours[i % len(colours)],
        )
    ax.set_xlabel("$m_{HH}$ [GeV]")
    ax.set_ylabel("Density")
    ax.set_title("Normalised $m_{HH}$")
    ax.legend()

    # ---- panel 2: resolution bar chart ----
    ax = axes[2]
    labels, resolutions = [], []
    for r in results:
        m = r["m_hh"]
        in_range = (m >= 200) & (m <= 1200)
        mu = np.mean(m[in_range])
        sigma = np.std(m[in_range])
        labels.append(r["_label"])
        resolutions.append(sigma / mu)

    x_pos = np.arange(len(labels))
    bars = ax.bar(
        x_pos,
        resolutions,
        color=[colours[i % len(colours)] for i in range(len(labels))],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Resolution ($\\sigma / \\mu$)")
    ax.set_title("$m_{HH}$ Resolution Comparison")
    for bar, val in zip(bars, resolutions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"mhh_tagger_comparison_{tag}.png")
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()


def plot_higgs_masses(results, tag="all"):
    """Compare individual Higgs masses across taggers."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    colours = plt.cm.Set2.colors

    for i, (r, ax) in enumerate(zip(results, axes)):
        ax.scatter(
            r["m_h1"], r["m_h2"], alpha=0.15, s=5, color=colours[i % len(colours)]
        )
        ax.axhline(125, color="red", linestyle="--", alpha=0.6)
        ax.axvline(125, color="red", linestyle="--", alpha=0.6)
        ax.set_xlabel("$m_{H_1}$ [GeV]")
        ax.set_ylabel("$m_{H_2}$ [GeV]")
        ax.set_title(r["_label"])
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"higgs_masses_comparison_{tag}.png")
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()


def plot_efficiency_table(results, n_total):
    """Print table of event-selection efficiencies."""
    print("\n" + "=" * 70)
    print(
        f"{'Tagger':<25s} {'Events (>=4j)':>12s} {'Efficiency':>12s} "
        f"{'m_HH mean':>10s} {'m_HH std':>10s} {'Resol.':>8s}"
    )
    print("-" * 70)
    for r in results:
        m = r["m_hh"]
        in_range = (m >= 200) & (m <= 1200)
        mu = np.mean(m[in_range])
        sigma = np.std(m[in_range])
        eff = r["_n_pass"] / n_total
        print(
            f"{r['_label']:<25s} {r['_n_pass']:>12d} {eff:>12.4f} "
            f"{mu:>10.1f} {sigma:>10.1f} {sigma/mu:>8.4f}"
        )
    print("=" * 70)


# ---- main -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Tagger comparison")
    parser.add_argument(
        "--skip-part",
        action="store_true",
        help="Skip ParT inference (use existing taggers only)",
    )
    parser.add_argument(
        "--part-checkpoint",
        type=str,
        default="final_model_78qruhsg.pth",
        help="ParT model checkpoint path",
    )
    parser.add_argument(
        "--part-config",
        type=str,
        default="config_part.json",
        help="ParT config JSON path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Tagger Comparison: m_HH Resolution Study")
    print("=" * 60)

    with open("hh-bbbb-obj-config.json") as f:
        config = json.load(f)

    # Collections to compare (conventional taggers)
    keys_to_compare = [
        ("offline", "Offline (PNet)"),
        ("l1ng", "L1 NG"),
        ("l1ext", "L1 Ext"),
    ]

    # Determine which collections we need
    collection_names = set()
    for key, _ in keys_to_compare:
        collection_names.add(config[key]["collection_name"])
    # Always need GenPart and L1BarrelExtPuppi (for ParT)
    collections = list(collection_names) + ["GenPart"]
    if not args.skip_part:
        collections.append("L1BarrelExtPuppi")

    print(f"\n1. Loading data (collections: {collections})...")
    events = load_and_prepare_data(
        config["file_pattern"],
        config["tree_name"],
        collections,
        max_events=None,
        correct_pt=True,
        CONFIG=config,
    )
    n_total = len(events)
    print(f"   Loaded {n_total} events")

    print("\n2. Extracting features per tagger...")
    results = []
    for key, label in keys_to_compare:
        print(f"\n  --- {label} ({key}) ---")
        jets = load_jets_for_collection(config, key, events)
        if jets is None:
            continue
        feats = features_for_jets(events, jets, label)
        if feats is not None:
            results.append(feats)

    # ParT tagger
    if not args.skip_part:
        print(f"\n  --- ParT (L1 constituents) ---")
        part_jets, part_mask = run_part_inference(
            events,
            config,
            args.part_checkpoint,
            args.part_config,
        )

        # Extract features for events with >= 4 jets
        part_events = events[part_mask]
        part_jets_sel = part_jets[part_mask]
        n_pass = int(ak.sum(part_mask))

        if n_pass > 0:
            feats = extract_hh_features(part_events, part_jets_sel)
            feats["_label"] = "ParT (L1)"
            feats["_n_pass"] = n_pass
            results.append(feats)

    if len(results) == 0:
        print("ERROR: No results to compare!")
        return

    print(f"\n3. Generating comparison plots...")
    plot_mhh_comparison(results)
    plot_higgs_masses(results)
    plot_efficiency_table(results, n_total)

    # Save NPZ with all results for downstream SBI training
    npz_path = os.path.join(PLOT_DIR, "..", "tagger_comparison_features.npz")
    npz_data = {}
    for r in results:
        prefix = r["_label"].replace(" ", "_").replace("(", "").replace(")", "").lower()
        for k, v in r.items():
            if k.startswith("_"):
                continue
            npz_data[f"{prefix}_{k}"] = v
    np.savez(npz_path, **npz_data)
    print(f"\n4. Saved features to {npz_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
