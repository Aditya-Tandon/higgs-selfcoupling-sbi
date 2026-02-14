#!/usr/bin/env python3
"""
Compare m_HH resolution across different tagger configurations.

Loads offline (PNet), L1NG, and L1ext jet collections, reconstructs
di-Higgs via D_HH pairing for each, and produces comparison plots.

Run with:
    conda run -n hep-root-ml python run_tagger_comparison.py
"""

import os
import json
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
            color=colours[i],
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
            color=colours[i],
        )
    ax.set_xlabel("$m_{HH}$ [GeV]")
    ax.set_ylabel("Density")
    ax.set_title("Normalised $m_{HH}$")
    ax.legend()

    # ---- panel 2: resolution bar chart ----
    ax = axes[2]
    labels, means, stds, resolutions = [], [], [], []
    for r in results:
        m = r["m_hh"]
        in_range = (m >= 200) & (m <= 1200)
        mu = np.mean(m[in_range])
        sigma = np.std(m[in_range])
        labels.append(r["_label"])
        means.append(mu)
        stds.append(sigma)
        resolutions.append(sigma / mu)

    x_pos = np.arange(len(labels))
    bars = ax.bar(
        x_pos,
        resolutions,
        color=[colours[i] for i in range(len(labels))],
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
        ax.scatter(r["m_h1"], r["m_h2"], alpha=0.15, s=5, color=colours[i])
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
    """Print / save table of event-selection efficiencies."""
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
    print("=" * 60)
    print("Tagger Comparison: m_HH Resolution Study")
    print("=" * 60)

    with open("hh-bbbb-obj-config.json") as f:
        config = json.load(f)

    # Collections to compare
    keys_to_compare = [
        ("offline", "Offline (PNet)"),
        ("l1ng", "L1 NG"),
        ("l1ext", "L1 Ext"),
    ]

    # Determine which collections we need
    collection_names = set()
    for key, _ in keys_to_compare:
        collection_names.add(config[key]["collection_name"])
    # Always need GenPart for reference
    collections = list(collection_names) + ["GenPart"]

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
