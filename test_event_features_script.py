#!/usr/bin/env python3
"""
Test script for event-level feature extraction.
Run with: conda run -n hep-root-ml python test_event_features_script.py
"""

import os
import sys
import json
import numpy as np
import awkward as ak
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from data_loading_helpers import load_and_prepare_data, apply_custom_cuts
from event_features import pair_jets_to_higgs, extract_hh_features


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

print("=" * 60)
print("Testing event-level feature extraction for SBI")
print("=" * 60)

os.makedirs("Updates/sbi-test/plots", exist_ok=True)

print("\n1. Loading configuration...")
with open("hh-bbbb-obj-config.json", "r") as f:
    config = json.load(f)
print(f"   Config loaded. Tree: {config['tree_name']}")

print("\n2. Loading small sample of data...")
file_pattern = config["file_pattern"]
print(f"   File pattern: {file_pattern}")

collections = ["Jet", "GenPart"]

events = load_and_prepare_data(
    file_pattern,
    config["tree_name"],
    collections,
    max_events=None,
    correct_pt=True,
    CONFIG=config,
)

print(f"   Loaded {len(events)} events")

print("\n3. Applying cuts and selecting jets...")
jets = events.Jet
n_jets_before = ak.sum(ak.num(jets, axis=1))
print(f"   Jets before cuts: {n_jets_before}")

jets = apply_custom_cuts(jets, config, "offline", kinematic_only=True)
n_jets_after = ak.sum(ak.num(jets, axis=1))
print(f"   Jets after cuts: {n_jets_after}")

event_mask = ak.num(jets, axis=1) >= 4
jets = jets[event_mask]
events = events[event_mask]

print(f"   Events with >= 4 jets: {len(jets)}")

print("\n4. Testing jet pairing...")
h1, h2 = pair_jets_to_higgs(jets)
hh = h1 + h2
print(f"   m_HH range: {float(ak.min(hh.mass)):.1f} - {float(ak.max(hh.mass)):.1f} GeV")
print(f"   Mean m_H1: {float(ak.mean(h1.mass)):.1f} GeV")
print(f"   Mean m_H2: {float(ak.mean(h2.mass)):.1f} GeV")

print("\n5. Extracting event-level features...")
features = extract_hh_features(events, jets)

print(f"   Extracted {len(features['m_hh'])} events")
print(f"   Feature names: {list(features.keys())}")
print(f"\n   Sample statistics:")
for key in ["m_hh", "pt_hh", "m_h1", "m_h2", "delta_r_hh"]:
    vals = features[key]
    print(
        f"     {key:15s}: mean={np.mean(vals):7.1f}, std={np.std(vals):7.1f}, min={np.min(vals):7.1f}, max={np.max(vals):7.1f}"
    )

print("\n6. Creating plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

plot_vars = [
    ("m_hh", "Di-Higgs Mass [GeV]", (200, 1000), 50),
    ("pt_hh", "Di-Higgs pT [GeV]", (0, 500), 50),
    ("m_h1", "Leading Higgs Mass [GeV]", (50, 200), 50),
    ("m_h2", "Subleading Higgs Mass [GeV]", (50, 200), 50),
    ("delta_r_hh", "DeltaR(H1, H2)", (0, 6), 50),
    ("n_jets", "Number of Jets", (4, 15), 11),
]

for ax, (var, label, xlim, bins) in zip(axes, plot_vars):
    data = features[var]
    mask = (data >= xlim[0]) & (data <= xlim[1])
    ax.hist(data[mask], bins=bins, alpha=0.7, edgecolor="black")
    ax.set_xlabel(label)
    ax.set_ylabel("Events")
    ax.set_xlim(xlim)
    if var in ["m_h1", "m_h2"]:
        ax.axvline(125, color="red", linestyle="--", alpha=0.5, label="$m_H = 125$ GeV")
        ax.legend()

plt.tight_layout()
plt.savefig("Updates/sbi-test/plots/event_features_test.png", dpi=150)
print("   Saved: Updates/sbi-test/plots/event_features_test.png")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

axes2[0].scatter(features["m_h1"], features["m_h2"], alpha=0.3, s=10)
axes2[0].axhline(125, color="red", linestyle="--", alpha=0.5)
axes2[0].axvline(125, color="red", linestyle="--", alpha=0.5)
axes2[0].set_xlabel("$m_{H1}$ [GeV]")
axes2[0].set_ylabel("$m_{H2}$ [GeV]")
axes2[0].set_title("Reconstructed Higgs Masses")
axes2[0].set_xlim(0, 500)
axes2[0].set_ylim(0, 500)

axes2[1].hist2d(
    features["m_hh"],
    features["pt_hh"],
    bins=50,
    cmap="viridis",
    range=[[0, 1000], [0, 500]],
)
axes2[1].set_xlabel("$m_{HH}$ [GeV]")
axes2[1].set_ylabel("$p_T^{HH}$ [GeV]")
axes2[1].set_title("$m_{HH}$ vs $p_T^{HH}$")
axes2[1].set_xlim(0, 1000)
axes2[1].set_ylim(0, 500)
plt.colorbar(axes2[1].collections[0], ax=axes2[1], label="Events")

plt.tight_layout()
plt.savefig("Updates/sbi-test/plots/mhh_reconstruction.png", dpi=150)
print("   Saved: Updates/sbi-test/plots/mhh_reconstruction.png")

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)
print("\nSummary:")
print(f"  - Processed {len(features['m_hh'])} events")
print(
    f"  - m_HH range: {np.min(features['m_hh']):.1f} - {np.max(features['m_hh']):.1f} GeV"
)
print(
    f"  - Mean m_H1: {np.mean(features['m_h1']):.1f} +/- {np.std(features['m_h1']):.1f} GeV"
)
print(
    f"  - Mean m_H2: {np.mean(features['m_h2']):.1f} +/- {np.std(features['m_h2']):.1f} GeV"
)
print(f"  - Saved 2 plots to Updates/sbi-test/plots/")
print("\nEvent feature extraction is working correctly!")
