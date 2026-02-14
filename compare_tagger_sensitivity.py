#!/usr/bin/env python3
"""
Compare kappa_lambda sensitivity across taggers using trained SNRE models.

Loads pre-trained SNRE checkpoints for each tagger and compares:
  1. Profile likelihood widths at SM (kl=1.0)
  2. 68% and 95% CL intervals
  3. Expected sensitivity sigma(kl) as a function of number of events

Run with:
    conda run -n hep-root-ml python compare_tagger_sensitivity.py
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sbi_models import BoxUniform, RatioEstimator
from train_hh_sbi import (
    load_tagger_features,
    standardize_features,
    FEATURE_NAMES,
    generate_synthetic_kl_sample,
    event_level_loglik_scan,
    extract_confidence_interval,
    hh_cross_section_ratio,
)

sns.set_style("whitegrid")
PLOT_DIR = "Updates/sbi-test/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

TAGGERS = {
    "offline_pnet": {"label": "Offline PNet", "color": "#2196F3", "marker": "o"},
    "l1_ext": {"label": "L1 Ext", "color": "#FF5722", "marker": "s"},
    "l1_ng": {"label": "L1 NG", "color": "#4CAF50", "marker": "^"},
}


def load_model(tagger_prefix, kl_range=(0.0, 3.0)):
    """Load a trained SNRE model from checkpoint."""
    ckpt_path = f"Updates/sbi-test/checkpoints/snre_{tagger_prefix}.pth"
    if not os.path.exists(ckpt_path):
        return None, None, None, None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    estimator = RatioEstimator(
        x_dim=len(FEATURE_NAMES),
        theta_dim=1,
        hidden_dims=[128, 128, 64],
    )
    estimator.load_state_dict(ckpt["model_state"])
    estimator.eval()
    return estimator, ckpt["feat_mean"], ckpt["feat_std"], kl_range


def compute_sensitivity(
    model,
    features_sm,
    feat_mean,
    feat_std,
    kl_range,
    true_kl=1.0,
    n_events=500,
    n_trials=20,
    seed=42,
):
    """
    Compute expected sigma(kl) via repeated pseudo-experiments.

    For each trial:
      1. Draw n_events from the morphed distribution at true_kl
      2. Scan the log-likelihood
      3. Extract 68% CL width

    Returns:
        mean_width, std_width, all_best_fits, all_widths
    """
    kl_scan = np.linspace(kl_range[0], kl_range[1], 80)
    widths = []
    best_fits = []

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        x_test, _ = generate_synthetic_kl_sample(
            features_sm,
            true_kl,
            FEATURE_NAMES,
            n_events=n_events,
            rng=rng,
        )
        x_test_std = ((x_test - feat_mean) / feat_std).astype(np.float32)
        log_likes = event_level_loglik_scan(model, x_test_std, kl_scan)

        best_kl, kl_lo, kl_hi = extract_confidence_interval(
            kl_scan,
            log_likes,
            delta=0.5,
        )
        width = kl_hi - kl_lo
        widths.append(width)
        best_fits.append(best_kl)

    return np.mean(widths), np.std(widths), np.array(best_fits), np.array(widths)


def run_comparison():
    """Run full tagger sensitivity comparison."""
    print("=" * 70)
    print("Tagger Sensitivity Comparison for kappa_lambda")
    print("=" * 70)

    npz_path = "Updates/sbi-test/tagger_comparison_features.npz"

    # --- 1. Load models and data ---
    print("\n1. Loading models and data...")
    tagger_data = {}
    for tag, info in TAGGERS.items():
        model, feat_mean, feat_std, kl_range = load_model(tag)
        if model is None:
            print(f"   {info['label']}: no checkpoint, skipping")
            continue
        features = load_tagger_features(npz_path, tag)
        if "m_hh" not in features:
            print(f"   {info['label']}: no features, skipping")
            continue
        tagger_data[tag] = {
            "model": model,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "features": features,
            "n_events": len(features["m_hh"]),
            "kl_range": kl_range,
            **info,
        }
        print(f"   {info['label']}: loaded ({len(features['m_hh'])} events)")

    if len(tagger_data) < 2:
        print("\nNeed at least 2 taggers with trained models.")
        print("Run: python train_hh_sbi.py --mode closure --tagger <tag>")
        return

    # --- 2. SM likelihood comparison ---
    print("\n2. SM profile likelihood comparison...")
    fig, ax = plt.subplots(figsize=(9, 6))
    kl_scan = np.linspace(0.0, 3.0, 80)
    sm_results = {}

    for tag, td in tagger_data.items():
        x_test, _ = generate_synthetic_kl_sample(
            td["features"],
            1.0,
            FEATURE_NAMES,
            n_events=1000,
            rng=np.random.default_rng(42),
        )
        x_test_std = ((x_test - td["feat_mean"]) / td["feat_std"]).astype(np.float32)
        log_likes = event_level_loglik_scan(td["model"], x_test_std, kl_scan)

        best_kl, kl_lo, kl_hi = extract_confidence_interval(kl_scan, log_likes)
        _, kl_lo95, kl_hi95 = extract_confidence_interval(kl_scan, log_likes, delta=2.0)

        ax.plot(
            kl_scan,
            log_likes,
            color=td["color"],
            linewidth=2.5,
            label=f"{td['label']}: {best_kl:.2f} " f"[{kl_lo:.2f}, {kl_hi:.2f}]",
        )

        sm_results[tag] = {
            "best_kl": best_kl,
            "68_lo": kl_lo,
            "68_hi": kl_hi,
            "95_lo": kl_lo95,
            "95_hi": kl_hi95,
            "68_width": kl_hi - kl_lo,
            "95_width": kl_hi95 - kl_lo95,
        }
        print(
            f"   {td['label']}: best={best_kl:.2f}, "
            f"68%=[{kl_lo:.2f},{kl_hi:.2f}] (w={kl_hi-kl_lo:.2f}), "
            f"95%=[{kl_lo95:.2f},{kl_hi95:.2f}]"
        )

    ax.axhline(-0.5, color="orange", linestyle="--", alpha=0.7, label="68% CL")
    ax.axhline(-2.0, color="red", linestyle="--", alpha=0.7, label="95% CL")
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5, label="SM")
    ax.set_xlabel("$\\kappa_\\lambda$", fontsize=14)
    ax.set_ylabel("$\\Delta \\log L$", fontsize=14)
    ax.set_title("Profile Likelihood Comparison — SM pseudo-data (N=1000)", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(-8, 0.5)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "likelihood_comparison_taggers.png")
    plt.savefig(path, dpi=150)
    print(f"   Saved: {path}")
    plt.close()

    # --- 3. Sensitivity vs number of events ---
    print("\n3. Sensitivity vs number of events...")
    n_events_list = [100, 200, 500, 1000, 2000, 5000]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for tag, td in tagger_data.items():
        sigmas = []
        sigma_errs = []
        biases = []
        for n_ev in n_events_list:
            mean_w, std_w, bf, _ = compute_sensitivity(
                td["model"],
                td["features"],
                td["feat_mean"],
                td["feat_std"],
                td["kl_range"],
                true_kl=1.0,
                n_events=n_ev,
                n_trials=15,
                seed=tag.__hash__() % 10000,
            )
            sigmas.append(mean_w)
            sigma_errs.append(std_w)
            biases.append(np.mean(bf) - 1.0)
            print(
                f"   {td['label']}, N={n_ev}: "
                f"sigma(kl)={mean_w:.3f}+/-{std_w:.3f}, "
                f"bias={np.mean(bf)-1.0:.3f}"
            )

        axes[0].errorbar(
            n_events_list,
            sigmas,
            yerr=sigma_errs,
            marker=td["marker"],
            color=td["color"],
            linewidth=2,
            capsize=4,
            markersize=7,
            label=td["label"],
        )
        axes[1].plot(
            n_events_list,
            biases,
            marker=td["marker"],
            color=td["color"],
            linewidth=2,
            markersize=7,
            label=td["label"],
        )

    axes[0].set_xlabel("Number of events", fontsize=13)
    axes[0].set_ylabel("$\\sigma(\\kappa_\\lambda)$ [68% CL width]", fontsize=13)
    axes[0].set_title("Expected sensitivity", fontsize=14)
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Number of events", fontsize=13)
    axes[1].set_ylabel(
        "Bias ($\\hat{\\kappa}_\\lambda - \\kappa_\\lambda^{true}$)", fontsize=13
    )
    axes[1].set_title("Estimator bias at SM", fontsize=14)
    axes[1].set_xscale("log")
    axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "sensitivity_vs_events.png")
    plt.savefig(path, dpi=150)
    print(f"   Saved: {path}")
    plt.close()

    # --- 4. Closure comparison ---
    print("\n4. Closure comparison across taggers...")
    test_kl_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    kl_scan = np.linspace(0.0, 3.0, 60)

    fig, axes = plt.subplots(
        1, len(test_kl_values), figsize=(4 * len(test_kl_values), 4.5), sharey=True
    )

    for i, true_kl in enumerate(test_kl_values):
        ax = axes[i]
        for tag, td in tagger_data.items():
            x_test, _ = generate_synthetic_kl_sample(
                td["features"],
                true_kl,
                FEATURE_NAMES,
                n_events=500,
                rng=np.random.default_rng(int(true_kl * 1000)),
            )
            x_test_std = ((x_test - td["feat_mean"]) / td["feat_std"]).astype(
                np.float32
            )
            log_likes = event_level_loglik_scan(td["model"], x_test_std, kl_scan)
            ax.plot(
                kl_scan, log_likes, color=td["color"], linewidth=2, label=td["label"]
            )

        ax.axvline(true_kl, color="black", linestyle="--", alpha=0.5)
        ax.axhline(-0.5, color="orange", linestyle="--", alpha=0.3)
        ax.axhline(-2.0, color="red", linestyle="--", alpha=0.3)
        ax.set_xlabel("$\\kappa_\\lambda$")
        if i == 0:
            ax.set_ylabel("$\\Delta \\log L$")
        ax.set_title(f"$\\kappa_\\lambda^{{true}}$ = {true_kl:.1f}")
        ax.set_ylim(-6, 0.5)
        if i == len(test_kl_values) - 1:
            ax.legend(fontsize=8)

    plt.suptitle("Profile Likelihood: Tagger Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "closure_comparison_taggers.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   Saved: {path}")
    plt.close()

    # --- 5. Summary table ---
    print("\n" + "=" * 70)
    print("Summary: SM Sensitivity (N=1000 events)")
    print("=" * 70)
    print(f"{'Tagger':<20} {'Best fit':>10} {'68% width':>12} {'95% width':>12}")
    print("-" * 54)
    for tag, td in tagger_data.items():
        r = sm_results[tag]
        print(
            f"{td['label']:<20} {r['best_kl']:>10.3f} "
            f"{r['68_width']:>12.3f} {r['95_width']:>12.3f}"
        )
    print("=" * 70)

    return sm_results


if __name__ == "__main__":
    run_comparison()
