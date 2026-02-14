#!/usr/bin/env python3
"""
Train SNRE on HH->4b event features for kappa_lambda inference.

This script provides two modes:
  1. Closure test: synthetic kappa_lambda reweighting of existing SM data
  2. Full training: real simulation samples at multiple kappa_lambda (when available)

The closure test uses a morphing approach: the m_HH spectrum shape changes
with kappa_lambda because the HH cross-section depends on the interference
between triangle and box diagrams.

Run with:
    conda run -n hep-root-ml python train_hh_sbi.py --mode closure
    conda run -n hep-root-ml python train_hh_sbi.py --mode closure --tagger l1_ext

Usage:
    python train_hh_sbi.py --mode closure [--tagger offline_pnet] [--n-kl-points 9]
"""

import os
import argparse
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sbi_models import (
    BoxUniform,
    RatioEstimator,
    SNRETrainer,
    MCMCPosterior,
)

sns.set_style("whitegrid")
PLOT_DIR = "Updates/sbi-test/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Physics: kappa_lambda-dependent m_HH spectrum morphing
# ---------------------------------------------------------------------------


def hh_cross_section_ratio(m_hh, kappa_lambda, kappa_lambda_ref=1.0):
    """
    Approximate ratio of dsigma/dm_HH at kappa_lambda vs kappa_lambda_ref.

    The HH cross-section depends on kl via triangle-box interference:
        dsigma/dm_HH ~ |A_box + kl * A_tri(m)|^2

    We model A_tri(m) as peaking near 2*m_H threshold (~300 GeV) and
    falling at high mass.

    Returns:
        weight array: w(m_HH) = dsigma(kl) / dsigma(kl_ref)
    """
    m_hh = np.asarray(m_hh, dtype=np.float64)

    # Triangle-to-box amplitude ratio, peaking near threshold
    m_threshold = 300.0  # GeV
    sigma_peak = 200.0

    r = np.exp(-0.5 * ((m_hh - m_threshold) / sigma_peak) ** 2)

    # Weight = |1 + kl * r|^2 / |1 + kl_ref * r|^2
    num = (1.0 + kappa_lambda * r) ** 2
    den = (1.0 + kappa_lambda_ref * r) ** 2
    weights = np.where(den > 1e-10, num / den, 1.0)
    return np.clip(weights, 0.01, 100.0)


def generate_synthetic_kl_sample(
    features_sm, kappa_lambda, feature_names, n_events=None, rng=None
):
    """Generate synthetic sample at given kl by importance resampling SM data."""
    if rng is None:
        rng = np.random.default_rng(42)

    m_hh = features_sm["m_hh"]
    weights = hh_cross_section_ratio(m_hh, kappa_lambda, kappa_lambda_ref=1.0)
    x = np.column_stack([features_sm[f] for f in feature_names])

    if n_events is not None:
        probs = weights / weights.sum()
        idx = rng.choice(len(x), size=n_events, replace=True, p=probs)
        return x[idx].astype(np.float32), None
    else:
        return x.astype(np.float32), weights.astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "m_hh",
    "pt_hh",
    "eta_hh",
    "m_h1",
    "m_h2",
    "pt_h1",
    "pt_h2",
    "delta_r_hh",
    "delta_eta_hh",
    "delta_phi_hh",
    "cos_theta_star",
]


def load_tagger_features(npz_path, tagger_prefix):
    """Load features for a specific tagger from the comparison NPZ."""
    data = np.load(npz_path)
    features = {}
    for name in FEATURE_NAMES:
        key = f"{tagger_prefix}_{name}"
        if key in data:
            features[name] = data[key]
        else:
            print(f"  Warning: key '{key}' not found")
    return features


def standardize_features(x, mean=None, std=None):
    """Standardize features to zero mean, unit variance."""
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0
    return (x - mean) / std, mean, std


# ---------------------------------------------------------------------------
# Event-level log-likelihood scan
# ---------------------------------------------------------------------------


def event_level_loglik_scan(model, x_events, kl_grid, device="cpu"):
    """
    Compute summed log-ratio across all events for each kl value.

    For N observed events x_1,...,x_N:
        log L(kl) = sum_i log r(x_i, kl)

    This is the proper unbinned likelihood from the SNRE.

    Args:
        model: trained RatioEstimator
        x_events: (N, d) standardized feature array
        kl_grid: array of kl values to scan
        device: torch device

    Returns:
        log_likes: (len(kl_grid),) array of log-likelihood values
    """
    model.eval()
    x_t = torch.from_numpy(x_events.astype(np.float32)).to(device)
    log_likes = []

    with torch.no_grad():
        for kl in kl_grid:
            theta_t = torch.full(
                (len(x_events), 1), kl, dtype=torch.float32, device=device
            )
            log_r = model(x_t, theta_t)
            log_likes.append(log_r.sum().item())

    log_likes = np.array(log_likes)
    log_likes -= log_likes.max()
    return log_likes


def extract_confidence_interval(kl_grid, log_likes, delta=0.5):
    """Extract confidence interval from profile log-likelihood.

    delta=0.5 -> 68% CL, delta=2.0 -> 95% CL
    """
    best_idx = np.argmax(log_likes)
    best_kl = kl_grid[best_idx]

    above = kl_grid[log_likes > -delta]
    if len(above) > 0:
        kl_lo = above[0]
        kl_hi = above[-1]
    else:
        kl_lo = kl_grid[0]
        kl_hi = kl_grid[-1]
    return best_kl, kl_lo, kl_hi


# ---------------------------------------------------------------------------
# Closure test
# ---------------------------------------------------------------------------


def run_closure_test(
    tagger_prefix="offline_pnet",
    n_kl_points=9,
    kl_range=(0.0, 3.0),
    n_events_per_kl=5000,
    n_epochs=200,
    batch_size=256,
    device="cpu",
):
    """
    Closure test: train SNRE on synthetic kl-morphed data,
    then verify recovery using event-level log-likelihood scans.
    """
    print("=" * 70)
    print("HH SBI Closure Test (event-level log-likelihood)")
    print(f"  Tagger: {tagger_prefix}")
    print(f"  kl range: {kl_range}, {n_kl_points} points")
    print(f"  Events per kl: {n_events_per_kl}")
    print("=" * 70)

    # 1. Load SM data
    npz_path = "Updates/sbi-test/tagger_comparison_features.npz"
    print(f"\n1. Loading features from {npz_path}...")
    features_sm = load_tagger_features(npz_path, tagger_prefix)
    n_total = len(features_sm["m_hh"])
    print(f"   Loaded {n_total} events, {len(FEATURE_NAMES)} features")

    # 2. Generate training data
    print(f"\n2. Generating synthetic samples...")
    kl_values = np.linspace(kl_range[0], kl_range[1], n_kl_points)
    print(f"   kl grid: {kl_values}")

    all_theta = []
    all_x = []
    rng = np.random.default_rng(42)

    for kl in kl_values:
        x_kl, _ = generate_synthetic_kl_sample(
            features_sm,
            kl,
            FEATURE_NAMES,
            n_events=n_events_per_kl,
            rng=rng,
        )
        theta_kl = np.full((n_events_per_kl, 1), kl, dtype=np.float32)
        all_theta.append(theta_kl)
        all_x.append(x_kl)
        print(f"   kl={kl:.2f}: m_HH mean={x_kl[:, 0].mean():.1f} GeV")

    theta_all = np.concatenate(all_theta, axis=0)
    x_all = np.concatenate(all_x, axis=0)

    # Standardize features
    x_all_std, feat_mean, feat_std = standardize_features(x_all)
    print(f"\n   Total: {len(x_all)} events, {x_all.shape[1]} features")

    # 3. Train SNRE
    print(f"\n3. Training SNRE-B...")
    prior = BoxUniform(
        low=torch.tensor([kl_range[0]]),
        high=torch.tensor([kl_range[1]]),
    )

    estimator = RatioEstimator(
        x_dim=x_all_std.shape[1],
        theta_dim=1,
        hidden_dims=[128, 128, 64],
    )
    trainer = SNRETrainer(prior, estimator=estimator, device=device, lr=1e-3)
    trainer.append_simulations(
        torch.from_numpy(theta_all),
        torch.from_numpy(x_all_std),
    )
    trained = trainer.train(
        n_epochs=n_epochs,
        batch_size=batch_size,
        patience=40,
        verbose=True,
    )

    # 4. Closure test: event-level log-likelihood scan
    print(f"\n4. Running closure tests (event-level log-likelihood)...")
    test_kl_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    kl_scan = np.linspace(kl_range[0], kl_range[1], 60)
    results = []

    for true_kl in test_kl_values:
        # Generate N pseudo-data events at this kl
        n_test = 500
        x_test, _ = generate_synthetic_kl_sample(
            features_sm,
            true_kl,
            FEATURE_NAMES,
            n_events=n_test,
            rng=np.random.default_rng(int(true_kl * 1000)),
        )
        x_test_std = ((x_test - feat_mean) / feat_std).astype(np.float32)

        # Event-level log-likelihood scan
        log_likes = event_level_loglik_scan(
            trained,
            x_test_std,
            kl_scan,
            device=device,
        )

        best_kl, kl_lo, kl_hi = extract_confidence_interval(
            kl_scan,
            log_likes,
            delta=0.5,
        )
        _, kl_lo95, kl_hi95 = extract_confidence_interval(
            kl_scan,
            log_likes,
            delta=2.0,
        )

        results.append(
            {
                "true_kl": true_kl,
                "best_kl": best_kl,
                "kl_68": (kl_lo, kl_hi),
                "kl_95": (kl_lo95, kl_hi95),
                "log_likes": log_likes,
                "kl_scan": kl_scan,
            }
        )

        print(
            f"   kl_true={true_kl:.1f} | "
            f"best={best_kl:.3f} | "
            f"68%=[{kl_lo:.2f}, {kl_hi:.2f}] | "
            f"95%=[{kl_lo95:.2f}, {kl_hi95:.2f}]"
        )

    # 5. Plots
    print(f"\n5. Generating plots...")
    _plot_closure(results, kl_range, tagger_prefix)
    _plot_morphing(features_sm, kl_values)
    _plot_loglik_grid(results, kl_range, tagger_prefix)
    _plot_likelihood_scan_sm(
        trained, features_sm, feat_mean, feat_std, kl_range, tagger_prefix, device
    )

    # 6. Save
    ckpt_dir = "Updates/sbi-test/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"snre_{tagger_prefix}.pth")
    torch.save(
        {
            "model_state": trained.state_dict(),
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "feature_names": FEATURE_NAMES,
            "kl_range": kl_range,
            "tagger": tagger_prefix,
        },
        ckpt_path,
    )
    print(f"\n6. Saved checkpoint: {ckpt_path}")

    print("\n" + "=" * 70)
    print("Closure test complete!")
    print("=" * 70)
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_closure(results, kl_range, tagger_prefix):
    """Best-fit vs true kl with confidence intervals."""
    fig, ax = plt.subplots(figsize=(7, 6))

    true_kls = [r["true_kl"] for r in results]
    best_kls = [r["best_kl"] for r in results]
    lo68 = [r["best_kl"] - r["kl_68"][0] for r in results]
    hi68 = [r["kl_68"][1] - r["best_kl"] for r in results]

    ax.errorbar(
        true_kls,
        best_kls,
        yerr=[lo68, hi68],
        fmt="o",
        capsize=5,
        markersize=8,
        label="SNRE best fit (68% CL)",
        color="steelblue",
        linewidth=2,
    )

    diag = np.linspace(kl_range[0], kl_range[1], 100)
    ax.plot(diag, diag, "k--", alpha=0.5, label="Perfect recovery")

    ax.set_xlabel("True $\\kappa_\\lambda$", fontsize=13)
    ax.set_ylabel("Recovered $\\kappa_\\lambda$", fontsize=13)
    ax.set_title(f"Closure Test ({tagger_prefix})", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(kl_range)
    ax.set_ylim(kl_range)
    ax.set_aspect("equal")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"closure_test_{tagger_prefix}.png")
    plt.savefig(path, dpi=150)
    print(f"   Saved: {path}")
    plt.close()


def _plot_morphing(features_sm, kl_values):
    """m_HH distributions at different kl values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    m_hh = features_sm["m_hh"]
    bins = np.linspace(200, 1200, 50)

    for kl in kl_values:
        weights = hh_cross_section_ratio(m_hh, kl, kappa_lambda_ref=1.0)
        ax.hist(
            m_hh,
            bins=bins,
            weights=weights,
            density=True,
            histtype="step",
            linewidth=2,
            label=f"$\\kappa_\\lambda$={kl:.1f}",
        )

    ax.set_xlabel("$m_{HH}$ [GeV]", fontsize=13)
    ax.set_ylabel("Normalised density", fontsize=13)
    ax.set_title("$m_{HH}$ morphing with $\\kappa_\\lambda$", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "mhh_kl_morphing.png")
    plt.savefig(path, dpi=150)
    print(f"   Saved: {path}")
    plt.close()


def _plot_loglik_grid(results, kl_range, tagger_prefix):
    """Log-likelihood profiles for each test kl value."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for i, r in enumerate(results):
        ax = axes[i]
        ax.plot(r["kl_scan"], r["log_likes"], "b-", linewidth=2)
        ax.axvline(
            r["true_kl"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"True: {r['true_kl']:.1f}",
        )
        ax.axvline(
            r["best_kl"], color="blue", linestyle=":", label=f"Best: {r['best_kl']:.2f}"
        )
        ax.axhline(-0.5, color="orange", linestyle="--", alpha=0.5)
        ax.axhline(-2.0, color="red", linestyle="--", alpha=0.3)
        ax.fill_betweenx(
            [-5, 0.5], r["kl_68"][0], r["kl_68"][1], alpha=0.1, color="steelblue"
        )
        ax.set_xlabel("$\\kappa_\\lambda$")
        if i == 0:
            ax.set_ylabel("$\\Delta \\log L$")
        ax.set_title(f"$\\kappa_\\lambda$ = {r['true_kl']:.1f}")
        ax.set_ylim(-5, 0.5)
        ax.legend(fontsize=8)

    plt.suptitle(f"Profile Likelihoods ({tagger_prefix})", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"loglik_grid_{tagger_prefix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   Saved: {path}")
    plt.close()


def _plot_likelihood_scan_sm(
    model, features_sm, feat_mean, feat_std, kl_range, tagger_prefix, device
):
    """Profile likelihood for SM pseudo-data (kl=1.0)."""
    x_test, _ = generate_synthetic_kl_sample(
        features_sm,
        1.0,
        FEATURE_NAMES,
        n_events=1000,
        rng=np.random.default_rng(999),
    )
    x_test_std = ((x_test - feat_mean) / feat_std).astype(np.float32)
    kl_grid = np.linspace(kl_range[0], kl_range[1], 80)

    log_likes = event_level_loglik_scan(model, x_test_std, kl_grid, device)
    best_kl, kl_lo, kl_hi = extract_confidence_interval(kl_grid, log_likes)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kl_grid, log_likes, "b-", linewidth=2)
    ax.axhline(-0.5, color="orange", linestyle="--", alpha=0.7, label="68% CL")
    ax.axhline(-2.0, color="red", linestyle="--", alpha=0.7, label="95% CL")
    ax.axvline(1.0, color="green", linestyle=":", linewidth=2, label="SM (kl=1)")
    ax.axvline(
        best_kl,
        color="blue",
        linestyle=":",
        linewidth=2,
        label=f"Best fit: {best_kl:.2f}",
    )
    ax.fill_betweenx(
        [-8, 0.5],
        kl_lo,
        kl_hi,
        alpha=0.15,
        color="steelblue",
        label=f"68%: [{kl_lo:.2f}, {kl_hi:.2f}]",
    )

    ax.set_xlabel("$\\kappa_\\lambda$", fontsize=13)
    ax.set_ylabel("$\\Delta \\log L(\\kappa_\\lambda)$", fontsize=13)
    ax.set_title(f"Profile Likelihood - SM pseudo-data ({tagger_prefix})", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(-8, 0.5)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"likelihood_scan_{tagger_prefix}.png")
    plt.savefig(path, dpi=150)
    print(f"   Saved: {path}")
    plt.close()

    print(
        f"\n   SM likelihood scan: best_kl={best_kl:.2f}, "
        f"68% CL=[{kl_lo:.2f}, {kl_hi:.2f}]"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="HH SBI Training")
    parser.add_argument("--mode", choices=["closure", "full"], default="closure")
    parser.add_argument(
        "--tagger",
        default="offline_pnet",
        choices=["offline_pnet", "l1_ng", "l1_ext", "part_l1"],
    )
    parser.add_argument("--n-kl-points", type=int, default=9)
    parser.add_argument("--n-events", type=int, default=5000)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.mode == "closure":
        run_closure_test(
            tagger_prefix=args.tagger,
            n_kl_points=args.n_kl_points,
            n_events_per_kl=args.n_events,
            n_epochs=args.n_epochs,
            device=args.device,
        )
    elif args.mode == "full":
        print("Full training mode not yet implemented.")
        print("Waiting for simulation samples from collaborator.")


if __name__ == "__main__":
    main()
