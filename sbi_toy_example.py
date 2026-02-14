#!/usr/bin/env python3
"""
SBI toy example: Neural Ratio Estimation for a 1D Gaussian.

This validates the SBI library setup and demonstrates the SNRE workflow
before applying it to full HH->4b data. The toy problem:
    - Simulator: x ~ Normal(theta, sigma_fixed)
    - Prior: theta ~ Uniform(-5, 5)
    - Goal: infer posterior p(theta | x_obs)

Run with:
    conda run -n hep-root-ml python sbi_toy_example.py
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.distributions import Uniform

PLOT_DIR = "Updates/sbi-test/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def run_toy_sbi():
    print("=" * 60)
    print("SBI Toy Example: SNRE on 1D Gaussian")
    print("=" * 60)

    # -- step 0: check sbi is importable ------
    try:
        import sbi
        from sbi.inference import SNRE, simulate_for_sbi
        from sbi.utils import BoxUniform

        print(f"  sbi version: {sbi.__version__}")
    except ImportError as e:
        print(f"ERROR: cannot import sbi ({e})")
        print("Install with:  pip install sbi")
        return

    # -- step 1: define prior and simulator -----
    print("\n1. Setting up prior and simulator...")
    prior = BoxUniform(low=torch.tensor([-5.0]), high=torch.tensor([5.0]))
    sigma = 1.0  # fixed noise

    def simulator(theta):
        """theta: (N, 1) tensor -> x: (N, 1) tensor (noisy observation)."""
        return theta + sigma * torch.randn_like(theta)

    # -- step 2: simulate training data ---------
    print("2. Simulating training data (10 000 samples)...")
    n_sim = 10_000
    theta_samples = prior.sample((n_sim,))
    x_samples = simulator(theta_samples)

    print(f"   theta shape: {theta_samples.shape}, x shape: {x_samples.shape}")

    # -- step 3: train SNRE ---------------------
    print("3. Training SNRE...")
    inference = SNRE(prior=prior)
    inference.append_simulations(theta_samples, x_samples)
    density_estimator = inference.train(
        training_batch_size=256, max_num_epochs=50, show_train_summary=True
    )
    print("   Training complete.")

    # -- step 4: build posterior -----------------
    print("4. Building posterior...")
    posterior = inference.build_posterior(density_estimator)

    # -- step 5: sample from posterior given observation ----
    print("5. Sampling posterior for x_obs = 2.0...")
    x_obs = torch.tensor([[2.0]])
    n_posterior = 10_000
    theta_posterior = posterior.sample((n_posterior,), x=x_obs)
    theta_np = theta_posterior.numpy().flatten()

    post_mean = np.mean(theta_np)
    post_std = np.std(theta_np)
    print(f"   Posterior mean: {post_mean:.3f}")
    print(f"   Posterior std:  {post_std:.3f}")

    # Analytic posterior for Gaussian likelihood + uniform prior is
    # approximately Normal(x_obs, sigma) truncated to [-5, 5]
    print(f"   Expected (analytic): mean ~ 2.0, std ~ {sigma:.1f}")

    # -- step 6: plot results --------------------
    print("6. Creating plot...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # panel 0: prior vs posterior
    ax = axes[0]
    bins = np.linspace(-5, 5, 60)
    ax.hist(
        prior.sample((10000,)).numpy().flatten(),
        bins=bins,
        alpha=0.4,
        label="Prior",
        density=True,
    )
    ax.hist(
        theta_np,
        bins=bins,
        alpha=0.6,
        label="Posterior",
        density=True,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(2.0, color="red", linestyle="--", label="$x_{obs}$", linewidth=2)
    ax.axvline(
        post_mean, color="blue", linestyle=":", label=f"Post. mean={post_mean:.2f}"
    )
    ax.set_xlabel("$\\theta$")
    ax.set_ylabel("Density")
    ax.set_title("Prior vs Posterior")
    ax.legend()

    # panel 1: training data
    ax = axes[1]
    ax.scatter(theta_samples.numpy()[:500], x_samples.numpy()[:500], alpha=0.3, s=5)
    ax.set_xlabel("$\\theta$ (parameter)")
    ax.set_ylabel("$x$ (observation)")
    ax.set_title("Simulated Training Data")
    ax.plot([-5, 5], [-5, 5], "r--", alpha=0.5, label="$x = \\theta$")
    ax.legend()

    # panel 2: posterior samples vs analytic
    ax = axes[2]
    from scipy.stats import norm, truncnorm

    x_grid = np.linspace(-5, 5, 200)
    # truncated normal analytic posterior
    a, b = (-5 - 2.0) / sigma, (5 - 2.0) / sigma
    analytic = truncnorm.pdf(x_grid, a, b, loc=2.0, scale=sigma)
    ax.plot(x_grid, analytic, "r-", linewidth=2, label="Analytic posterior")
    ax.hist(
        theta_np,
        bins=60,
        density=True,
        alpha=0.5,
        label="SNRE posterior",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("$\\theta$")
    ax.set_ylabel("Density")
    ax.set_title("SNRE vs Analytic")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "sbi_toy_example.png")
    plt.savefig(path, dpi=150)
    print(f"   Saved: {path}")
    plt.close()

    # -- step 7: validation metrics ---------------
    # KS test between SNRE samples and analytic
    from scipy.stats import ks_2samp

    analytic_samples = truncnorm.rvs(a, b, loc=2.0, scale=sigma, size=10000)
    ks_stat, ks_pval = ks_2samp(theta_np, analytic_samples)
    print(f"\n7. Validation:")
    print(f"   KS statistic: {ks_stat:.4f}")
    print(f"   KS p-value:   {ks_pval:.4f}")
    if ks_pval > 0.01:
        print("   PASSED: SNRE posterior is consistent with analytic posterior")
    else:
        print(
            "   WARNING: SNRE posterior deviates from analytic (may need more training)"
        )

    print("\n" + "=" * 60)
    print("SBI toy example complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_toy_sbi()
