"""
Custom SBI (Simulation-Based Inference) implementation.

Implements SNRE-B (Sequential Neural Ratio Estimation) from scratch,
bypassing the sbi library which is incompatible with Python 3.14.

Reference: Durkan et al. 2020, "On Contrastive Learning for Likelihood-Free Inference"

Components:
    - BoxUniform: uniform prior distribution
    - RatioEstimator: MLP classifier for log-ratio estimation
    - SNRETrainer: training loop with contrastive joint/marginal data
    - MCMCPosterior: Metropolis-Hastings sampler using learned ratio
    - train_snre(): end-to-end convenience function
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------


class BoxUniform:
    """
    Uniform prior over a box [low, high].

    Mimics sbi.utils.BoxUniform interface.

    Args:
        low: tensor of lower bounds, shape (dim,)
        high: tensor of upper bounds, shape (dim,)
    """

    def __init__(self, low, high):
        self.low = torch.as_tensor(low, dtype=torch.float32)
        self.high = torch.as_tensor(high, dtype=torch.float32)
        self.dim = len(self.low)
        self._log_prob_val = -torch.sum(torch.log(self.high - self.low)).item()

    def sample(self, shape=(1,)):
        if isinstance(shape, int):
            shape = (shape,)
        n = shape[0]
        return self.low + (self.high - self.low) * torch.rand(n, self.dim)

    def log_prob(self, theta):
        """Log probability. Returns -inf outside the box."""
        theta = torch.as_tensor(theta, dtype=torch.float32)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        inside = torch.all((theta >= self.low) & (theta <= self.high), dim=-1)
        lp = torch.full((theta.shape[0],), float("-inf"))
        lp[inside] = self._log_prob_val
        return lp


# ---------------------------------------------------------------------------
# Ratio Estimator (neural network)
# ---------------------------------------------------------------------------


class RatioEstimator(nn.Module):
    """
    MLP classifier for SNRE-B log-ratio estimation.

    Takes concatenated [x, theta] and outputs a logit f(x, theta).
    At optimality, f(x, theta) = log r(x, theta) = log [p(theta, x) / p(theta)p(x)].

    The posterior is then:
        p(theta | x) proportional to prior(theta) * exp(f(x, theta))

    Args:
        x_dim: dimensionality of observations x
        theta_dim: dimensionality of parameters theta (typically 1 for kappa_lambda)
        hidden_dims: list of hidden layer sizes
        dropout: dropout rate
    """

    def __init__(self, x_dim, theta_dim=1, hidden_dims=None, dropout=0.05):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        input_dim = x_dim + theta_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, theta):
        """
        Args:
            x: (batch, x_dim) observations
            theta: (batch, theta_dim) parameters
        Returns:
            logits: (batch,) log-ratio estimates
        """
        inp = torch.cat([x, theta], dim=-1)
        return self.net(inp).squeeze(-1)


# ---------------------------------------------------------------------------
# SNRE-B Trainer
# ---------------------------------------------------------------------------


class SNRETrainer:
    """
    Trains a RatioEstimator using the SNRE-B (contrastive) algorithm.

    Algorithm:
        1. Receive joint samples (theta_i, x_i) from simulator
        2. Create marginal pairs by permuting theta indices
        3. Train binary classifier: label=1 for joint, label=0 for marginal
        4. The learned logit approximates the log density ratio

    Args:
        prior: BoxUniform instance
        estimator: RatioEstimator (or None to auto-create)
        device: 'cpu', 'cuda', or 'mps'
        lr: learning rate
    """

    def __init__(self, prior, estimator=None, device="cpu", lr=1e-3):
        self.prior = prior
        self.estimator = estimator
        self.device = device
        self.lr = lr
        self._theta = []
        self._x = []

    def append_simulations(self, theta, x):
        """
        Add simulation pairs to the training buffer.

        Args:
            theta: (N, theta_dim) parameter values
            x: (N, x_dim) simulated observations
        """
        self._theta.append(torch.as_tensor(theta, dtype=torch.float32))
        self._x.append(torch.as_tensor(x, dtype=torch.float32))

    def train(
        self,
        n_epochs=200,
        batch_size=256,
        val_fraction=0.1,
        patience=20,
        verbose=True,
    ):
        """
        Train the ratio estimator.

        Returns:
            Trained RatioEstimator
        """
        theta_all = torch.cat(self._theta, dim=0)
        x_all = torch.cat(self._x, dim=0)
        n = len(theta_all)

        if theta_all.dim() == 1:
            theta_all = theta_all.unsqueeze(-1)
        if x_all.dim() == 1:
            x_all = x_all.unsqueeze(-1)

        x_dim = x_all.shape[-1]
        theta_dim = theta_all.shape[-1]

        # Auto-create estimator if needed
        if self.estimator is None:
            self.estimator = RatioEstimator(x_dim, theta_dim)
        self.estimator = self.estimator.to(self.device)

        # Train/val split
        n_val = max(1, int(n * val_fraction))
        n_train = n - n_val
        perm = torch.randperm(n)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        theta_train, x_train = theta_all[train_idx], x_all[train_idx]
        theta_val, x_val = theta_all[val_idx], x_all[val_idx]

        optimizer = torch.optim.Adam(
            self.estimator.parameters(), lr=self.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=self.lr * 0.01
        )

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(n_epochs):
            # ---- training ----
            self.estimator.train()
            train_loss = self._run_epoch(theta_train, x_train, batch_size, optimizer)

            # ---- validation ----
            self.estimator.eval()
            with torch.no_grad():
                val_loss = self._compute_loss(theta_val, x_val)

            scheduler.step()

            if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
                print(
                    f"  Epoch {epoch:4d}/{n_epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            # Early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {
                    k: v.clone() for k, v in self.estimator.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.estimator.load_state_dict(best_state)

        self.estimator.eval()
        if verbose:
            print(f"  Best val loss: {best_val_loss:.4f}")
        return self.estimator

    def _run_epoch(self, theta, x, batch_size, optimizer):
        """Run one training epoch."""
        n = len(theta)
        # Global permutation for marginal pairs (important: must be across
        # full dataset, not within-batch, to break joint structure)
        global_perm = torch.randperm(n)
        theta_marginal_all = theta[global_perm]

        batch_perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = batch_perm[i : i + batch_size]
            theta_batch = theta[idx].to(self.device)
            x_batch = x[idx].to(self.device)
            theta_marginal = theta_marginal_all[idx].to(self.device)

            # Joint labels = 1, marginal labels = 0
            logits_joint = self.estimator(x_batch, theta_batch)
            logits_marginal = self.estimator(x_batch, theta_marginal)

            loss_joint = F.binary_cross_entropy_with_logits(
                logits_joint, torch.ones_like(logits_joint)
            )
            loss_marginal = F.binary_cross_entropy_with_logits(
                logits_marginal, torch.zeros_like(logits_marginal)
            )
            loss = loss_joint + loss_marginal

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.estimator.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _compute_loss(self, theta, x):
        """Compute validation loss (no gradient, averaged over multiple shuffles)."""
        theta_d = theta.to(self.device)
        x_d = x.to(self.device)

        total = 0.0
        n_repeats = 3
        for _ in range(n_repeats):
            perm_idx = torch.randperm(len(theta_d))
            theta_marginal = theta_d[perm_idx]
            logits_joint = self.estimator(x_d, theta_d)
            logits_marginal = self.estimator(x_d, theta_marginal)
            loss = F.binary_cross_entropy_with_logits(
                logits_joint, torch.ones_like(logits_joint)
            ) + F.binary_cross_entropy_with_logits(
                logits_marginal, torch.zeros_like(logits_marginal)
            )
            total += loss.item()
        return total / n_repeats

    def build_posterior(self, estimator=None):
        """
        Build an MCMCPosterior from the trained estimator.

        Args:
            estimator: trained RatioEstimator (uses self.estimator if None)

        Returns:
            MCMCPosterior instance
        """
        if estimator is None:
            estimator = self.estimator
        return MCMCPosterior(estimator, self.prior, device=self.device)


# ---------------------------------------------------------------------------
# MCMC Posterior Sampler
# ---------------------------------------------------------------------------


class MCMCPosterior:
    """
    Metropolis-Hastings MCMC sampler using a learned log-ratio.

    Samples from:
        p(theta | x_obs) proportional to prior(theta) * exp(f(x_obs, theta))

    where f is the learned log density ratio from RatioEstimator.

    Args:
        estimator: trained RatioEstimator
        prior: BoxUniform instance
        device: device for estimator inference
    """

    def __init__(self, estimator, prior, device="cpu"):
        self.estimator = estimator.to(device)
        self.estimator.eval()
        self.prior = prior
        self.device = device

    def _log_posterior(self, theta, x_obs):
        """Unnormalised log posterior: log prior + log ratio."""
        theta_t = torch.as_tensor(theta, dtype=torch.float32).unsqueeze(0)
        x_t = torch.as_tensor(x_obs, dtype=torch.float32).unsqueeze(0)

        log_prior = self.prior.log_prob(theta_t).item()
        if log_prior == float("-inf"):
            return float("-inf")

        with torch.no_grad():
            theta_t = theta_t.to(self.device)
            x_t = x_t.to(self.device)
            log_ratio = self.estimator(x_t, theta_t).item()

        return log_prior + log_ratio

    def sample(
        self,
        n_samples,
        x,
        proposal_width=None,
        burn_in=1000,
        thin=5,
        init_theta=None,
        verbose=False,
    ):
        """
        Draw posterior samples via Metropolis-Hastings.

        Args:
            n_samples: number of posterior samples to return
            x: observed data, shape (x_dim,) or (1, x_dim)
            proposal_width: std of Gaussian proposal (auto-set if None)
            burn_in: number of initial samples to discard
            thin: keep every thin-th sample
            init_theta: starting point (prior mean if None)
            verbose: print acceptance rate

        Returns:
            samples: tensor (n_samples, theta_dim)
        """
        x_obs = torch.as_tensor(x, dtype=torch.float32)
        if x_obs.dim() == 2:
            x_obs = x_obs.squeeze(0)

        dim = self.prior.dim

        if init_theta is None:
            init_theta = ((self.prior.low + self.prior.high) / 2).numpy()
        else:
            init_theta = np.asarray(init_theta, dtype=np.float32)

        if proposal_width is None:
            proposal_width = 0.1 * (self.prior.high - self.prior.low).numpy()

        total_needed = burn_in + n_samples * thin
        samples = np.zeros((total_needed, dim), dtype=np.float32)
        current = init_theta.copy()
        current_lp = self._log_posterior(current, x_obs)
        n_accept = 0

        for i in range(total_needed):
            proposal = current + proposal_width * np.random.randn(dim)
            proposal_lp = self._log_posterior(proposal, x_obs)

            log_alpha = proposal_lp - current_lp
            if np.log(np.random.rand()) < log_alpha:
                current = proposal
                current_lp = proposal_lp
                n_accept += 1

            samples[i] = current

        accept_rate = n_accept / total_needed
        if verbose:
            print(f"  MCMC acceptance rate: {accept_rate:.3f}")

        # Discard burn-in and thin
        samples = samples[burn_in::thin][:n_samples]
        return torch.from_numpy(samples)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def train_snre(
    prior,
    simulator,
    n_sim=10000,
    n_epochs=200,
    batch_size=256,
    hidden_dims=None,
    device="cpu",
    lr=1e-3,
    patience=20,
    verbose=True,
):
    """
    End-to-end SNRE training.

    Args:
        prior: BoxUniform instance
        simulator: callable theta -> x (both tensors)
        n_sim: number of simulations
        n_epochs: training epochs
        batch_size: batch size
        hidden_dims: list of hidden layer sizes
        device: 'cpu', 'cuda', or 'mps'
        lr: learning rate
        patience: early stopping patience
        verbose: print progress

    Returns:
        (posterior, estimator): MCMCPosterior and trained RatioEstimator
    """
    if verbose:
        print(f"Simulating {n_sim} samples...")
    theta = prior.sample((n_sim,))
    x = simulator(theta)

    if verbose:
        print(f"  theta shape: {theta.shape}, x shape: {x.shape}")

    estimator = RatioEstimator(
        x_dim=x.shape[-1],
        theta_dim=theta.shape[-1],
        hidden_dims=hidden_dims,
    )

    if verbose:
        print(f"Training SNRE-B ({n_epochs} epochs, batch_size={batch_size})...")
    trainer = SNRETrainer(prior, estimator=estimator, device=device, lr=lr)
    trainer.append_simulations(theta, x)
    trained = trainer.train(
        n_epochs=n_epochs,
        batch_size=batch_size,
        patience=patience,
        verbose=verbose,
    )

    posterior = MCMCPosterior(trained, prior, device=device)
    return posterior, trained


# ---------------------------------------------------------------------------
# Toy example (validation)
# ---------------------------------------------------------------------------


def run_toy_example(plot_dir=None):
    """
    Validate SNRE on 1D Gaussian: x ~ N(theta, 1), prior theta ~ U(-5, 5).

    The analytic posterior for x_obs is a truncated normal.

    Args:
        plot_dir: directory to save plots (None = don't save)

    Returns:
        dict with posterior mean, std, KS statistic, KS p-value
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import truncnorm, ks_2samp

    print("=" * 60)
    print("SBI Toy Example: Custom SNRE-B on 1D Gaussian")
    print("=" * 60)

    prior = BoxUniform(low=torch.tensor([-5.0]), high=torch.tensor([5.0]))
    sigma = 1.0

    def simulator(theta):
        return theta + sigma * torch.randn_like(theta)

    # Train with enough data for the ratio to converge
    posterior, estimator = train_snre(
        prior,
        simulator,
        n_sim=50000,
        n_epochs=300,
        batch_size=512,
        device="cpu",
        lr=1e-3,
        patience=40,
    )

    # Sample posterior for x_obs = 2.0
    x_obs = torch.tensor([2.0])
    print("\nSampling posterior for x_obs = 2.0...")
    samples = posterior.sample(10000, x=x_obs, burn_in=2000, thin=3, verbose=True)
    theta_np = samples.numpy().flatten()

    post_mean = np.mean(theta_np)
    post_std = np.std(theta_np)
    print(f"  Posterior mean: {post_mean:.3f} (expected ~2.0)")
    print(f"  Posterior std:  {post_std:.3f} (expected ~{sigma:.1f})")

    # Analytic: truncated normal
    a, b = (-5 - 2.0) / sigma, (5 - 2.0) / sigma
    analytic_samples = truncnorm.rvs(a, b, loc=2.0, scale=sigma, size=10000)
    ks_stat, ks_pval = ks_2samp(theta_np, analytic_samples)

    print(f"\nValidation:")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  KS p-value:   {ks_pval:.4f}")
    if ks_pval > 0.01:
        print("  PASSED: posterior consistent with analytic")
    else:
        print("  WARNING: posterior deviates from analytic")

    # Plot
    if plot_dir is not None:
        import os

        os.makedirs(plot_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Panel 0: prior vs posterior
        bins = np.linspace(-5, 5, 60)
        axes[0].hist(
            prior.sample((10000,)).numpy().flatten(),
            bins=bins,
            alpha=0.4,
            label="Prior",
            density=True,
        )
        axes[0].hist(
            theta_np,
            bins=bins,
            alpha=0.6,
            label="Posterior",
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )
        axes[0].axvline(2.0, color="red", linestyle="--", label="$x_{obs}$")
        axes[0].axvline(
            post_mean, color="blue", linestyle=":", label=f"mean={post_mean:.2f}"
        )
        axes[0].set_xlabel("$\\theta$")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Prior vs Posterior")
        axes[0].legend()

        # Panel 1: SNRE vs analytic
        x_grid = np.linspace(-5, 5, 200)
        analytic_pdf = truncnorm.pdf(x_grid, a, b, loc=2.0, scale=sigma)
        axes[1].plot(
            x_grid, analytic_pdf, "r-", linewidth=2, label="Analytic posterior"
        )
        axes[1].hist(
            theta_np,
            bins=60,
            density=True,
            alpha=0.5,
            label="SNRE posterior",
            edgecolor="black",
            linewidth=0.5,
        )
        axes[1].set_xlabel("$\\theta$")
        axes[1].set_ylabel("Density")
        axes[1].set_title(f"SNRE vs Analytic (KS p={ks_pval:.3f})")
        axes[1].legend()

        # Panel 2: trace plot
        axes[2].plot(theta_np[:500], alpha=0.7, linewidth=0.5)
        axes[2].axhline(2.0, color="red", linestyle="--", alpha=0.5)
        axes[2].set_xlabel("Sample index")
        axes[2].set_ylabel("$\\theta$")
        axes[2].set_title("MCMC Trace (first 500)")

        plt.tight_layout()
        path = os.path.join(plot_dir, "sbi_toy_validation.png")
        plt.savefig(path, dpi=150)
        print(f"\nSaved plot: {path}")
        plt.close()

    print("=" * 60)
    return {
        "posterior_mean": post_mean,
        "posterior_std": post_std,
        "ks_stat": ks_stat,
        "ks_pval": ks_pval,
    }


if __name__ == "__main__":
    result = run_toy_example(plot_dir="Updates/sbi-test/plots")
