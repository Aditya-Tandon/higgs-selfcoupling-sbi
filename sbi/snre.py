"""
SNRE-B (Sequential Neural Ratio Estimation, contrastive) implemented from scratch.

Ported from the stale ``sbi-test`` branch (sbi_models.py) onto ``main``; the only
edits are trimming the bundled matplotlib toy and grouping the inference helpers
(``event_level_loglik_scan``, ``extract_confidence_interval``) here so the whole
ratio-estimation stack lives in one module. Depends only on torch / numpy.

At optimality the network logit f(x, theta) = log [p(x|theta)/p(x)], so for an
unbinned set of observed events x_1..x_N the profile likelihood is
    log L(theta) = sum_i f(x_i, theta)  (+ theta-independent const).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoxUniform:
    """Uniform prior over a box [low, high] (subset of sbi.utils.BoxUniform API)."""

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
        theta = torch.as_tensor(theta, dtype=torch.float32)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        inside = torch.all((theta >= self.low) & (theta <= self.high), dim=-1)
        lp = torch.full((theta.shape[0],), float("-inf"))
        lp[inside] = self._log_prob_val
        return lp


class RatioEstimator(nn.Module):
    """MLP classifier for SNRE-B: input [x, theta] -> scalar logit f(x, theta)."""

    def __init__(self, x_dim, theta_dim=1, hidden_dims=None, dropout=0.05):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 64]
        layers, prev = [], x_dim + theta_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, theta):
        return self.net(torch.cat([x, theta], dim=-1)).squeeze(-1)


class SNRETrainer:
    """Contrastive SNRE-B trainer: joint (theta_i, x_i) vs marginal (perm theta, x_i)."""

    def __init__(self, prior, estimator=None, device="cpu", lr=1e-3):
        self.prior = prior
        self.estimator = estimator
        self.device = device
        self.lr = lr
        self._theta, self._x = [], []

    def append_simulations(self, theta, x):
        self._theta.append(torch.as_tensor(theta, dtype=torch.float32))
        self._x.append(torch.as_tensor(x, dtype=torch.float32))

    def train(self, n_epochs=200, batch_size=256, val_fraction=0.1, patience=20,
              verbose=True):
        theta_all = torch.cat(self._theta, 0)
        x_all = torch.cat(self._x, 0)
        if theta_all.dim() == 1:
            theta_all = theta_all.unsqueeze(-1)
        if x_all.dim() == 1:
            x_all = x_all.unsqueeze(-1)
        n = len(theta_all)
        if self.estimator is None:
            self.estimator = RatioEstimator(x_all.shape[-1], theta_all.shape[-1])
        self.estimator = self.estimator.to(self.device)

        n_val = max(1, int(n * val_fraction))
        perm = torch.randperm(n)
        tr, va = perm[:n - n_val], perm[n - n_val:]
        theta_tr, x_tr = theta_all[tr], x_all[tr]
        theta_va, x_va = theta_all[va], x_all[va]

        opt = torch.optim.Adam(self.estimator.parameters(), lr=self.lr,
                               weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs,
                                                         eta_min=self.lr * 0.01)
        best, best_state, bad = float("inf"), None, 0
        for epoch in range(n_epochs):
            self.estimator.train()
            tl = self._run_epoch(theta_tr, x_tr, batch_size, opt)
            self.estimator.eval()
            with torch.no_grad():
                vl = self._compute_loss(theta_va, x_va)
            sch.step()
            if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
                print(f"  epoch {epoch:4d}/{n_epochs} train={tl:.4f} val={vl:.4f} "
                      f"lr={sch.get_last_lr()[0]:.2e}", flush=True)
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.clone()
                              for k, v in self.estimator.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"  early stop @ epoch {epoch}", flush=True)
                    break
        if best_state is not None:
            self.estimator.load_state_dict(best_state)
        self.estimator.eval()
        if verbose:
            print(f"  best val loss: {best:.4f}", flush=True)
        return self.estimator

    def _run_epoch(self, theta, x, batch_size, opt):
        n = len(theta)
        theta_marg_all = theta[torch.randperm(n)]
        bperm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = bperm[i:i + batch_size]
            tb = theta[idx].to(self.device)
            xb = x[idx].to(self.device)
            tm = theta_marg_all[idx].to(self.device)
            lj = self.estimator(xb, tb)
            lm = self.estimator(xb, tm)
            loss = (F.binary_cross_entropy_with_logits(lj, torch.ones_like(lj))
                    + F.binary_cross_entropy_with_logits(lm, torch.zeros_like(lm)))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.estimator.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb += 1
        return tot / max(nb, 1)

    def _compute_loss(self, theta, x):
        td, xd = theta.to(self.device), x.to(self.device)
        tot = 0.0
        for _ in range(3):
            tm = td[torch.randperm(len(td))]
            lj = self.estimator(xd, td)
            lm = self.estimator(xd, tm)
            tot += (F.binary_cross_entropy_with_logits(lj, torch.ones_like(lj))
                    + F.binary_cross_entropy_with_logits(lm, torch.zeros_like(lm))).item()
        return tot / 3


def event_level_loglik_scan(model, x_events, kl_grid, device="cpu"):
    """log L(kl) = sum_i f(x_i, kl) over a grid of kl, normalised to max 0."""
    model.eval()
    x_t = torch.as_tensor(np.asarray(x_events, np.float32), device=device)
    out = []
    with torch.no_grad():
        for kl in kl_grid:
            theta = torch.full((len(x_t), 1), float(kl), device=device)
            out.append(model(x_t, theta).sum().item())
    out = np.asarray(out)
    return out - out.max()


def extract_confidence_interval(kl_grid, log_likes, delta=0.5):
    """68% CL (delta=0.5 -> Delta(-2lnL)=1) interval from a profile log-likelihood."""
    best_kl = kl_grid[int(np.argmax(log_likes))]
    above = kl_grid[log_likes > -delta]
    if len(above):
        return best_kl, float(above[0]), float(above[-1])
    return best_kl, float(kl_grid[0]), float(kl_grid[-1])
