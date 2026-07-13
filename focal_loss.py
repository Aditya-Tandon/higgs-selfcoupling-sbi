import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction="mean"):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits:  [Batch_Size, 1] - Raw output from model (NO SIGMOID applied yet)
        targets: [Batch_Size, 1] - Binary labels (0.0 or 1.0), same shape as logits
        """

        # 1. BCEWithLogitsLoss calculates -log(pt)
        # This is numerically stable because it combines Sigmoid + Log internally.
        # If you took torch.log(torch.sigmoid(logits)), you would get NaNs for large negative logits.
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # 2. Recover the probability of the true class (pt)
        # Since bce_loss = -log(pt), we can just exponentiate the negative bce_loss.
        pt = torch.exp(-bce_loss)

        # 3. Calculate the Focal Term (1 - pt)^gamma
        # As pt -> 1 (confident correct), this term -> 0.
        focal_term = (1 - pt) ** self.gamma

        # 4. Apply the loss components
        loss = focal_term * bce_loss

        # 5. Apply Alpha Balancing (optional)
        if self.alpha is not None:
            # If target=1, use alpha. If target=0, use (1-alpha).
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # 6. Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
