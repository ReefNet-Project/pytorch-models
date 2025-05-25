# losses/custom_losses.py
import math, torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------- #
# 1. Focal Loss (single-label)
# --------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean", eps=1e-7):
        super().__init__()
        self.gamma, self.alpha, self.reduction, self.eps = gamma, alpha, reduction, eps

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)                  # (B,C)
        p = logp.exp()
        pt = p.gather(1, targets.unsqueeze(1)).clamp_(self.eps, 1.0)
        log_pt = logp.gather(1, targets.unsqueeze(1))

        focal_term = (1. - pt) ** self.gamma
        if self.alpha is not None:
            at = logits.new_full((logits.size(1),), 1 - self.alpha)
            at[targets] = self.alpha
            loss = -at[targets] * focal_term.squeeze(1) * log_pt.squeeze(1)
        else:
            loss = -focal_term.squeeze(1) * log_pt.squeeze(1)

        return loss.mean() if self.reduction == "mean" else loss.sum()

# --------------------------------------------------------------------- #
# 2.  Class-Balanced Loss (CE & Focal) – Cui et al. CVPR 2019
# --------------------------------------------------------------------- #
def _cb_weights(samples_per_cls, beta):
    effective_n = 1.0 - torch.pow(beta, samples_per_cls.float())
    weights = (1. - beta) / (effective_n + 1e-8)
    return weights / weights.sum() * len(samples_per_cls)



class CBCE(nn.Module):
    def __init__(self, samples_per_cls, beta=0.999, reduction="mean"):
        super().__init__()
        self.register_buffer("w", _cb_weights(samples_per_cls, beta))
        self.reduction = reduction

    def forward(self, logits, targets):
        weight = self.w.to(logits.device, dtype=logits.dtype)   # ← NEW
        return F.cross_entropy(logits, targets,
                            weight=weight, reduction=self.reduction)

class CBFocal(nn.Module):
    def __init__(self, samples_per_cls, beta=0.999, gamma=2.0, reduction="mean"):
        super().__init__()
        self.w = _cb_weights(samples_per_cls, beta)
        self.gamma, self.reduction = gamma, reduction

    def forward(self, logits, targets):
        weight = self.w.to(logits.device, dtype=logits.dtype)
        ce = F.cross_entropy(logits, targets,
                            reduction='none', weight=weight)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean() if self.reduction == "mean" else focal.sum()
