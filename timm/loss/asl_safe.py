# timm/loss/asl_safe.py
from timm.loss.asymmetric_loss import AsymmetricLossSingleLabel
import torch

class SafeASLSingleLabel(AsymmetricLossSingleLabel):
    """Accepts targets shaped [B], [B,1], or one-hot/soft [B,C]."""
    def forward(self, inputs, target, reduction=None):
        if target.ndim > 1:
            # [B,1]  ->  [B]
            if target.size(1) == 1:
                target = target.squeeze(1)
            # [B,C] (one-hot or MixUp soft targets) -> class indices via argmax
            else:
                target = torch.argmax(target, dim=1)
        return super().forward(inputs, target, reduction)
