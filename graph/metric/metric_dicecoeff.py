import torch
import torch.nn as nn

__all__ = [
    "DiceCoeffMetric"
]


class DiceCoeffMetric(nn.Module):
    """
    Dice coefficient metric for all batches.
    """
    def __init__(self, eps: float = 1e-6, reduction: str = "none") -> None:
        """
        Args:
            eps (float): epsilon value.
            reduction (str): specifies the reduction to apply to the output: "none", "mean", "sum".
                "none": dice coeff for each channel (mask).
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def _dice_coeff(self, prd: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prd (Tensor): predicted mask in shape of (B, 1, H, W).
            tgt (Tensor): target mask in shape of (B, 1, H, W).
        Returns:
            (Tensor): dice score of the single mask.
        """
        intersect = torch.dot(prd.view(-1), tgt.view(-1))
        union = torch.sum(prd) + torch.sum(tgt) + self.eps
        return (2 * intersect.float() + self.eps) / union.float()

    def forward(self, prd: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prd (Tensor): predicted tensor in shape of (B, C, H, W).
            tgt (Tensor): target tensor in shape of (B, C, H, W).
        Returns:
            (Tensor): dice coefficient scalar or tensor in shape of (C,).
        """
        _, C, _, _ = prd.shape
        dice_coeff = []
        for i in range(C):
            dice_coeff.append(self._dice_coeff(prd[:, i, :, :], tgt[:, i, :, :]))
        dice_coeff = torch.tensor(dice_coeff).float()
        if self.reduction == "mean":
            return dice_coeff.mean()
        elif self.reduction == "sum":
            return dice_coeff.sum()
        return dice_coeff
