import torch
import torch.nn as nn

__all__ = [
    "IouMetric"
]

class IouMetric(nn.Module):
    """
    Dice coefficient metric for all batches.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "none") -> None:
        """
        Args:
            eps (float): eps value.
            reduction (str): specifies the reduction to apply to the output: "none", "mean", "sum".
                "none": dice coeff for each channel (mask).
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def _iou(self, prd: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prd (Tensor): predict tensor in shape of (B, H, W).
            tgt (Tensor): target tensor in shape of (B, H, W).
        Returns:
            (Tensor): iou score of the single mask.
        """

        inter = (prd & tgt).sum((1, 2))
        union = (prd | tgt).sum((1, 2))
        return (inter + self.eps) / (union + self.eps)

    def forward(self, prd: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prd (Tensor): predicted tensor in shape of (B, C, H, W).
            tgt (Tensor): target tensor in shape of (B, C, H, W).
        Returns:
            (Tensor): iou scalar or tensor in shape of (C,).
        """
        C = prd.shape[1]
        iou_score = []
        for i in range(C):
            iou_score.append(self._iou(prd[:, i, :, :].int(), tgt[:, i, :, :].int()))
        iou_score = torch.tensor(iou_score).float()
        if self.reduction == "mean":
            return iou_score.mean()
        elif self.reduction == "sum":
            return iou_score.sum()
        return iou_score
