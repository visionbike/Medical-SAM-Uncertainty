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
        self.eps = 1e-6
        self.reduction = reduction

    def _iou(self, prd: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prd (Tensor): predict tensor in shape of (B, 1, H, W).
            tgt (Tensor): target tensor in shape of (B, 1, H, W).
        Returns:
            (Tensor): iou score of the single mask.
        """
        intersect = torch.dot(prd.view(-1), tgt.view(-1)) + self.eps
        union = torch.sum(prd) + torch.sum(tgt) + self.eps
        return intersect / union

    def forward(self, prd: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prd (Tensor): predicted tensor in shape of (B, C, H, W).
            tgt (Tensor): target tensor in shape of (B, C, H, W).
        Returns:
            (Tensor): iou scalar or tensor in shape of (C,).
        """
        _, C, _, _ = prd.shape
        iou_score = []
        for i in range(C):
            iou_score.append(self._iou(prd[:, i, :, :], tgt[:, i, :, :]))
        iou_score = torch.tensor(iou_score).float()
        if self.reduction == "mean":
            return iou_score.mean()
        elif self.reduction == "sum":
            return iou_score.sum()
        return iou_score
