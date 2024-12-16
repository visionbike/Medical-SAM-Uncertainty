import torch
import torch.nn as nn

__all__ = [
    "MAEMetric"
]

class MAEMetric(nn.Module):
    """
    The implementation of entropy metric of each tensor's channel.
    """

    def __init__(self, reduction: str = "none") -> None:
        """
        Args:
            reduction (str): specifies the reduction to apply to the output: "none", "sum", "mean".
                "none": entropy map for each channel (mask).
        """
        super().__init__()
        self.reduction = reduction
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, prd: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prd (Tensor): predicted tensor in shape of (B, C, H, W).
            tgt (Tensor): ground truth tensor in shape of (B, C, H, W).
        Returns:
            (Tensor): if `reduction` is "none", the same shape as the input.
        """
        if torch.max(prd) > 0 or torch.min(prd) < 0:
            prd = torch.sigmoid(prd)

        x = self.mae(prd, tgt)

        if self.reduction == "mean":
            return x.mean(dim=(2, 3))
        elif self.reduction == "sum":
            return x.sum(dim=(2, 3))
        return x
