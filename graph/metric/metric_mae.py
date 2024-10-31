import torch
import torch.nn as nn

__all__ = [
    "MAEMetric"
]

class MAEMetric(nn.Module):
    """
    The implementation of entropy metric of each tensor's channel.
    """

    def __init__(self, reduction: str = "none", act: str = "none", scale: bool = False) -> None:
        """
        Args:
            reduction (str): specifies the reduction to apply to the output: "none", "sum", "mean".
                "none": entropy map for each channel (mask).
            act (str): specify activation function to apply: "none", "sigmoid", "softmax".
                "none": no activation function.
            scale (bool): whether to apply data scaling to range [0, 1].
        """
        super().__init__()
        self.reduction = reduction
        self.act = act
        self.scale = scale
        self.mae = nn.L1Loss(reduction=reduction)

    def forward(self, prd: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prd (Tensor): predicted tensor in shape of (B, C, H, W).
            tgt (Tensor): ground truth tensor in shape of (B, C, H, W).
        Returns:
            (Tensor): if `reduction` is "none", the same shape as the input.
        """
        if self.act == "sigmoid":
            prd = torch.sigmoid(prd)
        elif self.act == "softmax":
            prd = torch.softmax(prd, dim=1)

        x = self.mae(prd, tgt)

        if self.scale:
            if self.reduction == "none":
                x = (x - x.amin(dim=(-1, -2), keepdim=True)) / (x.amax(dim=(-1, -2), keepdim=True) - x.amin(dim=(-1, -2), keepdim=True))
            else:
                x = (x - x.min()) / (x.max() - x.min())
        return x
