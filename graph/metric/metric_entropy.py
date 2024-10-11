import torch
import torch.nn as nn

__all__ = [
    "EntropyMetric"
]

class EntropyMetric(nn.Module):
    """
    The implementation of entropy metric of each tensor's channel.
    """

    def __init__(self, reduction: str = "none", act: str = "none") -> None:
        """
        Args:
            reduction (str): specifies the reduction to apply to the output: "none", "sum", "mean".
                "none": entropy map for each channel (mask).
            act (str): specify activation function to apply: "none", "sigmoid", "softmax".
                "none": no activation function.
        """
        super().__init__()
        self.reduction = reduction
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): input tensor in shape of (B, C, H, W).
        Returns:
            (Tensor): if `reduction` is "none", the same shape as the input.
        """
        if self.act == "sigmoid":
            x = torch.sigmoid(x)
        elif self.act == "softmax":
            x = torch.softmax(x, dim=1)

        if self.reduction == "none":
            x = -(x * torch.log(x))
        elif self.reduction == "sum":
            x = -torch.sum(x * torch.log(x), dim=1)
        elif self.reduction == "mean":
            x = -torch.mean(x * torch.log(x), dim=1)

        if self.reduction == "none":
            x = (x - x.amin(dim=(-1, -2), keepdim=True)) / x.amax(dim=(-1, -2), keepdim=True)
        else:
            x = (x - x.min()) / x.max()
        return x
