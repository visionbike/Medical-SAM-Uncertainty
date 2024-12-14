import torch
import torch.nn as nn

__all__ = [
    "EntropyMetric"
]

class EntropyMetric(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): input tensor in shape of (B, C, H, W).
        Returns:
            (Tensor): if `reduction` is "none", the same shape as the input.
        """
        C = x.shape[1]

        if torch.max(x) > 0 or torch.min(x) < 0:
            x = torch.sigmoid(x)

        # get entropy
        x = -x * torch.log(x)
        for i in range(C):
            x[:, i, :, :] = (x[:, i, :, :] - x[:, i, :, :].min()) / (x[:, i, :, :].max() - x[:, i, :, :].min())
        if self.reduction == "mean":
            return x.mean(dim=(2, 3))
        elif self.reduction == "sum":
            return x.sum(dim=(2, 3))
        return x
