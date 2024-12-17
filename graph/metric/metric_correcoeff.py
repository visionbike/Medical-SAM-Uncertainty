import torch
import torch.nn as nn

__all__ = [
    "CorrCoeffMetric"
]

class CorrCoeffMetric(nn.Module):
    """
    Correlation coefficient between two tensors.
    """

    def __init__(self, num_steps: int = 100, reduction: str = "none") -> None:
        """
        Args:
            num_steps (int): number of steps to get samples from tensors in computing correlation coefficient.
            reduction (str): specifies the reduction to apply to the output: "none", "mean", "sum".
                "none": dice coeff for each channel (mask).
        """
        super().__init__()
        self.num_steps = num_steps
        self.reduction = reduction

    def _corr_coeff(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1 (Tensor): 1st 1D input tensor in shape (B, H, W).
            x2 (Tensor): 2nd 1D input tensor in shape (B, H, W).
        Returns:
            (Tensor): Correlation coefficient scalar.
        """
        x1_flatten = x1.flatten()[::self.num_steps]
        x2_flatten = x2.flatten()[::self.num_steps]
        covar = torch.corrcoef(torch.stack([x1_flatten, x2_flatten]))
        return covar[0][1] / (covar[0][0].sqrt() * covar[1][1].sqrt())

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1 (Tensor): 1st input tensor in shape of (B, C, H, W).
            x2 (Tensor): 2nd input tensor in shape of (B, C, H, W).
        Returns:
            (Tensor): iou scalar or tensor in shape of (B,).
        """
        C = x1.shape[1]
        corr_coeffs = []
        for i in range(C):
            corr_coeffs.append(self._corr_coeff(x1[:, i, :, :], x2[:, i, :, :]))
        corr_coeffs = torch.tensor(corr_coeffs).float()
        if self.reduction == "mean":
            return corr_coeffs.mean()
        elif self.reduction == "sum":
            return corr_coeffs.sum()
        return corr_coeffs
