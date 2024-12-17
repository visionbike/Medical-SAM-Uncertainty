import torch
import torch.nn as nn
import torch.autograd as autograd

__all__ = [
    "DiceCoeffMetric"
]


class DiceCoeff(autograd.Function):
    """
    Dice coeff for the individual example
    """

    def forward(self, prd: torch.Tensor, tgt: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
        self.save_for_backward(prd, tgt)
        self.inter = torch.dot(prd.view(-1), tgt.view(-1))
        self.union = torch.sum(prd) + torch.sum(tgt) + eps
        return (2 * self.inter.float() + eps) / self.union.float()

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        prd, tgt = self.saved_variables
        grad_prd = grad_tgt = None

        if self.needs_input_grad[0]:
            grad_prd = grad_tgt * 2 * (tgt * self.union - self.inter) / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_tgt = None

        return grad_prd, grad_tgt


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
            prd (Tensor): predicted mask in shape of (B, H, W).
            tgt (Tensor): target mask in shape of (B, H, W).
        Returns:
            (Tensor): dice score of the single mask.
        """
        s = torch.zeros(1, dtype=torch.float32, device=prd.device)
        i = 0
        for i, c in enumerate(zip(prd, tgt)):
            s += DiceCoeff().forward(prd[i], tgt[i], eps=self.eps)
        return s / (i + 1)

    def forward(self, prd: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prd (Tensor): predicted tensor in shape of (B, C, H, W).
            tgt (Tensor): target tensor in shape of (B, C, H, W).
        Returns:
            (Tensor): dice coefficient scalar or tensor in shape of (C,).
        """
        C = prd.shape[1]
        dice_coeffs = []
        for i in range(C):
            dice_coeffs.append(self._dice_coeff(prd[:, i, :, :], tgt[:, i, :, :]))
        dice_coeffs = torch.tensor(dice_coeffs).float()
        if self.reduction == "mean":
            return dice_coeffs.mean()
        elif self.reduction == "sum":
            return dice_coeffs.sum()
        return dice_coeffs
