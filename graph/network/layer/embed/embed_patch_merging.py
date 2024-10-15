from typing import Type, Tuple
import torch
import torch.nn as nn
from ..conv import Conv2dBN

__all__ = [
    "PatchMerging"
]


class PatchMerging(nn.Module):
    def __init__(
            self,
            dim: int,
            out_dim: int,
            activation: Type[nn.Module],
            resolution: Tuple[int, int],
    ) -> None:
        """
        Args:
            dim (int): dimension of input tensor.
            out_dim (int): dimension of output tensor.
            activation (nn.Module): activation function.
            resolution (Tuple[int, int]): image resolution (size).
        """
        super().__init__()
        self.input_resolution = resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2dBN(dim, out_dim, 1, 1, 0)
        stride_c = 2
        if out_dim == 320 or out_dim == 448 or out_dim == 576:
            stride_c = 1
        self.conv2 = Conv2dBN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2dBN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): output tensor.
        """
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # B x C x H x W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x
