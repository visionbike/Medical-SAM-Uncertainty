from typing import Type
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .conv2d_bn import Conv2dBN

__all__ = [
    "MBConv",
]


class MBConv(nn.Module):
    """
    The MBConv layer ofr TinyViT.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float,
            activation: Type[nn.Module],
            drop_path: float
    ) -> None:
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            expand_ratio (float): ratio of expanded channels.
            activation (nn.Module): activation function.
            drop_path (float): drop path ratio.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = int(in_channels * expand_ratio)
        self.out_channels = out_channels
        self.conv1 = Conv2dBN(in_channels, self.hidden_chans, kernel_size=1)
        self.act1 = activation()
        self.conv2 = Conv2dBN(
            self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, pad=1, groups=self.hidden_channels
        )
        self.act2 = activation()
        self.conv3 = Conv2dBN(
            self.hidden_channels, out_channels, kernel_size=1, bn_weight_init=0.)
        self.act3 = activation()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): output tensor.
        """
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        x = self.act3(x)
        return x
