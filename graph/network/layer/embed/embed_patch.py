from typing import Tuple, Type
import torch
import torch.nn as nn
from timm.layers import to_2tuple
from ..conv import Conv2dBN

__all__ = [
    "PatchEmbed1",
    "PatchEmbed2",
]


class PatchEmbed1(nn.Module):
    """
    Image to Patch Embedding for ViT encoder.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_channels (int): number of input image channels.
            embed_dim (int): patch embedding dimension.
        """

        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            x (torch.Tensor): patch embedding.
        """
        x = self.proj(x)
        # B x C x H x W -> B x H x W x C
        x = x.permute(0, 2, 3, 1)
        return x


class PatchEmbed2(nn.Module):
    """
    Image to Patch Embedding for Tiny ViT encoder.
    """
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        resolution: int,
        activation: Type[nn.Module] = nn.GELU
    ) -> None:
        """
        Args:
            in_channels (int): number of input image channels.
            embed_dim (int): patch embedding dimension.
            resolution (int): patch image size.
            activation (nn.Module): activation function.
        """
        super().__init__()
        img_size: Tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.seq = nn.Sequential(
            Conv2dBN(in_channels, embed_dim // 2, 3, 2, 1),
            activation(),
            Conv2dBN(embed_dim // 2, embed_dim, 3, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): patch embedding.
        """
        return self.seq(x)
