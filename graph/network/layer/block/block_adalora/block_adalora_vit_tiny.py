from typing import Tuple, Type
import torch.nn as nn
import torch.nn.functional as fn
from timm.layers import DropPath
from ...conv import Conv2dBN
from ...lora import LoRALeViTAttention2, LoRAMLP4

__all__ = [
    "TinyViTAdaLoRABlock"
]


class TinyViTAdaLoRABlock(nn.Module):
    """
    TinyViT Block with AdaLoRA module.
    """

    def __init__(
        self,
        dim: int,
        resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.,
        drop: float = 0.,
        drop_path: float = 0.,
        local_conv_size: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        mid_dim: int = 768,
        **kwargs
    ) -> None:
        """
        Args:
            dim (int): number of input channels.
            resolution (Tuple): input resolution (size).
            num_heads (int): number of attention heads.
            window_size (int): window size.
            mlp_ratio (float): ratio of mlp hidden dim to embedding dim.
            drop (float): dropout rate.
            drop_path (float): stochastic depth rate.
            local_conv_size (int): the kernel size of the convolution between attention and MLP modules.
            activation (nn.Module): activation function.
            mid_dim (int): hidden dim for MLP layers.
        """
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.num_heads = num_heads
        assert (window_size > 0), "window_size must be greater than 0"
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        if mid_dim is not None:
            lora_rank = mid_dim
        else:
            lora_rank = 4
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        assert (dim % num_heads == 0), "dim must be divisible by num_heads"
        head_dim = dim // num_heads
        window_resolution = (window_size, window_size)
        self.attn = LoRALeViTAttention2(dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution, lora_rank=lora_rank)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = LoRAMLP4(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=mlp_activation, drop=drop, lora_rank=lora_rank)
        pad = local_conv_size // 2
        self.local_conv = Conv2dBN(dim, dim, kernel_size=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H %
                     self.window_size) % self.window_size
            pad_r = (self.window_size - W %
                     self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = fn.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_size * self.window_size, C)
            x = self.attn(x)
            # window reverse
            x = x.view(B, nH, nW, self.window_size, self.window_size,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.view(B, L, C)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        x = x + self.drop_path(self.mlp(x))
        return x
