from typing import Optional, Tuple, Type
import torch
import torch.nn as nn
from ...adapter import Adapter
from ...attention import MultiheadAttention
from ...mlp import MLP1
from ...utils import window_partition, window_unpartition

__all__ = [
    "ViTAdapterBlock"
]


class ViTAdapterBlock(nn.Module):
    """
    Transformer blocks with block_adapter for ViT with support of window attention and residual propagation blocks.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        scale: float = 0.5,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        mid_dim: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Args:
            dim (int): number of input channels.
            num_heads (int): number of attention heads in each ViT block.
            mlp_ratio (float): ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): if True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): normalization layer.
            act_layer (nn.Module): activation layer.
            use_rel_pos (bool): if True, add relative positional embeddings to the attention map.
            window_size (int): window size for window attention blocks.
                If it equals 0, then use global attention.
            input_size (Tuple): input resolution for calculating the relative
                positional parameter size.
            mid_dim (int): middle dim of block_adapter or the rank of lora matrix.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiheadAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        if mid_dim is not None:
            adapter_dim = mid_dim
        else:
            adapter_dim = dim
        self.adapter_mlp = Adapter(adapter_dim, skip_connect=False)  # MLP-block_adapter, no skip connection
        self.adapter_spatial = Adapter(adapter_dim)  # with skip connection
        self.scale = scale
        # self.Depth_Adapter = Adapter(adapter_dim, skip_connect=False)  # no skip connection
        self.norm2 = norm_layer(dim)
        self.mlp = MLP1(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): output tensor.
        """
        shortcut = x
        # window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        # ## 3d branch
        # if self.args.thd:
        #     hh, ww = x.shape[1], x.shape[2]
        #     if self.args.chunk:
        #         depth = self.args.chunk
        #     else:
        #         depth = x.shape[0]
        #     xd = rearrange(x, '(b d) h w c -> (b h w) d c ', d=depth)
        #     # xd = rearrange(xd, '(b d) n c -> (b n) d c', d=self.in_chans)
        #     xd = self.norm1(xd)
        #     dh, _ = closest_numbers(depth)
        #     xd = rearrange(xd, 'bhw (dh dw) c -> bhw dh dw c', dh= dh)
        #     xd = self.Depth_Adapter(self.attn(xd))
        #     xd = rearrange(xd, '(b n) dh dw c ->(b dh dw) n c', n= hh * ww )
        x = self.norm1(x)
        x = self.attn(x)
        x = self.adapter_spatial(x)
        # if self.args.thd:
        #     xd = rearrange(xd, 'b (hh ww) c -> b  hh ww c', hh= hh )
        #     x = x + xd
        # reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        xn = self.norm2(x)
        x = x + self.mlp(xn) + self.scale * self.adapter_mlp(xn)
        return x
