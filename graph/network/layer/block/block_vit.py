from typing import Optional, Tuple, Type
import torch
import torch.nn as nn
from ..attention import MultiheadAttention
from ..mlp import MLP1
from ..utils import window_partition, window_unpartition

__all__ = [
    "ViTBlock"
]


class ViTBlock(nn.Module):
    """
    Transformer standard block with support of window attention and residual propagation blocks.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
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
            window_size (int): window size for window attention blocks. If it equals 0, then use global attention.
            input_size (Tuple): input resolution for calculating the relative positional parameter size.
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
        x = self.norm1(x)
        # window partition.
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            x = self.attn(x)
            # reverse window partition.
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            x = self.attn(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
