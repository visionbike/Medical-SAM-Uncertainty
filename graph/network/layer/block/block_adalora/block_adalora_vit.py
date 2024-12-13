from typing import Optional, Type
import torch
import torch.nn as nn
from ...lora import LoRAMultiheadAttention2, LoRAMLP2
from ...utils import window_partition, window_unpartition

__all__ = [
    "ViTAdaLoRABlock"
]


class ViTAdaLoRABlock(nn.Module):
    """Transformer blocks with AdaLoRA for ViT with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        window_size: int = 0,
        mid_dim: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            window_size (int): Window size for window attention blocks.
                If it equals 0, then use global attention.
            mid_dim (int): middle dim of block_adapter or the rank of lora matrix.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        if mid_dim is not None:
            lora_rank = mid_dim
        else:
            lora_rank = 4
        self.attn = LoRAMultiheadAttention2(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            lora_rank=lora_rank,
            input_size=(64, 64) if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = norm_layer(dim)
        self.mlp = LoRAMLP2(
            embedding_dim=dim,
            mlp_dim=int(dim * mlp_ratio),
            act=act_layer,
            lora_rank=lora_rank
        )
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
        # window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            x = self.attn(x)
            # reverse window partition
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            x = self.attn(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
