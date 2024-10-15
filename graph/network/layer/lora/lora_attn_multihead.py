from typing import Optional, Tuple
import torch
import torch.nn as nn
from ..utils import add_decomposed_rel_pos
from .lora_linear import LoRAMergedLinear, LoRASVDLinear

__all__ = [
    "LoRAMultiheadAttention1",
    "LoRAMultiheadAttention2",
]


class LoRAMultiheadAttention1(nn.Module):
    """
    Multi-head Attention block with LoRA with relative position embeddings.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        lora_rank: int = 4,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): number of input channels.
            num_heads (int): number of attention heads.
            qkv_bias (bool): if True, add a learnable bias to query, key, value.
            use_rel_pos (bool): if True, add relative positional embeddings to the attention map.
            lora_rank (int): rank of lora heads.
            input_size (Tuple): input resolution for calculating the relative positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** (-0.5)
        self.qkv = LoRAMergedLinear(dim, dim * 3, bias=qkv_bias, r=lora_rank, enable_lora=(True, False, True))
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (input_size is not None), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings.
            self.rel_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): output tensor.
        """
        B, H, W, n = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        x = x.reshape(B, H*W, n)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_h, self.rel_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class LoRAMultiheadAttention2(nn.Module):
    """
    Multi-head Attention block with LoRA with relative position embeddings.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        lora_rank: int = 4,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): number of input channels.
            num_heads (int): number of attention heads.
            qkv_bias (bool): if True, add a learnable bias to query, key, value.
            use_rel_pos (bool): if True, add relative positional embeddings to the attention map.
            lora_rank (int): rank of lora heads.
            input_size (Tuple): input resolution for calculating the relative positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** (-0.5)
        self.qkv = LoRASVDLinear(dim, dim * 3, bias=qkv_bias, r=lora_rank, enable_lora=(True, False, True))
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (input_size is not None), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings.
            self.rel_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): output tensor.
        """
        B, H, W, n = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        x = x.reshape(B, H*W, n)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_h, self.rel_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x
