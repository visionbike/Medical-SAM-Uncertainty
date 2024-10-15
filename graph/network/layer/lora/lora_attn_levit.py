from typing import Tuple
from itertools import product
import torch
import torch.nn as nn
from .lora_linear import LoRAMergedLinear, LoRASVDLinear

__all__ = [
    "LoRALeViTAttention1",
    "LoRALeViTAttention2"
]


class LoRALeViTAttention1(nn.Module):
    """
    The attention module with LoRA for LeViT and TinyViT.
    """
    def __init__(
        self,
        in_dim: int,
        key_dim: int,
        num_heads: int = 8,
        attn_ratio: float = 4.,
        resolution: Tuple[int, int] = (14, 14),
        lora_rank: int = 4
    ) -> None:
        """
        Args:
            in_dim (int): dimension of the input tensor.
            key_dim (int): dimension of the key tensor.
            num_heads (int): number of attention heads.
            attn_ratio (float): attention ratio.
            resolution (Tuple): the image resolution (size).
            lora_rank (int): rank of the LoRA layer.
        """
        super().__init__()
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim**(-0.5)
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.norm = nn.LayerNorm(in_dim)
        self.qkv = LoRAMergedLinear(in_dim, h, r=lora_rank, enable_lora=(True, False, True))
        self.proj = nn.Linear(self.dh, in_dim)
        points = list(
            product(range(resolution[0]), range(resolution[1]))
        )
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))
        )
        self.register_buffer(
            "attention_bias_idxs",
            torch.LongTensor(idxs).view(N, N),
            persistent=False
        )
        self.ab = None

    @torch.no_grad()
    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]
            # self.register_buffer('ab',
            #                    self.attention_biases[:, self.attention_bias_idxs],
            #                    persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): the input tensor.
        Returns:
            (torch.Tensor): the output tensor.
        """
        B, N, _ = x.shape
        # normalization.
        x = self.norm(x)
        qkv = self.qkv(x)
        # q, k, v tensors have shape of B x N x num_heads x d
        q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class LoRALeViTAttention2(nn.Module):
    """
    The attention module with LoRA for LeViT and TinyViT.
    """
    def __init__(
        self,
        in_dim: int,
        key_dim: int,
        num_heads: int = 8,
        attn_ratio: float = 4.,
        resolution: Tuple[int, int] = (14, 14),
        lora_rank: int = 4
    ) -> None:
        """
        Args:
            in_dim (int): dimension of the input tensor.
            key_dim (int): dimension of the key tensor.
            num_heads (int): number of attention heads.
            attn_ratio (float): attention ratio.
            resolution (Tuple): the image resolution (size).
            lora_rank (int): rank of the LoRA layer.
        """
        super().__init__()
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim**(-0.5)
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.norm = nn.LayerNorm(in_dim)
        self.qkv = LoRASVDLinear(in_dim, h, r=lora_rank, enable_lora=(True, False, True))
        self.proj = LoRASVDLinear(self.dh, in_dim)
        points = list(
            product(range(resolution[0]), range(resolution[1]))
        )
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))
        )
        self.register_buffer(
            "attention_bias_idxs",
            torch.LongTensor(idxs).view(N, N),
            persistent=False
        )
        self.ab = None

    @torch.no_grad()
    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]
            # self.register_buffer('ab',
            #                    self.attention_biases[:, self.attention_bias_idxs],
            #                    persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): the input tensor.
        Returns:
            (torch.Tensor): the output tensor.
        """
        B, N, _ = x.shape
        # normalization.
        x = self.norm(x)
        qkv = self.qkv(x)
        # q, k, v tensors have shape of B x N x num_heads x d
        q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x
