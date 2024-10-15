import math
import torch
import torch.nn as nn

__all__ = [
    "TwoWayAttention"
]


class TwoWayAttention(nn.Module):
    """
    An attention layer in Two-way Transformer module that allows
    for downscaling the size of the embedding after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        """
        Args:
            embedding_dim (int): the dimension of the embeddings.
            num_heads (int): the number of attention heads.
            downsample_rate (int): the downsampling rate.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (self.internal_dim % num_heads == 0), "num_heads must divide embedding_dim."
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    @staticmethod
    def _separate_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """
        Separate heads for multi-head attention.

        Args:
            x (torch.Tensor): the input tensor.
            num_heads (int): the number of attention heads.
        Returns:
            (torch.Tensor): the separated heads.
        """
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    @staticmethod
    def _recombine_heads(x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads for multi-head attention.

        Args:
            x (torch.Tensor): the input tensor.
        Returns:
            (torch.Tensor): the recombined heads.
        """
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): the query tensor.
            k (torch.Tensor): the key tensor.
            v (torch.Tensor): the value tensor.
        Returns:
            (torch.Tensor): the output tensor.
        """
        # input projections.
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        # separate into heads.
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        # attention.
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        # get output.
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        #
        return out
