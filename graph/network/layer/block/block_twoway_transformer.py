from typing import Type, Tuple
import torch
import torch.nn as nn
from ..attention import TwoWayAttention
from ..mlp import MLP1

__all__ = [
    "TwoWayAttentionBlock",
    "TwoWayTransformer"
]


class TwoWayAttentionBlock(nn.Module):
    """
    A transformer block with four layers:
        (1) self-attention of sparse inputs,
        (2) cross attention of sparse inputs to dense inputs,
        (3) mlp block on sparse inputs,
        (4) cross attention of dense inputs to sparse inputs.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        Args:
            embedding_dim (int): the channel dimension of the embeddings.
            num_heads: (int) the number of attention heads.
            mlp_dim (int): the hidden dimension of the mlp block.
            activation (nn.Module): the activation function.
            attention_downsample_rate (int): the downsample rate of the attention heads.:
            skip_first_layer_pe (bool): whether to skip the PE on the first layer of the attention block.
        """
        super().__init__()
        self.self_attn = TwoWayAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = TwoWayAttention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP1(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = TwoWayAttention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate
        )
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        query_pe: torch.Tensor,
        key_pe: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries (torch.Tensor): the query tensor.
            keys (torch.Tensor): the key tensor.
            query_pe (torch.Tensor): the query position embedding.
            key_pe (torch.Tensor): the key position embedding.
        Returns:
            (torch.Tensor): the processed point_embedding.
            (torch.Tensor): the processed image_embedding.
        """
        # self attention block.
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        # cross attention block, tokens attending to image embedding.
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        # MLP block.
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        # cross attention block, image embedding attending to tokens.
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


class TwoWayTransformer(nn.Module):
    """
    A transformer decoder that attends to an input image using queries whose positional embedding is supplied.
    """
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        Args:
            depth (int): number of layers in the transformer.
            embedding_dim (int): the channel dimension for the input embeddings.
            num_heads (int): the number of heads for multihead attention. Must divide embedding_dim.
            mlp_dim (int): the channel dimension internal to the MLP block.
            activation (nn.Module): the activation to use in the MLP block.
            attention_downsample_rate (int): the downsample rate of the attention heads.
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )
        self.final_attn_token_to_image = TwoWayAttention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        point_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_embedding (torch.Tensor): image to attend to.
                Should be shape B x embedding_dim x h x w for any h and w.
            image_pe (torch.Tensor): the positional encoding to add to the image.
                Must have the same shape as image_embedding.
            point_embedding (torch.Tensor): the embedding to add to the query points.
                Must have shape B x N_points x embedding_dim for any N_points.
        Returns:
          torch.Tensor: the processed point_embedding.
          torch.Tensor: the processed image_embedding.
        """
        # B x C x H x W -> B x HW x C == B x N_image_tokens x C
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        # prepare queries.
        queries = point_embedding
        keys = image_embedding
        # apply transformer blocks and final layernorm.
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
        # apply the final attention layer from the points to the image.
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys
