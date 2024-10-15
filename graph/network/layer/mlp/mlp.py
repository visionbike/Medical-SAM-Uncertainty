from typing import Type, Optional
import torch
import torch.nn as nn
import torch.nn.functional as fn

__all__ = [
    "MLP1",
    "MLP2",
    "MLP3",
]


class MLP1(nn.Module):
    """
    MLP block for ViT encoder
    """
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Args:
            embedding_dim (int): embedding dimension.
            mlp_dim (int): MLP output dimension.
            act (nn.Module): activation function.
        """
        super().__init__()
        self.ln1 = nn.Linear(embedding_dim, mlp_dim)
        self.ln2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): output tensor.
        """
        return self.ln2(self.act(self.ln1(x)))


class MLP2(nn.Module):
    """
    MLP block for Two-way Transformer, lightly adapted from
    https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        """
        Args:
            input_dim (int): the input dimension.
            hidden_dim (int): the hidden dimension.
            output_dim (int): the output dimension.
            num_layers (int): the number of layers.
            sigmoid_output (bool): whether to use sigmoid.
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): output tensor.
        """
        for i, layer in enumerate(self.layers):
            x = fn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = fn.sigmoid(x)
        return x


class MLP3(nn.Module):
    """
    MLP block for TinyViT.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.
    ) -> None:
        """
        Args:
            in_features:
            hidden_features:
            out_features:
            act_layer:
            drop:
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.ln1 = nn.Linear(in_features, hidden_features)
        self.ln2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): output tensor.
        """
        x = self.norm(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.ln2(x)
        x = self.drop(x)
        return x
