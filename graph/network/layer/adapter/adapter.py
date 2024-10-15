from typing import Type
import torch
import torch.nn as nn

__all__ = [
    "Adapter"
]


class Adapter(nn.Module):
    def __init__(
        self,
        num_features: int,
        mlp_ratio: float = 0.25,
        act_layer: Type[nn.Module] = nn.GELU,
        skip_connect: bool = True
    ) -> None:
        """
        Args:
            num_features (int): number of features in the input.
            mlp_ratio (float): ratio of mlp hidden layer.
            act_layer (nn.Module): activation layer.
            skip_connect (bool): whether to add skip connection.
        """
        super().__init__()
        self.skip_connect = skip_connect
        num_hidden_features = int(num_features * mlp_ratio)
        self.act = act_layer()
        self.ln1 = nn.Linear(num_features, num_hidden_features)
        self.ln2 = nn.Linear(num_hidden_features, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): output tensor.
        """
        # x in shape of (BT, H*W + 1, D)
        xs = self.ln1(x)
        xs = self.act(xs)
        xs = self.ln2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
