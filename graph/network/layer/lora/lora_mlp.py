from typing import Type, Optional
import torch
import torch.nn as nn
from .lora_linear import LoRALinear, LoRASVDLinear

__all__ = [
    "LoRAMLP1",
    "LoRAMLP2",
    "LoRAMLP3",
    "LoRAMLP4"
]


class LoRAMLP1(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
        lora_rank: int = 4,
    ) -> None:
        """
        Args:
            embedding_dim (int): embedding dimension.
            mlp_dim (int): MLP output dimension.
            act (nn.Module): activation function.
            lora_rank (int): rank of Lora MLP.
        """
        super().__init__()
        self.ln1 = LoRALinear(embedding_dim, mlp_dim, r=lora_rank)
        self.ln2 = LoRALinear(mlp_dim, embedding_dim, r=lora_rank)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): output tensor.
        """
        return self.ln2(self.act(self.ln1(x)))


class LoRAMLP2(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
        lora_rank: int = 4,
    ) -> None:
        """
        Args:
            embedding_dim (int): embedding dimension.
            mlp_dim (int): MLP output dimension.
            act (nn.Module): activation function.
            lora_rank (int): rank of Lora MLP.
        """
        super().__init__()
        self.ln1 = LoRASVDLinear(embedding_dim, mlp_dim, r=lora_rank)
        self.ln2 = LoRASVDLinear(mlp_dim, embedding_dim, r=lora_rank)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): output tensor.
        """
        return self.ln2(self.act(self.ln1(x)))


class LoRAMLP3(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.,
        lora_rank: int = 4
    ) -> None:
        """
        Args:
            in_features:
            hidden_features:
            out_features:
            act_layer:
            drop:
            lora_rank:
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.ln1 = LoRALinear(in_features, hidden_features, r=lora_rank)
        self.ln2 = LoRALinear(hidden_features, out_features, r=lora_rank)
        # self.fc1 = nn.Linear(in_features, hidden_features)
        # self.fc2 = nn.Linear(hidden_features, out_features)
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


class LoRAMLP4(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.,
        lora_rank: int = 4
    ) -> None:
        """
        Args:
            in_features:
            hidden_features:
            out_features:
            act_layer:
            drop:
            lora_rank:
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.ln1 = LoRASVDLinear(in_features, hidden_features, r=lora_rank)
        self.ln2 = LoRASVDLinear(hidden_features, out_features, r=lora_rank)
        # self.fc1 = nn.Linear(in_features, hidden_features)
        # self.fc2 = nn.Linear(hidden_features, out_features)
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
