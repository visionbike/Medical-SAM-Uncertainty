from typing import Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "PositionEmbeddingRandom"
]


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        """
        Args:
            num_pos_feats (int): The size of the position embedding:
            scale (float): If provided, it is used to scale the position embedding.
        """
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Positionally encode points that are normalized to [0,1].

        Args:
            coords (torch.Tensor): input coordinates in shape of (B, N, C).
        Returns:
            (torch.Tensor): encoded coordinates.
        """
        """"""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape.
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape.
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Positionally encode points that are not normalized to [0,1].

        Args:
            coords_input (torch.Tensor): input coordinates in shape of (B, N, C).
            image_size (Tuple): input image size.
        Returns:
            (torch.Tensor): encoded coordinates.
        """
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate positional encoding for a grid of the specified size.

        Args:
            size (Tuple): size of the grid.
        Returns:
            (torch.Tensor): encoded coordinates.
        """
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W
