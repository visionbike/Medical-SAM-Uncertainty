from typing import Type, Optional
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from ..conv import MBConv

__all__ = [
    "MBConvBlock"
]


class MBConvBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        resolution: int,
        depth: int,
        activation: Type[nn.Module],
        drop_path: float = 0.,
        downsample: Optional[Type[nn.Module]] = None,
        use_checkpoint: bool = False,
        out_dim: Optional[float] = None,
        conv_expand_ratio: float = 4.,
    ) -> None:
        """
        Args:
            dim (int): dimension of input tensor.
            resolution (Tuple[int, int]): image resolution (size).
            depth (int): depth of block.
            activation (nn.Module): activation function.
            drop_path (bool): drop path ratio.
            downsample (nn.Module): downsample module.
            use_checkpoint (bool): whether to use checkpoint.
            out_dim (float): output dimension.
            conv_expand_ratio (float): expansion ratio.
        """
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            MBConv(
                dim, dim, conv_expand_ratio, activation, drop_path[i] if isinstance(drop_path, list) else drop_path,
            ) for i in range(depth)
        ])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): output tensor.
        """
        for block in self.blocks:
            if self.use_checkpoint:
                x = cp.checkpoint(block, x)
            else:
                x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
