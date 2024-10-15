from typing import Optional, Tuple, Type
import torch
import torch.nn as nn
from ..layer import (
    ViTBlock,
    ViTAdapterBlock,
    ViTLoRABlock,
    ViTAdaLoRABlock,
    LayerNorm2d,
    PatchEmbed1
)

__all__ = [
    "ViT"
]


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        out_channels: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        block_name: str = "default",
        mid_dim: int = 768,
    ) -> None:
        """
        Args:
            img_size (int): input image size.
            patch_size (int): patch size.
            in_channels (int): number of input image channels.
            embed_dim (int): patch embedding dimension.
            depth (int): number of ViT blocks.
            num_heads (int): number of attention heads in each ViT block.
            mlp_ratio (float): ratio of mlp hidden dim to embedding dim.
            out_channels (int): number of output tensor channels.
            qkv_bias (bool): if True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): normalization layer.
            act_layer (nn.Module): activation layer.
            use_abs_pos (bool): if True, use absolute positional embeddings.
            use_rel_pos (bool): if True, add relative positional embeddings to the attention map.
            window_size (int): window size for window attention blocks.
            global_attn_indexes (Tuple): indexes for blocks using global attention.
            block_name (str): name of the residual blocks.
            mid_dim (int): middle dim of adapter or the rank of lora matrix.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed1(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )
        self.blocks = nn.ModuleList()
        if block_name == "adapt":
            block_class = ViTAdapterBlock
        elif block_name == "lora":
            block_class = ViTLoRABlock
        elif block_name == "adalora":
            block_class = ViTAdaLoRABlock
        else:
            block_class = ViTBlock
        for i in range(depth):
            block = block_class(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                mid_dim=mid_dim
            )
            self.blocks.append(block)
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): output tensor.
        """
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x
