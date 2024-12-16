from typing import Tuple, Type, Set, Optional
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from timm.layers import trunc_normal_
from ..layer import (
    PatchEmbed2,
    LayerNorm2d,
    PatchMerging,
    MBConvBlock,
    TinyViTBlock,
    TinyViTAdapterBlock,
    TinyViTLoRABlock,
    TinyViTAdaLoRABlock
)

__all__ = [
    "TinyViT"
]


class BasicBlock(nn.Module):
    """
    A Basic Tiny ViT layer for one stage.
    """

    def __init__(
            self,
            dim: int,
            resolution: Tuple[int, int],
            depth: int,
            num_heads: int,
            window_size: int,
            mlp_ratio: float = 4.,
            drop: float = 0.,
            drop_path: float = 0.,
            downsample: Optional[nn.Module] = None,
            use_checkpoint: bool = False,
            local_conv_size: int = 3,
            activation: type[nn.Module] = nn.GELU,
            out_dim: Optional[int] = None,
            block_name: str = "default",
            mid_dim: int = 768,
    ) -> None:
        """
        Args:
            dim (int): input dimension.
            resolution (Tuple): input resolution.
            depth (int): number of the blocks.
            num_heads (int): number of attention heads.
            window_size (int): local window size.
            mlp_ratio (float): ratio of mlp hidden dim to embedding dim.
            drop (float): dropout rate.
            drop_path (float): stochastic depth rate.
            downsample (nn.Module): downsample block at the end of the block.
            use_checkpoint (bool): whether to use checkpoint.
            local_conv_size (int): the kernel size of depth-wise convolution between attention and MLP modules.
            activation (nn.Module): activation function.
            out_dim (int): output dimension.
            block_name (str): name of the residual block.
            mid_dim (int): middle dim of adapter or the rank of lora matrix.
        """
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        if block_name == "adapt":
            block_class = TinyViTAdapterBlock
        elif block_name == "lora":
            block_class = TinyViTLoRABlock
        elif block_name == "adalora":
            block_class = TinyViTAdaLoRABlock
        else:
            block_class = TinyViTBlock
        self.blocks = nn.ModuleList([
            block_class(
                dim=dim,
                resolution=resolution,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                local_conv_size=local_conv_size,
                activation=activation,
                mid_dim=mid_dim
            )
            for i in range(depth)
        ])
        # patch merging layer.
        if downsample is not None:
            self.downsample = downsample(
                resolution, dim=dim, out_dim=out_dim, activation=activation
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [B, C, H, W].
        """
        for block in self.blocks:
            if self.use_checkpoint:
                x = cp.checkpoint(block, x)
            else:
                x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class TinyViT(nn.Module):
    def __init__(
            self,
            img_size: int = 224,
            in_channels: int = 3,
            num_classes: int = 1000,
            embed_dims: Tuple[int, ...] = (96, 192, 384, 768),
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            mlp_ratio: float = 4.,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            act_layer: Type[nn.Module] = nn.GELU,
            window_sizes: Tuple[int, ...] = (7, 7, 14, 7),
            use_checkpoint: bool = False,
            mbconv_expand_ratio: float = 4.,
            local_conv_size: int = 3,
            layer_lr_decay: float = 1.,
            block_name: str = "default",
            mid_dim: int = 768,
    ) -> None:
        """
        Args:
            img_size (int): input image size.
            in_channels (int): number of input image channels.
            num_classes (int): number of classes.
            embed_dims (Tuple): patch embedding dimensions.
            depths (Tuple): depth of blocks.
            num_heads (Tuple): number of attention heads.:
            mlp_ratio (float): ratio of mlp hidden dim to embedding dim.
            drop_rate (float): dropout rate.
            drop_path_rate (float): stochastic depth rate.
            act_layer (nn.Module): activation function.
            window_sizes (Tuple):  window size for window attention blocks.
            use_checkpoint (bool): whether to use checkpoint.
            mbconv_expand_ratio (float): ratio of mbconv hidden dim to embedding dim.
            local_conv_size (int): the kernel size of depth-wise convolution between attention and MLP modules.
            layer_lr_decay (float): decay rate of depth-wise convolution between attention and MLP modules.
            block_name (str): name of the residual block.
            mid_dim (int): middle dim of adapter or the rank of lora matrix.
        """
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed2(
            in_channels=in_channels,
            embed_dim=embed_dims[0],
            resolution=img_size,
            activation=act_layer
        )
        self.patches_resolution = self.patch_embed.patches_resolution
        # stochastic depth.
        # stochastic depth decay rule.
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        # build layers.
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            kwargs = dict(
                dim=embed_dims[i],
                input_resolution=(
                    self.patches_resolution[0] // (2 ** (i - 1 if i == 3 else i)),
                    self.patches_resolution[1] // (2 ** (i - 1 if i == 3 else i))
                ),
                depth=depths[i],
                drop_path=dpr[sum(depths[: i]): sum(depths[: i + 1])],
                downsample=PatchMerging if (i < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                out_dim=embed_dims[min(i + 1, len(embed_dims) - 1)],
                activation=act_layer
            )
            if i == 0:
                layer = MBConvBlock(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = BasicBlock(
                    num_heads=num_heads[i],
                    window_size=window_sizes[i],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    block_name=block_name,
                    mid_dim=mid_dim,
                    **kwargs
                )
            self.layers.append(layer)
        # classifier head.
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dims[-1], 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

    def set_layer_lr_decay(self, layer_lr_decay: float) -> None:
        def _set_lr_scale(m_: nn.Module, scale_: float) -> None:
            for p_ in m_.parameters():
                p_.lr_scale = scale_

        def _check_lr_scale(m_: nn.Module) -> None:
            for p_ in m_.parameters():
                assert hasattr(p_, "lr_scale"), p.param_name

        decay_rate = layer_lr_decay
        # layers -> blocks (depth).
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]
        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))
        for k, p in self.named_parameters():
            p.param_name = k
        self.apply(_check_lr_scale)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """
        Initialize the weights.

        Args:
            m (nn.Module): network module.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self) -> Set[str]:
        return {"attention_biases"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): the input tensor of shape (B, C, H, W).
        Returns:
            (torch.Tensor): the output tensor.
        """
        x = self.patch_embed(x)
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        B, _, C = x.size()
        x = x.view(B, self.img_size // 16, self.img_size // 16, C)
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)
        # x = self.norm_head(x)
        # x = self.head(x)
        return x
