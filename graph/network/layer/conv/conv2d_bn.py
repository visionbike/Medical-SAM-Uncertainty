import torch
import torch.nn as nn

__all__ = [
    "Conv2dBN"
]


class Conv2dBN(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1.
    ) -> None:
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): kernel size.
            stride (int): stride.
            pad (int): padding.
            dilation (int): dilation.
            groups (int): groups.
            bn_weight_init (int): init bn weights.
        """
        super().__init__()
        self.add_module(
            "c",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                pad,
                dilation,
                groups,
                bias=False
            )
        )
        bn = nn.BatchNorm2d(out_channels)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)

    @torch.no_grad()
    def fuse(self) -> nn.Module:
        """
        Returns:
            (nn.Module): fused module.
        """
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5
        m = nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
