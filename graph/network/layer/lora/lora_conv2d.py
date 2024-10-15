import math
import torch
import torch.nn as nn
import torch.nn.functional as fn
from .lora_base import LoRA

__al__ = [
    "LoRaConv2d"
]


class LoRAConv2d(nn.Conv2d, LoRA):
    """
    LoRA implemented in a Conv2d layer.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ) -> None:
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): kernel size.
            r (int): number of receptive field.
            lora_alpha (int):
            lora_dropout (float): dropout rate.
            fan_in_fan_out (bool):
            merge_weights (bool): set True if the layer to replace stores weight like (fan_in, fan_out).
        """
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRA.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert type(kernel_size) is int
        # actual trainable parameters.
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels * kernel_size, r * kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A as the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True) -> None:
        """
        Args:
            mode (bool):
        """
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            # make sure that the weights are not merged.
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False

    def eval(self) -> None:
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            # merge the weights and mark it.
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): output tensor.
        """
        if self.r > 0 and not self.merged:
            return fn.conv2d(
                x,
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups
            )
        return nn.Conv2d.forward(self, x)
