from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as fn
from .lora_base import LoRA

__all__ = [
    "LoRALinear",
    "LoRAMergedLinear",
    "LoRASVDLinear"
]


class LoRALinear(nn.Linear, LoRA):
    """
    LoRA implemented in a dense layer.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ) -> None:
        """

        Args:
            in_features (int): number of input features.
            out_features (int): number of output features.
            r (int): number of receptive field.
            lora_alpha (int):
            lora_dropout (float): dropout rate.
            fan_in_fan_out (bool):
            merge_weights (bool): set True if the layer to replace stores weight like (fan_in, fan_out).
        """
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRA.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        # actual trainable parameters.
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # freezing the pre-trained weight matrix.
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A as the same way as the default for nn.Linear and B to zero.
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True) -> None:
        """
        Args:
            mode (bool):
        """
        def T(w_: torch.Tensor) -> torch.Tensor:
            """
            Args:
                w_ (torch.Tensor): input weights.
            Returns:
                (torch.Tensor): output transposed weights.
            """
            return w_.T if self.fan_in_fan_out else w_

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # make sure that the weights are not merged.
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self) -> None:
        def T(w_: torch.Tensor) -> torch.Tensor:
            """
            Args:
                w_ (torch.Tensor): input weights.
            Returns:
                (torch.Tensor): output transposed weights.
            """
            return w_.T if self.fan_in_fan_out else w_

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def T(w_: torch.Tensor) -> torch.Tensor:
            """
            Args:
                w_ (torch.Tensor): input weights.
            Returns:
                (torch.Tensor): output transposed weights.
            """
            return w_.T if self.fan_in_fan_out else w_

        if self.r > 0 and not self.merged:
            result = fn.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return fn.linear(x, T(self.weight), bias=self.bias)


class LoRAMergedLinear(nn.Linear, LoRA):
    """

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        enable_lora: Tuple[bool, ...] = (False,),
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ) -> None:
        """
        Args:
            in_features (int): number of input features.
            out_features (int): number of output features.
            r (int): number of receptive field.
            lora_alpha (int):
            lora_dropout (float): dropout rate.
            enable_lora (Tuple):
            fan_in_fan_out (bool):
            merge_weights (bool): set True if the layer to replace stores weight like (fan_in, fan_out).
        """
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRA.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert {out_features % len(enable_lora) == 0}, "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # actual trainable parameters.
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A as the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): zero padded tensor.
        """
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True) -> None:
        """
        Args:
            mode (bool):
        """
        def T(w_: torch.Tensor) -> torch.Tensor:
            """
            Args:
                w_ (torch.Tensor): input weights.
            Returns:
                (torch.Tensor): output transposed weights.
            """
            return w_.T if self.fan_in_fan_out else w_

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # make sure that the weights are not merged.
            if self.r > 0 and any(self.enable_lora):
                delta_w = fn.conv1d(
                    self.lora_A.data.unsqueeze(0),
                    self.lora_B.data.unsqueeze(-1),
                    groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False

    def eval(self) -> None:
        def T(w_: torch.Tensor) -> torch.Tensor:
            """
            Args:
                w_ (torch.Tensor): input weights.
            Returns:
                (torch.Tensor): output transposed weights.
            """
            return w_.T if self.fan_in_fan_out else w_

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # merge the weights and mark it.
            if self.r > 0 and any(self.enable_lora):
                delta_w = fn.conv1d(
                    self.lora_A.data.unsqueeze(0),
                    self.lora_B.data.unsqueeze(-1),
                    groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): output tensor.
        """
        def T(w_: torch.Tensor) -> torch.Tensor:
            """
            Args:
                w_ (torch.Tensor): input weights.
            Returns:
                (torch.Tensor): output transposed weights.
            """
            return w_.T if self.fan_in_fan_out else w_

        if self.merged:
            return fn.linear(x, T(self.weight), bias=self.bias)
        else:
            result = fn.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = fn.linear(self.lora_dropout(x), self.lora_A)
                after_B = fn.conv1d(
                    after_A.transpose(-2, -1),  # B, 12, M
                    self.lora_B.unsqueeze(-1),  # 3072, 4, 1
                    groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result


class LoRASVDLinear(nn.Linear, LoRA):
    """
    SVD-based adaptation implemented in a dense layer
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            **kwargs
    ) -> None:
        """
        Args:
            in_features (int): number of input features.
            out_features (int): number of output features.
            r (int): number of receptive field.
            lora_alpha (int):
            lora_dropout (float): dropout rate.
            fan_in_fan_out (bool):
            merge_weights (bool): set True if the layer to replace stores weight like (fan_in, fan_out).
        """
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRA.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        # actual trainable parameters.
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )
            self.lora_E = nn.Parameter(
                self.weight.new_zeros(r, 1)
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )
            self.rank_num = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            self.rank_num.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha > 0 else float(self.r)
            # freezing the pre-trained weight matrix.
            self.weight.requires_grad = False
            self.rank_num.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A,B the same way as the default for nn.Linear
            # and E (singular values) for zero.
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def train(self, mode: bool = True) -> None:
        """
        Args:
            mode (bool):
        """
        def T(w_: torch.Tensor) -> torch.Tensor:
            """
            Args:
                w_ (torch.Tensor): input weights.
            Returns:
                (torch.Tensor): output transposed weights.
            """
            return w_.T if self.fan_in_fan_out else w_

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ (self.lora_A * self.lora_E)) * self.scaling / (self.ranknum + 1e-5)
            self.merged = False

    def eval(self) -> None:
        def T(w_: torch.Tensor) -> torch.Tensor:
            """
            Args:
                w_ (torch.Tensor): input weights.
            Returns:
                (torch.Tensor): output transposed weights.
            """
            return w_.T if self.fan_in_fan_out else w_

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # merge the weights and mark it.
            if self.r > 0:
                self.weight.data += T(self.lora_B @ (self.lora_A * self.lora_E)) * self.scaling / (self.ranknum + 1e-5)
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
        Returns:
            (torch.Tensor): output tensor.
        """
        def T(w_: torch.Tensor) -> torch.Tensor:
            """
            Args:
                w_ (torch.Tensor): input weights.
            Returns:
                (torch.Tensor): output transposed weights.
            """
            return w_.T if self.fan_in_fan_out else w_

        if self.r > 0 and not self.merged:
            result = fn.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T) * self.scaling / (self.ranknum + 1e-5)
            return result
        else:
            return fn.linear(x, T(self.weight), bias=self.bias)
