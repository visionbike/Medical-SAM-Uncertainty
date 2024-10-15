import torch
import torch.nn as nn
import torch.nn.functional as fn
from .lora_base import LoRA

__all__ = [
    "LoRAEmbedding"
]


class LoRAEmbedding(nn.Embedding, LoRA):
    """
    LoRA implemented in an embedding layer.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ) -> None:
        """
        Args:
            num_embeddings (int): number of embeddings.
            embedding_dim (int): embedding dimension.
            r (int):
            lora_alpha (int):
            merge_weights (bool):
        """
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRA.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
        # actual trainable parameters.
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.Embedding.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A as the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True) -> None:
        """
        Args:
            mode (bool):
        """
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            # make sure that the weights are not merged.
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False

    def eval(self) -> None:
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # merge the weights and mark it.
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor):
        Returns:
            (torch.Tensor):
        """
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = fn.embedding(
                    x,
                    self.lora_A.T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse
                )
                result += (after_A @ self.lora_B.T) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
