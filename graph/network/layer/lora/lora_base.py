import torch.nn as nn

__all__ = [
    "LoRA"
]


class LoRA:
    """
    LoRA layer implementation
    """
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ) -> None:
        """
        Args:
            r (int):
            lora_alpha (int):
            lora_dropout (float):
            merge_weights (bool):
        """
        self.r = r
        self.lora_alpha = lora_alpha
        # optional dropout.
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # mark the weight as unmerged.
        self.merged = False
        self.merge_weights = merge_weights
