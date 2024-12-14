from typing import Optional
from argparse import Namespace
from torch.types import Device
import torch
from torch.nn.modules.loss import _Loss as Loss

__all__ = [
    "get_loss"
]

def get_loss(args: Namespace, device: Device) -> Optional[Loss]:
    """
    Args:
        args (Namespace): criterion configuration.
        device (Device): device index to select.
    Returns:
        (Loss): criterion function.
    """
    if args.loss == "bce_w_logit":
        from torch.nn import BCEWithLogitsLoss
        weight_pos = torch.ones(1).to(device) * 2
        return BCEWithLogitsLoss(pos_weight=weight_pos)
    else:
        print("The criterion is not supported now !!!")
        return None
