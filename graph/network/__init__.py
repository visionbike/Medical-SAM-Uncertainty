from typing import Optional, List
from argparse import Namespace
import sys
import torch
import torch.nn as nn
from .layer import *
from .decoder import *
from .encoder import *
from .mobile_sam_v2 import *

__all__ = [
    "get_network"
]

def get_network(
    args: Namespace,
    use_gpu: bool,
    device: torch.device,
    distributed: Optional[List[int]] = None,
    pretrain: Optional[str] = None
) -> nn.Module:
    """
    Get network.

    Args:
        args (Namespace): network configuration.
        use_gpu (bool): whether to use GPU.
        device (device): device.
        distributed (str): list of GPU devices for distributed training.
        pretrain (str): pretrained model path.
    Returns:
        (Module): network.
    """
    if args.net == "sam":
        from .sam import sam_model_registry
        options = ["default", "vit_b", "vit_l", "vit_h"]
        if args.encoder not in options:
            raise ValueError(f"Invalid encoder option. Please choose from: {options}")
        net = sam_model_registry[args.encoder](args, checkpoint=pretrain)
    elif args.net == "mobile_sam_v2":
        from .mobile_sam_v2 import mobile_sam_v2_model_registry
        options = ["default", "vit_t"]
        if args.encoder not in options:
            raise ValueError(f"Invalid encoder option. Please choose from: {options}")
        net = mobile_sam_v2_model_registry[args.encoder](args, checkpoint=pretrain)
    else:
        print("The network is not supported yet!")
        sys.exit()

    if use_gpu:
        if distributed is not None:
            net = nn.DataParallel(net, device_ids=distributed)
            return net.to(device=device).module
        return net.to(device=device)
    return net.to(device=device)
