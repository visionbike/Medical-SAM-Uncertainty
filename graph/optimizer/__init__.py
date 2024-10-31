from typing import Optional, Tuple
from argparse import Namespace
from torch.optim.optimizer import Optimizer, ParamsT
from torch.optim.lr_scheduler import LRScheduler


__all__ = [
    "get_optimizer"
]


def get_optimizer(args: Namespace, parameters: ParamsT) -> Tuple[Optimizer, Optional[LRScheduler]]:
    """
    Get training optimizer.

    Args:
        args (Namespace): optimizer configuration.
        parameters (ParamsT): network's parameters.
    Returns:
        (Optimizer): optimizer.
    """
    optimizer = None
    if args.optimizer == "adam":
        from torch.optim import Adam
        optimizer = Adam(parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    else:
        print("The optimizer is not supported now!!!")

    scheduler = None
    if args.lr_scheduler == "step":
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        print("The LR scheduler is not supported now !!!")
    return optimizer, scheduler
