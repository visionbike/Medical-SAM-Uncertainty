from typing import Optional, List, Tuple
import torch

__all__ = [
    "get_device"
]


def get_device(
    use_gpu: bool = True,
    gpu_device: Optional[int] = None,
    distributed: Optional[str] = None
) -> Tuple[torch.device, Optional[List[int]]]:
    """
    Get device and some set up in GPU usage.

    Args:
        use_gpu (bool): whether to use GPU device.
        gpu_device (int): GPU id to locate the model.
        distributed (str): list of GPU device for distributed training.
    Returns:
        (device): PyTorch device.
    """
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        if distributed is not None:
            device_ids = [int(i) for i in distributed.split(",")]
            return torch.device("cuda", device_ids[0]), device_ids
        return torch.device("cuda", gpu_device), None
    return torch.device("cpu"), None
