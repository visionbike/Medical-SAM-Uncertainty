import random
import numpy as np
import torch

__all__ = [
    "setup_seed"
]

def setup_seed(use_gpu: bool = True, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False