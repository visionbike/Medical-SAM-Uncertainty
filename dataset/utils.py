from typing import Tuple
from numpy.typing import NDArray
import random
import torch
import numpy as np

__all__ = [
    "random_box",
    "random_click"
]


def random_box(masks: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    Get random bboxes
    Args:
        masks (Tensor): masks evaluated by multiple subjects.
    Returns:
        (Tuple): return top-left and bottom-right coordinates (x_min, x_max, y_min, y_max).
    """
    max_value = torch.max(masks[:, 0, :, :], dim=0)[0]
    max_value_position = torch.nonzero(max_value)
    x_coords = max_value_position[:, 0]
    y_coords = max_value_position[:, 1]
    x_min = int(torch.min(x_coords))
    x_max = int(torch.max(x_coords))
    y_min = int(torch.min(y_coords))
    y_max = int(torch.max(y_coords))
    x_min = random.choice(np.arange(x_min - 10, x_min + 11))
    x_max = random.choice(np.arange(x_max - 10, x_max + 11))
    y_min = random.choice(np.arange(y_min - 10, y_min + 11))
    y_max = random.choice(np.arange(y_max - 10, y_max + 11))
    return x_min, x_max, y_min, y_max



def random_click(mask: NDArray, point_labels: int = 1) -> Tuple[int, NDArray]:
    """
    Get random click points.

    Args:
        mask (NDArray): a binary or multi-class mask that represents the ground truth or current segmentation state.
        point_labels (int): labels corresponding to each click (point),
            indicating whether the click is on the foreground (1 - positive) or background (0 - negative).
    Returns:
        (int):  label for each clicking point.
        (NDArray): list of positions for the clicks.
    """
    # check if all masks are black
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        point_labels = max_label
    # max agreement position
    indices = np.argwhere(mask == max_label)
    return point_labels, indices[np.random.randint(len(indices))]


