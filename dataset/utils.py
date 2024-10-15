from typing import Tuple
from numpy.typing import NDArray
import random
import torch
import numpy as np

__all__ = [
    "random_box",
    "random_click",
    "generate_click_prompt"
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


def generate_click_prompt(img, msk, pt_label = 1):
    # return: prompt, prompt mask
    pt_list = []
    msk_list = []
    b, c, h, w, d = msk.size()
    msk = msk[:,0,:,:,:]
    for i in range(d):
        pt_list_s = []
        msk_list_s = []
        for j in range(b):
            msk_s = msk[j,:,:,i]
            indices = torch.nonzero(msk_s)
            if indices.size(0) == 0:
                # generate a random array between [0-h, 0-h]:
                random_index = torch.randint(0, h, (2,)).to(device = msk.device)
                new_s = msk_s
            else:
                random_index = random.choice(indices)
                label = msk_s[random_index[0], random_index[1]]
                new_s = torch.zeros_like(msk_s)
                # convert bool tensor to int
                new_s = (msk_s == label).to(dtype = torch.float)
                # new_s[msk_s == label] = 1
            pt_list_s.append(random_index)
            msk_list_s.append(new_s)
        pts = torch.stack(pt_list_s, dim=0)
        msks = torch.stack(msk_list_s, dim=0)
        pt_list.append(pts)
        msk_list.append(msks)
    pt = torch.stack(pt_list, dim=-1)
    msk = torch.stack(msk_list, dim=-1)

    msk = msk.unsqueeze(1)

    return img, pt, msk #[b, 2, d], [b, c, h, w, d]
