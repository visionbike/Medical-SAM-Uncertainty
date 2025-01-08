from typing import Optional, Callable, Dict
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from .utils import *

__all__ = [
    "REFUGE"
]


class REFUGE(Dataset):
    """
    REFUGE Dataset for Optic-disc Segmentation from Fundus Images (2D).
    Link: https://huggingface.co/datasets/realslimman/REFUGE-MultiRater/tree/main
    """
    def __init__(
        self,
        path: str,
        mode: str,
        prompt: str = "click",
        image_size: int = 1024,
        transform: Optional[Callable] = None,
        transform_mask: Optional[Callable] = None
    ) -> None:
        """
        Args:
            path (str): data path.
            mode (str): mode for loading training or testing dataset.
            prompt (str): prompt types, including
                "none": no applying prompt.
                "click": applying click prompt.
                "box": applying bbox prompt.
            image_size (int): input image size.
            transform (Callable): transform functions for image.
            transform_mask (Callable): transform function for mask.
        """
        super().__init__()
        # get image, each folder is the image sample which is evaluated by 7 different subjects
        self.folder_images = [f for f in sorted(Path(path, f"{mode}-400").iterdir()) if f.is_dir()]
        self.mode = mode
        self.prompt = prompt
        self.image_size = image_size
        self.mask_size = image_size
        self.transform = transform
        self.transform_mask = transform_mask

    def __len__(self) -> int:
        return len(self.folder_images)

    def __getitem__(self, idx: int) -> Dict:
        """
        Args:
            idx (int): input index.
        Returns:
            (Dict): image, ground truth (mask), prompt data and related metadata.
        """
        # read image and labels (masks) which are evaluated by different subjects
        image = cv2.imread(f"{self.folder_images[idx]}/{self.folder_images[idx].name}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_cups, label_discs = [], []
        for i in range(1, 8):
            cup = cv2.imread(f"{self.folder_images[idx]}/{self.folder_images[idx].name}_seg_cup_{i}.png", cv2.IMREAD_GRAYSCALE)
            cup = cv2.resize(cup, (self.image_size, self.image_size))
            disc = cv2.imread(f"{self.folder_images[idx]}/{self.folder_images[idx].name}_seg_disc_{i}.png", cv2.IMREAD_GRAYSCALE)
            disc = cv2.resize(disc, (self.image_size, self.image_size))
            label_cups.append(cup)
            label_discs.append(disc)
        label_cups = np.stack(label_cups, axis=-1)
        label_discs = np.stack(label_discs, axis=-1)
        # get click points
        point_label, point_coord_cup, point_coord_disc = 1, np.array([0, 0], np.int32),  np.array([0, 0], np.int32)
        if self.prompt == "click":
            point_label, point_coord_cup = random_click(np.mean(label_cups, axis=-1) / 255., point_labels=1)
            # point_label, point_coord_disc = random_click(np.mean(label_discs, axis=-1) / 255., point_labels=1)
        # transform the input
        if self.transform:
            # save the current random number generate for reproducibility
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
        if self.transform_mask:
            # save the current random number generate for reproducibility
            state = torch.get_rng_state()
            label_cups = torch.as_tensor((self.transform_mask(label_cups) > 0.5).float(), dtype=torch.float32)
            label_discs = torch.as_tensor((self.transform_mask(label_discs) > 0.5).float(), dtype=torch.float32)
            torch.set_rng_state(state)
        label = torch.concat([label_cups.mean(dim=0, keepdim=True), label_discs.mean(dim=0, keepdim=True)], dim=0)
        # get bbox positions
        box_cup, box_disc = [0, 0, 0, 0], [0, 0, 0, 0]
        if self.prompt == "box":
           box_cup = random_box(label_cups)
           # box_disc = random_box(label_discs)
        name = Path(self.folder_images[idx]).name
        return {
            "image": image,
            "label": label,
            "point_label": point_label,
            "point_coord": point_coord_cup,
            "box_coord": box_cup,
            "filename": name
        }
