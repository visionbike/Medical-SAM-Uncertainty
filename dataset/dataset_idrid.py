from typing import Optional, Callable, Dict
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from .utils import *

__all__ = [
    "IDRiD"
]


class IDRiD(Dataset):
    """
    IDRiD Dataset for Indian Diabetic Retinopathy Image Dataset.
    Link: https://cecas.clemson.edu/~ahoover/stare/probing/index.html
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
            mode (str): mode for loading training and testing dataset.
            prompt (str): prompt types, including
                "none": no applying prompt.
                "click": applying click prompt.
                "box": applying bbox prompt.
            image_size (int): input image size.
            transform (Callable): transform functions for image.
            transform_mask (Callable): transform function for mask.
        """
        super().__init__()
        self.path = path
        self.mode = mode
        self.names = [f.name[:-4] for f in sorted(Path(path, "image", mode).iterdir()) if f.is_file()]
        self.prompt = prompt
        self.image_size = image_size
        self.transform = transform
        self.transform_mask = transform_mask

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> Dict:
        """
        Args:
            idx (int): input index.
        Returns:
            (Dict): image, ground truth (mask), prompt data and related metadata.
        """
        # read image and labels (masks) which are evaluated by different subjects
        image = cv2.imread(f"{self.path}/image/{self.mode}/{self.names[idx]}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = []
        for subdir in Path(self.path, "mask", self.mode).iterdir():
            if (subdir / f"{self.names[idx]}_{subdir.name}.tif").exists():
                mask = cv2.imread(f"{subdir}/{self.names[idx]}_{subdir.name}.tif", cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (self.image_size, self.image_size))
            else:
                mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            mask[mask != 0] = 255
            label.append(mask)
        label = np.stack(label, axis=-1)
        point_label, point_coord = 1, np.array([0, 0], np.int32)
        if self.prompt == "click":
            point_label, point_coord = random_click(label[:, :, 0] / 255., point_labels=1)
        # print(point_label, point_coord)
        if self.transform:
            # save the current random number generate for reproducibility
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
        if self.transform_mask:
            # save the current random number generate for reproducibility
            state = torch.get_rng_state()
            label = self.transform_mask(label)
            torch.set_rng_state(state)
        return {
            "image": image,
            "label": label,
            "point_label": point_label,
            "point_coord": point_coord,
            "filename": self.names[idx]
        }
