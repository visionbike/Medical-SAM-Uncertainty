from typing import Optional, Callable, Dict
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from .utils import *

__all__ = [
    "STARE"
]


class STARE(Dataset):
    """
    STARE Dataset for Retinal Blood Vessel (2D).
    Link: https://cecas.clemson.edu/~ahoover/stare/probing/index.html
    """
    def __init__(
        self,
        path: str,
        prompt: str = "click",
        image_size: int = 1024,
        transform: Optional[Callable] = None,
        transform_mask: Optional[Callable] = None
    ) -> None:
        """
        Args:
            path (str): data path.
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
        self.names = [f.name[:-4] for f in sorted(Path(path, f"images").iterdir()) if f.is_file()]
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
        image = cv2.imread(f"{self.path}/images/{self.names[idx]}.ppm")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(f"{self.path}/labels/{self.names[idx]}.ah.ppm", cv2.IMREAD_GRAYSCALE)
        # resize the label's resolution as same as image's
        label = cv2.resize(label, (self.image_size, self.image_size))
        # get click points
        point_label, point_coord = 1, np.array([0, 0], np.int32)
        if self.prompt == "click":
            point_label, point_coord = random_click(label / 255., point_labels=1)
        # transform the input
        if self.transform:
            # save the current random number generate for reproducibility
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
        if self.transform_mask:
            # save the current random number generate for reproducibility
            state = torch.get_rng_state()
            label = self.transform_mask(label).int()
            torch.set_rng_state(state)
        return {
            "image": image,
            "label": label,
            "point_label": point_label,
            "point_coord": point_coord,
            "filename": self.names[idx]
        }

