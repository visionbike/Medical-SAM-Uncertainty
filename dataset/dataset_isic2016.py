from typing import Optional, Callable, Dict
from pathlib import Path
import fireducks.pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from .utils import *

__all__ = [
    "ISIC2016"
]


class ISIC2016(Dataset):
    """
    ISIC Dataset ver. 2016 for Melanoma Segmentation from Skin Images (2D).
    Link: https://challenge.isic-archive.com/data/
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
            path(str): data path.
            mode (str): mode for loading training or testing dataset.
            prompt (str): prompt types, including
                "none": no applying prompt.
                "click": applying click prompt.
                "box": applying bbox prompt.
            image_size (int): input image size.
            transform (Callable): transform functions for image.
            transform_mask: transform function for mask.
        """
        super().__init__()
        df = pd.read_csv(Path(path, f"ISBI2016_ISIC_Part1_{mode}_GroundTruth.csv"), encoding="gbk")
        self.list_images = df.iloc[:, 1].tolist()
        self.list_labels = df.iloc[:, 2].tolist()
        self.path_data = path
        self.mode = mode
        self.prompt = prompt
        self.image_size = image_size
        self.transform = transform
        self.transform_mask = transform_mask

    def __len__(self) -> int:
        return len(self.list_images)

    def __getitem__(self, idx: int) -> Dict:
        """
        Args:
            idx (int): input index.
        Returns:
            (Dict): image, ground truth (mask), prompt data and related metadata.
        """
        # read image and label (mask)
        image = cv2.imread(Path(self.path_data, self.list_images[idx]).__str__())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(Path(self.path_data, self.list_labels[idx]).__str__(), cv2.IMREAD_GRAYSCALE)
        # get click points
        point_label, point_coord = 1, np.array([0, 0], np.int32)
        if self.prompt == "click":
            # resize label for generating point click
            label_ = cv2.resize(label, (self.image_size, self.image_size))
            point_label, point_coord = random_click(label_ / 255, point_labels=1)
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
        name = Path(self.list_images[idx]).name[:-4]
        return {
            "image": image,
            "label": label,
            "point_label": point_label,
            "point_coord": point_coord,
            "filename": name
        }

