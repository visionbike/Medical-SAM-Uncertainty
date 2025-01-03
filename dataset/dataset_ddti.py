from typing import Optional, Callable, Dict
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from .utils import *

__all__ = [
    "DDTI"
]


class DDTI(Dataset):
    """
    DDTI Dataset for Thyroid Ultrasound Images (2D).
    Link: https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images/data
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
        self.names = [f.name[:-4] for f in sorted(Path(path, f"p_image").iterdir()) if f.is_file()]
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
        image = cv2.imread(f"{self.path}/p_image/{self.names[idx]}.PNG")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(f"{self.path}/p_mask/{self.names[idx]}.PNG", cv2.IMREAD_GRAYSCALE)
        # resize the label's resolution as same as image's
        label = cv2.resize(label, (self.image_size, self.image_size))
        # get click points
        point_coord, point_label = 1, np.array([0, 0], np.int32)
        if self.prompt == "click":
            point_coord, point_label = [], []
            # find large enough connected components as the mask
            label = np.clip(label, 0, 1)
            num_labels, labels = cv2.connectedComponents(label.astype(np.uint8))
            for label_ in range(1, num_labels):
                component_mask = np.where(labels == label_, 1, 0)
                area = np.sum(component_mask)

                if area > 400:
                    random_label, random_point = random_click(component_mask)
                    point_coord.append(random_point)
                    point_label.append(random_label)

            if len(point_coord) == 1:
                point_coord.append(point_coord[0])
                point_label.append(point_label[0])

            if len(point_coord) > 2:
                point_coord = point_label[:2]
                point_label = point_label[:2]

            point_coord = np.array(point_coord)
            point_label = np.array(point_label)
        # transform the input
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
        label = label.clamp(min=0, max=1).int()
        return {
            "image": image,
            "label": label,
            "point_label": point_label,
            "point_coord": point_coord,
            "filename": self.names[idx],
        }
