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
        image = cv2.imread((self.folder_images[idx] / f"{self.folder_images[idx].name}.jpg").__str__())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_cups = [
            cv2.imread((self.folder_images[idx] / f"{self.folder_images[idx].name}_seg_cup_{i}.png").__str__(), cv2.IMREAD_GRAYSCALE)
            for i in range(1, 8)
        ]
        label_discs = [
            cv2.imread((self.folder_images[idx] / f"{self.folder_images[idx].name}_seg_disc_{i}.png").__str__(), cv2.IMREAD_GRAYSCALE)
            for i in range(1, 8)
        ]
        # resize the label's resolution as same as image's
        label_cups_ = [
            cv2.resize(label_, (self.image_size, self.image_size))
            for label_ in label_cups
        ]
        label_discs_ = [
            cv2.resize(label_, (self.image_size, self.image_size))
            for label_ in label_discs
        ]
        # get click points
        point_label, point_coord_cup, point_coord_disc = 1, np.array([0, 0], np.int32),  np.array([0, 0], np.int32)
        if self.prompt == "click":
            point_label, point_coord_cup = random_click(np.mean(np.stack(label_cups_), axis=0) / 255, point_labels=1)
            point_label, point_coord_disc = random_click(np.mean(np.stack(label_discs_), axis=0) / 255, point_labels=1)
        # transform the input
        if self.transform:
            # save the current random number generate for reproducibility
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
        if self.transform_mask:
            # save the current random number generate for reproducibility
            state = torch.get_rng_state()
            label_cups = [
                torch.as_tensor((self.transform_mask(label_) > 0.5).float(), dtype=torch.float32)
                for label_ in label_cups
            ]
            label_discs = [
                torch.as_tensor((self.transform_mask(label_) > 0.5).float(), dtype=torch.float32)
                for label_ in label_discs
            ]
            torch.set_rng_state(state)
        else:
            label_cups = [
                torch.as_tensor(((label_ / 255.) > 0.5).float(), dtype=torch.float32)
                for label_ in label_cups
            ]
            label_discs = [
                torch.as_tensor(((label_ / 255.) > 0.5).float(), dtype=torch.float32)
                for label_ in label_discs
            ]
        label_cups = torch.stack(label_cups, dim=0)
        label_discs = torch.stack(label_discs, dim=0)
        label = torch.concate([label_cups.mean(dim=0), label_discs.mean(dim=0)], dim=0)
        # get bbox positions
        box_cup, box_disc = [0, 0, 0, 0], [0, 0, 0, 0]
        if self.prompt == "box":
           box_cup = random_box(label_cups)
           box_disc = random_box(label_discs)
        name = Path(self.folder_images[idx]).name
        return {
            "image": image,
            "label": label,
            "point_label": point_label,
            "point_coord": point_coord_cup,
            "point_coord_cup": point_coord_cup,
            "point_coord_disc": point_coord_disc,
            "box_cup": box_cup,
            "box_disc": box_disc,
            "filename": name
        }

