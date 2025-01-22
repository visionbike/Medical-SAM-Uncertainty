from pathlib import Path
import numpy as np
import cv2
from dataset.utils import *

if __name__ == "__main__":
    path_images_nii = Path("./data/LiTS17/volumes_nii")
    path_labels_nii = Path("./data/LiTS17/labels_nii")
    #
    path_images = Path("./data/LiTS17/images")
    path_labels = Path("./data/LiTS17/labels")
    path_images.mkdir(parents=True, exist_ok=True)
    path_labels.mkdir(parents=True, exist_ok=True)
    #
    for i in range(0, 131):
        print(f"Processing {i}.nii...")
        image = load_nii(f"{path_images_nii}/volume-{i}.nii")
        label = load_nii(f"{path_labels_nii}/segmentation-{i}.nii")
        assert image.shape == label.shape
        print(f"CT Shape: {image.shape}, Label shape: {label.shape}")
        # only get liver mask
        label[label > 1] = 0
        label[label == 1] = 255
        label = label.astype(np.uint8)
        for j in range(0, image.shape[-1], 2):
            label_slice = label[..., j]
            image_slice = image[..., j]
            if label_slice.sum() == 0:
                continue
            image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
            image_slice = (np.expand_dims(image_slice, -1) * 255).astype(np.uint8)
            cv2.imwrite(f"{path_images}/LiTS17_{i}_slice_{j}.png", image_slice)
            cv2.imwrite(f"{path_labels}/LiTS17_{i}_slice_{j}.png", label_slice)
