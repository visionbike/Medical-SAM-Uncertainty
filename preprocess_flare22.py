from pathlib import Path
import numpy as np
import cv2
from dataset.utils import *

if __name__ == "__main__":
    path_data = Path("./data/FLARE22")
    path_image = Path("./data/FLARE22/images")
    path_label = Path("./data/FLARE22/labels")
    path_image.mkdir(parents=True, exist_ok=True)
    path_label.mkdir(parents=True, exist_ok=True)
    #
    for filepath in (path_data / "images_nii").iterdir():
        filename = filepath.name
        print(f"Processing {filename}...")
        image = load_nii(f"{path_data}/images_nii/{filename}")
        label = load_nii(f"{path_data}/labels_nii/{filename[:-9]}.nii/{filename[:-9]}.nii")
        assert image.shape == label.shape
        print(f"CT Shape: {image.shape}, Label shape: {label.shape}")
        # only get liver mask
        label[label > 1] = 0
        label[label == 1] = 255
        label = label.astype(np.uint8)
        # save one of every two CT slices
        for i in range(0, image.shape[-1], 2):
            label_slice = label[..., i]
            image_slice = image[..., i]
            if label_slice.sum() == 0:
                continue
            image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
            image_slice = (np.expand_dims(image_slice, -1) * 255).astype(np.uint8)
            cv2.imwrite(f"{path_image}/{filename[:-4]}_slice_{i}.png", image_slice)
            cv2.imwrite(f"{path_label}/{filename[:-4]}_slice_{i}.png", label_slice)
