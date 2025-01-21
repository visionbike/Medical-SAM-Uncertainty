from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from dataset.utils import *

if __name__ == "__main__":
    print("Preprocessing Site A (RUNMC) dataset...")
    path_data = Path("./data/PROSTATE/RUNMC")
    path_image = Path("./data/PROSTATE/RUNMC/images")
    path_label = Path("./data/PROSTATE/RUNMC/labels")
    path_image.mkdir(parents=True, exist_ok=True)
    path_label.mkdir(parents=True, exist_ok=True)
    filenames = [filepath.name[:6] for filepath in path_data.glob("*_segmentation.nii.gz")]
    with tqdm(total=len(filenames)) as pbar:
        for filename in filenames:
            image = load_nii(f"{path_data}/{filename}.nii.gz")
            label = load_nii(f"{path_data}/{filename}_segmentation.nii.gz")
            pbar.set_postfix(**{"label": len(np.unique(label)), "shape": image.shape})
            # unify the prostate label
            label[label >= 1] = 255
            label = label.astype(np.uint8)
            # save one of every two CT slices
            for i in range(image.shape[-1]):
                label_slice = label[..., i]
                image_slice = image[..., i]
                if label_slice.sum() == 0:
                    continue
                image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
                image_slice = (np.expand_dims(image_slice, -1) * 255).astype(np.uint8)
                cv2.imwrite(f"{path_image}/{filename}_slice_{i}.png", image_slice)
                cv2.imwrite(f"{path_label}/{filename}_slice_{i}.png", label_slice)
            pbar.update()

    print("Preprocessing Site B (BMC) dataset...")
    path_data = Path("./data/PROSTATE/BMC")
    path_image = Path("./data/PROSTATE/BMC/images")
    path_label = Path("./data/PROSTATE/BMC/labels")
    path_image.mkdir(parents=True, exist_ok=True)
    path_label.mkdir(parents=True, exist_ok=True)
    filenames = [filepath.name[:6] for filepath in path_data.glob("*_Segmentation.nii.gz")]
    with tqdm(total=len(filenames)) as pbar:
        for filename in filenames:
            image = load_nii(f"{path_data}/{filename}.nii.gz")
            label = load_nii(f"{path_data}/{filename}_Segmentation.nii.gz")
            pbar.set_postfix(**{"label": len(np.unique(label)), "shape": image.shape})
            # unify the prostate label
            label[label >= 1] = 255
            label = label.astype(np.uint8)
            # save one of every two CT slices
            for i in range(image.shape[-1]):
                label_slice = label[..., i]
                image_slice = image[..., i]
                if label_slice.sum() == 0:
                    continue
                image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
                image_slice = (np.expand_dims(image_slice, -1) * 255).astype(np.uint8)
                cv2.imwrite(f"{path_image}/{filename}_slice_{i}.png", image_slice)
                cv2.imwrite(f"{path_label}/{filename}_slice_{i}.png", label_slice)
            pbar.update()

    print("Preprocessing Site C (I2CVB) dataset...")
    path_data = Path("./data/PROSTATE/I2CVB")
    path_image = Path("./data/PROSTATE/I2CVB/images")
    path_label = Path("./data/PROSTATE/I2CVB/labels")
    path_image.mkdir(parents=True, exist_ok=True)
    path_label.mkdir(parents=True, exist_ok=True)
    filenames = [filepath.name[:6] for filepath in path_data.glob("*_segmentation.nii.gz")]
    with tqdm(total=len(filenames)) as pbar:
        for filename in filenames:
            image = load_nii(f"{path_data}/{filename}.nii.gz")
            label = load_nii(f"{path_data}/{filename}_segmentation.nii.gz")
            pbar.set_postfix(**{"label": len(np.unique(label)), "shape": image.shape})
            # unify the prostate label
            label[label >= 1] = 255
            label = label.astype(np.uint8)
            # save one of every two CT slices
            for i in range(image.shape[-1]):
                label_slice = label[..., i]
                image_slice = image[..., i]
                if label_slice.sum() == 0:
                    continue
                image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
                image_slice = (np.expand_dims(image_slice, -1) * 255).astype(np.uint8)
                cv2.imwrite(f"{path_image}/{filename}_slice_{i}.png", image_slice)
                cv2.imwrite(f"{path_label}/{filename}_slice_{i}.png", label_slice)
            pbar.update()

    print("Preprocessing Site D (UCL) dataset...")
    path_data = Path("./data/PROSTATE/UCL")
    path_image = Path("./data/PROSTATE/UCL/images")
    path_label = Path("./data/PROSTATE/UCL/labels")
    path_image.mkdir(parents=True, exist_ok=True)
    path_label.mkdir(parents=True, exist_ok=True)
    filenames = [filepath.name[:6] for filepath in path_data.glob("*_segmentation.nii.gz")]
    with tqdm(total=len(filenames)) as pbar:
        for filename in filenames:
            image = load_nii(f"{path_data}/{filename}.nii.gz")
            label = load_nii(f"{path_data}/{filename}_segmentation.nii.gz")
            pbar.set_postfix(**{"label": len(np.unique(label)), "shape": image.shape})
            # unify the prostate label
            label[label >= 1] = 255
            label = label.astype(np.uint8)
            # save one of every two CT slices
            for i in range(image.shape[-1]):
                label_slice = label[..., i]
                image_slice = image[..., i]
                if label_slice.sum() == 0:
                    continue
                image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
                image_slice = (np.expand_dims(image_slice, -1) * 255).astype(np.uint8)
                cv2.imwrite(f"{path_image}/{filename}_slice_{i}.png", image_slice)
                cv2.imwrite(f"{path_label}/{filename}_slice_{i}.png", label_slice)
            pbar.update()

    print("Preprocessing Site E (BIDMC) dataset...")
    path_data = Path("./data/PROSTATE/BIDMC")
    path_image = Path("./data/PROSTATE/BIDMC/images")
    path_label = Path("./data/PROSTATE/BIDMC/labels")
    path_image.mkdir(parents=True, exist_ok=True)
    path_label.mkdir(parents=True, exist_ok=True)
    filenames = [filepath.name[:6] for filepath in path_data.glob("*_segmentation.nii.gz")]
    with tqdm(total=len(filenames)) as pbar:
        for filename in filenames:
            image = load_nii(f"{path_data}/{filename}.nii.gz")
            label = load_nii(f"{path_data}/{filename}_segmentation.nii.gz")
            pbar.set_postfix(**{"label": len(np.unique(label)), "shape": image.shape})
            # unify the prostate label
            label[label >= 1] = 255
            label = label.astype(np.uint8)
            # save one of every two CT slices
            for i in range(image.shape[-1]):
                label_slice = label[..., i]
                image_slice = image[..., i]
                if label_slice.sum() == 0:
                    continue
                image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
                image_slice = (np.expand_dims(image_slice, -1) * 255).astype(np.uint8)
                cv2.imwrite(f"{path_image}/{filename}_slice_{i}.png", image_slice)
                cv2.imwrite(f"{path_label}/{filename}_slice_{i}.png", label_slice)
            pbar.update()

    print("Preprocessing Site F (HK) dataset...")
    path_data = Path("./data/PROSTATE/HK")
    path_image = Path("./data/PROSTATE/HK/images")
    path_label = Path("./data/PROSTATE/HK/labels")
    path_image.mkdir(parents=True, exist_ok=True)
    path_label.mkdir(parents=True, exist_ok=True)
    filenames = [filepath.name[:6] for filepath in path_data.glob("*_segmentation.nii.gz")]
    with tqdm(total=len(filenames)) as pbar:
        for filename in filenames:
            image = load_nii(f"{path_data}/{filename}.nii.gz")
            label = load_nii(f"{path_data}/{filename}_segmentation.nii.gz")
            pbar.set_postfix(**{"label": len(np.unique(label)), "shape": image.shape})
            # unify the prostate label
            label[label >= 1] = 255
            label = label.astype(np.uint8)
            # save one of every two CT slices
            for i in range(image.shape[-1]):
                label_slice = label[..., i]
                image_slice = image[..., i]
                if label_slice.sum() == 0:
                    continue
                image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
                image_slice = (np.expand_dims(image_slice, -1) * 255).astype(np.uint8)
                cv2.imwrite(f"{path_image}/{filename}_slice_{i}.png", image_slice)
                cv2.imwrite(f"{path_label}/{filename}_slice_{i}.png", label_slice)
            pbar.update()