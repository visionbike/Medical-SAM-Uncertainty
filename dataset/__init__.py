from typing import Tuple
from argparse import Namespace
import numpy as np
import torch
import torchvision.transforms.v2 as vtf2
from torch.utils.data import DataLoader, SubsetRandomSampler
from .utils import *

__all__ = [
    "get_dataloaders",
    "generate_click_prompt",
    "random_click",
    "random_box"
]


def get_dataloaders(args: Namespace) -> Tuple[DataLoader, DataLoader]:
    """
    Get dataloaders for training and validation
    Args:
        args (Namespace): Data configuration.
    Returns:
        (Tuple): train and test dataloaders.
    """
    # define image/mask transformation for train/test datasets
    transform_train_image = vtf2.Compose([
        vtf2.ToImage(),
        vtf2.ToDtype(torch.uint8),
        vtf2.Resize((args.image_size, args.image_size)),
        vtf2.ToDtype(torch.float32, scale=True),
    ])
    transform_train_label = vtf2.Compose([
        vtf2.ToImage(),
        vtf2.ToDtype(torch.uint8,),
        vtf2.Resize((args.output_size, args.output_size)),
        vtf2.ToDtype(torch.float32, scale=True),
    ])
    transform_test_image = vtf2.Compose([
        vtf2.ToImage(),
        vtf2.ToDtype(torch.uint8),
        vtf2.Resize((args.image_size, args.image_size)),
        vtf2.ToDtype(torch.float32, scale=True),
    ])
    transform_test_label = vtf2.Compose([
        vtf2.ToImage(),
        vtf2.ToDtype(torch.uint8),
        vtf2.Resize((args.output_size, args.output_size)),
        vtf2.ToDtype(torch.float32, scale=True),
    ])
    loader_train, loader_test = None, None
    if args.dataset == "isic":
        from .dataset_isic2016 import ISIC2016
        dataset_isic_train = ISIC2016(args.path, mode="Training", image_size=args.image_size, transform=transform_train_image, transform_mask=transform_train_label)
        dataset_isic_test = ISIC2016(args.path, mode="Test", image_size=args.image_size, transform=transform_test_image, transform_mask=transform_test_label)
        loader_train = DataLoader(dataset_isic_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_isic_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "refuge":
        from .dataset_refuge import REFUGE
        dataset_refuge_train = REFUGE(args.path, mode="Training", image_size=args.image_size, transform=transform_train_image, transform_mask=transform_train_label)
        dataset_refuge_test = REFUGE(args.path, mode="Test", image_size=args.image_size, transform=transform_test_image, transform_mask=transform_test_label)
        loader_train = DataLoader(dataset_refuge_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_refuge_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "ddti":
        from .dataset_ddti import DDTI
        dataset_ddti = DDTI(args.path, image_size=args.image_size, transform=transform_train_image, transform_mask=transform_train_label)
        dataset_ddti_size = len(dataset_ddti)
        dataset_ddti_indices = list(range(dataset_ddti_size))
        split = int(np.floor(0.2 * dataset_ddti_size))
        sampler_train = SubsetRandomSampler(dataset_ddti_indices[:split])
        sampler_test = SubsetRandomSampler(dataset_ddti_indices[split:])
        loader_train = DataLoader(dataset_ddti, batch_size=args.batch_size, sampler=sampler_train, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_ddti, batch_size=args.batch_size, sampler=sampler_test, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "stare":
        from .dataset_stare import STARE
        dataset_stare = STARE(args.path, image_size=args.image_size, transform=transform_train_image, transform_mask=transform_train_label)
        dataset_stare_size = len(dataset_stare)
        dataset_stare_indices = list(range(dataset_stare_size))
        split = int(np.floor(0.2 * dataset_stare_size))
        sampler_train = SubsetRandomSampler(dataset_stare_indices[:split])
        sampler_test = SubsetRandomSampler(dataset_stare_indices[split:])
        loader_train = DataLoader(dataset_stare, batch_size=args.batch_size, sampler=sampler_train, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_stare, batch_size=args.batch_size, sampler=sampler_test, num_workers=args.workers, pin_memory=True)
    else:
        print("The dataset is not supported now!!!")
    return loader_train, loader_test
