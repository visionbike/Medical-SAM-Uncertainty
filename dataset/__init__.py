from typing import Tuple
from argparse import Namespace
import torchvision.transforms as vtf
from torch.utils.data import DataLoader
from .utils import *

__all__ = [
    "get_dataloaders"
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
    transform_train = vtf.Compose([
        vtf.Resize((args.image_size, args.image_size)),
        vtf.ToTensor(),
    ])
    transform_train_seg = vtf.Compose([
        vtf.Resize((args.output_size, args.output_size,)),
        vtf.ToTensor(),
    ])
    transform_test = vtf.Compose([
        vtf.Resize((args.image_size, args.image_size)),
        vtf.ToTensor(),
    ])
    transform_test_seg = vtf.Compose([
        vtf.Resize((args.output_size, args.output_size)),
        vtf.ToTensor(),
    ])
    loader_train, loader_test = None, None
    if args.dataset == "isic":
        from .dataset_isic2016 import ISIC2016
        dataset_isic_train = ISIC2016(args.path, mode="Training", transform=transform_train, transform_mask=transform_train_seg)
        dataset_isic_test = ISIC2016(args.path, mode="Test", transform=transform_test, transform_mask=transform_test_seg)
        loader_train = DataLoader(dataset_isic_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_isic_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "refuge":
        from .dataset_refuge import REFUGE
        dataset_refuge_train = REFUGE(args.path, mode="Training", transform=transform_train, transform_mask=transform_train_seg)
        dataset_refuge_test = REFUGE(args.path, mode="Test", transform=transform_test, transform_mask=transform_test_seg)
        loader_train = DataLoader(dataset_refuge_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_refuge_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    else:
        print("The dataset is not supported now!!!")
    return loader_train, loader_test
