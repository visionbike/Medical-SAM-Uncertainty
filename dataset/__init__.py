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
    Get dataloaders for training and validation.

    Args:
        args (Namespace): Data configuration.
    Returns:
        (Tuple): train and test dataloaders.
    """
    # define image/mask transformation for train/test datasets
    transform_image = vtf2.Compose([
        vtf2.ToImage(),
        vtf2.ToDtype(torch.uint8),
        vtf2.Resize((args.image_size, args.image_size)),
        vtf2.ToDtype(torch.float32, scale=True),
    ])
    transform_label = vtf2.Compose([
        vtf2.ToImage(),
        vtf2.ToDtype(torch.uint8),
        vtf2.Resize((args.output_size, args.output_size)),
        vtf2.ToDtype(torch.float32, scale=True),
    ])
    loader_train, loader_test = None, None
    if args.dataset == "isic":
        from .dataset_isic2016 import ISIC2016
        dataset_isic_train = ISIC2016(args.path, mode="Training", image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        dataset_isic_test = ISIC2016(args.path, mode="Test", image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        loader_train = DataLoader(dataset_isic_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_isic_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "refuge":
        from .dataset_refuge import REFUGE
        dataset_refuge_train = REFUGE(args.path, mode="Training", image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        dataset_refuge_test = REFUGE(args.path, mode="Test", image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        loader_train = DataLoader(dataset_refuge_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_refuge_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "ddti":
        from .dataset_ddti import DDTI
        dataset_ddti = DDTI(args.path, image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        dataset_ddti_size = len(dataset_ddti)
        dataset_ddti_indices = list(range(dataset_ddti_size))
        split = int(np.floor(0.2 * dataset_ddti_size))
        sampler_test = SubsetRandomSampler(dataset_ddti_indices[:split])
        sampler_train = SubsetRandomSampler(dataset_ddti_indices[split:])
        loader_train = DataLoader(dataset_ddti, batch_size=args.batch_size, sampler=sampler_train, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_ddti, batch_size=args.batch_size, sampler=sampler_test, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "stare":
        from .dataset_stare import STARE
        dataset_stare = STARE(args.path, image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        dataset_stare_size = len(dataset_stare)
        dataset_stare_indices = list(range(dataset_stare_size))
        split = int(np.floor(0.2 * dataset_stare_size))
        sampler_test = SubsetRandomSampler(dataset_stare_indices[:split])
        sampler_train = SubsetRandomSampler(dataset_stare_indices[split:])
        loader_train = DataLoader(dataset_stare, batch_size=args.batch_size, sampler=sampler_train, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_stare, batch_size=args.batch_size, sampler=sampler_test, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "drive":
        from .dataset_drive import DRIVE
        dataset_drive_train = DRIVE(args.path, mode="train", image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        dataset_drive_test = DRIVE(args.path, mode="test", image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        loader_train = DataLoader(dataset_drive_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_drive_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "idrid":
        from .dataset_idrid import IDRiD
        dataset_idrid_train = IDRiD(args.path, mode="Training", image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        dataset_idrid_test = IDRiD(args.path, mode="Testing", image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        loader_train = DataLoader(dataset_idrid_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_idrid_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "flare":
        from .dataset_flare22 import FLARE22
        dataset_flare = FLARE22(args.path, image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        dataset_flare_size = len(dataset_flare)
        dataset_flare_indices = list(range(dataset_flare_size))
        split = int(np.floor(0.2 * dataset_flare_size))
        sampler_test = SubsetRandomSampler(dataset_flare_indices[:split])
        sampler_train = SubsetRandomSampler(dataset_flare_indices[split:])
        loader_train = DataLoader(dataset_flare, batch_size=args.batch_size, sampler=sampler_train, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_flare, batch_size=args.batch_size, sampler=sampler_test, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "lits":
        from .dataset_lits17 import LiTS17
        dataset_lits = LiTS17(args.path, image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        dataset_lits_size = len(dataset_lits)
        dataset_lits_indices = list(range(dataset_lits_size))
        split = int(np.floor(0.2 * dataset_lits_size))
        sampler_test = SubsetRandomSampler(dataset_lits_indices[:split])
        sampler_train = SubsetRandomSampler(dataset_lits_indices[split:])
        loader_train = DataLoader(dataset_lits, batch_size=args.batch_size, sampler=sampler_train, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_lits, batch_size=args.batch_size, sampler=sampler_test, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "runmc":
        from .dataset_runmc import RUNMC
        dataset_runmc = RUNMC(args.path, image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        dataset_runmc_size = len(dataset_runmc)
        dataset_runmc_indices = list(range(dataset_runmc_size))
        split = int(np.floor(0.2 * dataset_runmc_size))
        sampler_test = SubsetRandomSampler(dataset_runmc_indices[:split])
        sampler_train = SubsetRandomSampler(dataset_runmc_indices[split:])
        loader_train = DataLoader(dataset_runmc, batch_size=args.batch_size, sampler=sampler_train, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_runmc, batch_size=args.batch_size, sampler=sampler_test, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "bmc":
        from .dataset_bmc import BMC
        dataset_bmc = BMC(args.path, image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        dataset_bmc_size = len(dataset_bmc)
        dataset_bmc_indices = list(range(dataset_bmc_size))
        split = int(np.floor(0.2 * dataset_bmc_size))
        sampler_test = SubsetRandomSampler(dataset_bmc_indices[:split])
        sampler_train = SubsetRandomSampler(dataset_bmc_indices[split:])
        loader_train = DataLoader(dataset_bmc, batch_size=args.batch_size, sampler=sampler_train, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_bmc, batch_size=args.batch_size, sampler=sampler_test, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "i2cvb":
        from .dataset_i2cvb import I2CVB
        dataset_i2cvb = I2CVB(args.path, image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        dataset_i2cvb_size = len(dataset_i2cvb)
        dataset_i2cvb_indices = list(range(dataset_i2cvb_size))
        split = int(np.floor(0.2 * dataset_i2cvb_size))
        sampler_test = SubsetRandomSampler(dataset_i2cvb_indices[:split])
        sampler_train = SubsetRandomSampler(dataset_i2cvb_indices[split:])
        loader_train = DataLoader(dataset_i2cvb, batch_size=args.batch_size, sampler=sampler_train, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_i2cvb, batch_size=args.batch_size, sampler=sampler_test, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "bidmc":
        from .dataset_bidmc import BIDMC
        dataset_bidmc = BIDMC(args.path, image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        dataset_bidmc_size = len(dataset_bidmc)
        dataset_bidmc_indices = list(range(dataset_bidmc_size))
        split = int(np.floor(0.2 * dataset_bidmc_size))
        sampler_test = SubsetRandomSampler(dataset_bidmc_indices[:split])
        sampler_train = SubsetRandomSampler(dataset_bidmc_indices[split:])
        loader_train = DataLoader(dataset_bidmc, batch_size=args.batch_size, sampler=sampler_train, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_bidmc, batch_size=args.batch_size, sampler=sampler_test, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "ucl":
        from .dataset_ucl import UCL
        dataset_ucl = UCL(args.path, image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        dataset_ucl_size = len(dataset_ucl)
        dataset_ucl_indices = list(range(dataset_ucl_size))
        split = int(np.floor(0.2 * dataset_ucl_size))
        sampler_test = SubsetRandomSampler(dataset_ucl_indices[:split])
        sampler_train = SubsetRandomSampler(dataset_ucl_indices[split:])
        loader_train = DataLoader(dataset_ucl, batch_size=args.batch_size, sampler=sampler_train, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_ucl, batch_size=args.batch_size, sampler=sampler_test, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "hk":
        from .dataset_hk import HK
        dataset_hk = HK(args.path, image_size=args.image_size, transform=transform_image, transform_mask=transform_label)
        dataset_hk_size = len(dataset_hk)
        dataset_hk_indices = list(range(dataset_hk_size))
        split = int(np.floor(0.2 * dataset_hk_size))
        sampler_test = SubsetRandomSampler(dataset_hk_indices[:split])
        sampler_train = SubsetRandomSampler(dataset_hk_indices[split:])
        loader_train = DataLoader(dataset_hk, batch_size=args.batch_size, sampler=sampler_train, num_workers=args.workers, pin_memory=True)
        loader_test = DataLoader(dataset_hk, batch_size=args.batch_size, sampler=sampler_test, num_workers=args.workers, pin_memory=True)
    else:
        print("The dataset is not supported now!!!")
    return loader_train, loader_test
