from typing import Optional
from pathlib import Path
import torch
import torchvision.utils as vultils
import torchvision.transforms as vfn
from torch.utils.tensorboard import SummaryWriter

__all__ = [
    "visualize_images"
]


def visualize_images(
        image: torch.Tensor,
        mask_tgt: torch.Tensor,
        mask_prd: torch.Tensor,
        map_entropy: Optional[torch.Tensor] = None,
        map_mae: Optional[torch.Tensor] = None,
        save_path: Path = "logs",
        filename: str = "vis",
        prefix: Optional[str] = None,
        writer: Optional[SummaryWriter] = None,
        reverse: bool = False,
) -> None:
    """
    Visualize images: input, predicted and target images.

    Args:
        image (Tensor): input image.
        mask_tgt (Tensor): target mask.
        mask_prd (Tensor): predicted mask.
        map_entropy (Tensor): entropy map.
        map_mae (Tensor): MAE map.
        save_path (str): saving path.
        filename (str): image name.
        prefix (str): prefix string.
        writer (SummaryWriter): tensorboard writer.
        reverse (bool): whether to make reversed-value map.
    """
    file_path = save_path / (f"{prefix}_{filename}.jpg" if prefix is not None else f"{filename}.jpg")
    B, C, H, W = mask_prd.shape
    if image.shape[2] != mask_prd.shape[2]:
        image = vfn.Resize((H, W))(image)
    # number of visualized images
    num_rows = min(B, 4)

    if torch.max(mask_prd) > 1 or torch.min(mask_prd) < 0:
        mask_prd = torch.sigmoid(mask_prd)

    if reverse:
        mask_prd = 1. - mask_prd
        mask_tgt = 1. - mask_tgt

    mask_prds = []
    mask_tgts = []
    map_entropies = []
    map_maes = []

    for i in range(C):
        mask_prds.append(mask_prd[: num_rows, i, :, :].unsqueeze(1).expand(B, 3, H, W))
        mask_tgts.append(mask_tgt[: num_rows, i, :, :].unsqueeze(1).expand(B, 3, H, W))
        if map_entropy is not None:
            map_entropies.append(map_entropy[: num_rows, i, :, :].unsqueeze(1).expand(B, 3, H, W))
        if map_mae is not None:
            map_maes.append(map_mae[: num_rows, i, :, :].unsqueeze(1).expand(B, 3, H, W))
    compose = [image[: num_rows, :, :]] + mask_tgts + mask_prds
    if map_entropy is not None:
        compose += map_maes
    if map_mae is not None:
        compose += map_entropies
    compose = torch.cat(compose, dim=0)
    if writer is not None:
        writer.add_images("Visualization", compose, dataformats="NCHW")
    vultils.save_image(compose, fp=file_path, nrow=num_rows, padding=10)
