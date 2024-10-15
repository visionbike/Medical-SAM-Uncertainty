from typing import Optional
import torch
import torchvision.utils as vultils
import torchvision.transforms as vfn

__all__ = [
    "visualize_images"
]

def visualize_images(
        image: torch.Tensor,
        mask_prd: torch.Tensor,
        mask_tgt: torch.Tensor,
        map_entropy: Optional[torch.Tensor] = None,
        map_mae: Optional[torch.Tensor] = None,
        save_path: str = "",
        reverse: bool = False,
) -> None:
    """
    Visualize images: input, predicted and target images.

    Args:
        image (Tensor): input image.
        mask_prd (Tensor): predicted mask.
        mask_tgt (Tensor): target mask.
        map_entropy (Tensor): entropy map.
        map_mae (Tensor): MAE map.
        save_path (str): saving path.
        reverse (bool): whether to make reversed-value map.
    """
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
    if C > 1:
        for i in range(C):
            mask_prds.append(mask_prd[: num_rows, i, :, :].unsqueeze(1).expand(B, 3, H, W))
            mask_tgts.append(mask_tgt[: num_rows, i, :, :].unsqueeze(1).expand(B, 3, H, W))
            if map_entropy is not None:
                map_entropies.append(map_entropy[: num_rows, i, :, :].unsqueeze(1).expand(B, 3, H, W))
            if map_maes is not None:
                map_maes.append(map_mae[: num_rows, i, :, :].unsqueeze(1).expand(B, 3, H, W))
    else:
        mask_prds.append(mask_prd[: num_rows, 0, :, :].unsqueeze(1).expand(B, 3, H, W))
        mask_tgts.append(mask_tgt[: num_rows, 0, :, :].unsqueeze(1).expand(B, 3, H, W))
        if map_entropy is not None:
            map_entropies.append(map_entropy[: num_rows, 0, :, :].unsqueeze(1).expand(B, 3, H, W))
        if map_maes is not None:
            map_maes.append(map_mae[: num_rows, 0, :, :].unsqueeze(1).expand(B, 3, H, W))
    compose = torch.cat([image[: num_rows, :, :], mask_prds, mask_tgts], dim=0)
    if map_entropy is not None:
        compose = torch.cat([compose, map_entropies], dim=0)
    if map_maes is not None:
        compose = torch.cat([compose, map_maes], dim=0)
    vultils.save_image(compose, fp=save_path, nrow=num_rows, padding=10)
