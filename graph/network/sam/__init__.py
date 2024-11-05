from typing import Tuple, Optional
from argparse import Namespace
from functools import partial
from pathlib import Path
from urllib import request
import torch
import torch.nn as nn
from ..encoder import ViT, PromptEncoder
from ..decoder import MaskDecoder
from ..layer import TwoWayTransformer
from .sam import SAM

__all__ = [
    "build_sam_vit_h",
    "build_sam_vit_l",
    "build_sam_vit_b",
    "sam_model_registry"
]


def _build_sam_model(
    args: Namespace,
    encoder_embed_dim: int,
    encoder_depth: int,
    encoder_num_heads: int,
    encoder_global_attn_indexes: Tuple[int, ...],
    checkpoint: Optional[str] = None,
) -> nn.Module:
    """
    Function to build a SAM network.

    Args:
        args (Namespace): the SAM arguments.
        encoder_embed_dim (int): the embedding dimension.
        encoder_depth (int): the depth of the encoder.
        encoder_num_heads (int): the number of attention heads.
        encoder_global_attn_indexes (Tuple): indexes for blocks using global attention.
        checkpoint (str): the checkpoint path.
    Returns:
        (nn.Module): the SAM network.
    """
    prompt_embed_dim = 256
    image_size = args.image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam_model = SAM(
        image_encoder=ViT(
            img_size=image_size,
            patch_size=vit_patch_size,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=4,
            out_channels=prompt_embed_dim,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_rel_pos=True,
            window_size=14,
            global_attn_indexes=encoder_global_attn_indexes,
            block_name=args.block,
            mid_dim=args.mid_dim
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_channels=16,
        ),
        mask_decoder=MaskDecoder(
            transformer_dim=prompt_embed_dim,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            num_multimask_outputs=args.multimask_output,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=(123.675, 116.28, 103.53),
        pixel_std=(58.395, 57.12, 57.375),
    )
    sam_model.eval()
    
    checkpoint = Path(checkpoint)
    if checkpoint.name == "sam_vit_b_01ec64.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_b_01ec64.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-B checkpoint...")
            request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_h_4b8939.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_h_4b8939.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-H checkpoint...")
            request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_l_0b3195.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_l_0b3195.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-L checkpoint...")
            request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        # create a new state dictionary with only parameters that exist in the network
        state_dict_new = {
            k: v for k, v in state_dict.items() if k in sam_model.state_dict() and sam_model.state_dict()[k].shape == v.shape
        }
        sam_model.load_state_dict(state_dict_new, strict=False)
    return sam_model


def build_sam_vit_h(args: Namespace, checkpoint: Optional[str] = None) -> nn.Module:
    """
    Args:
        args (Namespace): the SAM arguments.
        checkpoint (str): the checkpoint path.
    Returns:
        (nn.Module): the SAM ViT-H network.
    """
    return _build_sam_model(
        args=args,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=(7, 15, 23, 31),
        checkpoint=checkpoint,
    )


def build_sam_vit_l(args: Namespace, checkpoint: Optional[str] = None) -> nn.Module:
    """
    Args:
        args (Namespace): the SAM arguments.
        checkpoint (str): the checkpoint path.
    Returns:
        (nn.Module): the SAM ViT-L network.
    """
    return _build_sam_model(
        args=args,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=(5, 11, 17, 23),
        checkpoint=checkpoint,
    )


def build_sam_vit_b(args: Namespace, checkpoint: Optional[str] = None) -> nn.Module:
    """
    Args:
        args (Namespace): the SAM arguments.
        checkpoint (str): the checkpoint path.
    Returns:
        (nn.Module): the SAM ViT-B network.
    """
    return _build_sam_model(
        args=args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=(2, 5, 8, 11),
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}
