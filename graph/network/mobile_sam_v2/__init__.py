from typing import Optional
from argparse import Namespace
import torch
import torch.nn as nn
from ..encoder import TinyViT, PromptEncoder
from ..decoder import MaskDecoder
from ..layer import TwoWayTransformer
from .mobile_sam_v2 import MobileSAMV2

__all__ = [
    "build_mobile_sam_v2_vit_t",
    "mobile_sam_v2_model_registry"
]


def _build_mobile_sam_v2_model(
    args: Namespace,
    checkpoint: Optional[str] = None,
) -> nn.Module:
    prompt_embed_dim = 256
    image_size = args.image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam_model = MobileSAMV2(
        image_encoder=TinyViT(
            img_size=image_size,
            in_channels=3,
            num_classes=1000,
            embed_dims=(64, 128, 160, 320),
            depths=(2, 2, 6, 2),
            num_heads=(2, 4, 5, 10),
            window_sizes=(7, 7, 14, 7),
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            use_checkpoint=False,
            mbconv_expand_ratio=4.,
            local_conv_size=3,
            layer_lr_decay=0.8,
            block_name=args.block_name,
            mid_dim=args.mid_dim
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_channels=16,
        ),
        mask_decoder=MaskDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            num_multimask_outputs=args.num_multimask_outputs,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=(123.675, 116.28, 103.53),
        pixel_std=(58.395, 57.12, 57.375),
    )
    mobile_sam_model.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        mobile_sam_model.load_state_dict(state_dict, strict=False)
    return mobile_sam_model


def build_mobile_sam_v2_vit_t(args: Namespace, checkpoint: Optional[str] = None) -> nn.Module:
    """
    Args:
        args (Namespace): the SAM arguments.
        checkpoint (str): the checkpoint path.
    Returns:
        (nn.Module): the SAM ViT-L network.
    """
    return _build_mobile_sam_v2_model(args, checkpoint)


mobile_sam_v2_model_registry = {
    "default": build_mobile_sam_v2_vit_t,
    "vit_t": build_mobile_sam_v2_vit_t
}
