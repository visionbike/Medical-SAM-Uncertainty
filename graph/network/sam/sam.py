from typing import Any, Dict, List, Tuple
import torch
from torch import nn
from torch.nn import functional as fn

__all__ = [
    "SAM"
]


class SAM(nn.Module):
    """
    SAM predicts object masks from an image and input prompts.
    """
    mask_threshold: float = 0.
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: nn.Module,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        pixel_mean: Tuple[float, ...] = (123.675, 116.28, 103.53),
        pixel_std: Tuple[float, ...] = (58.395, 57.12, 57.375),
    ) -> None:
        """
        Args:
            image_encoder (ViT): the backbone used to encode the image into image embeddings
                that allow for efficient mask prediction.
            prompt_encoder (PromptEncoder): encodes various types of input prompts.
            mask_decoder (MaskDecoder): predicts masks from the image embeddings and encoded prompts.
            pixel_mean (Tuple): mean values for normalizing pixels in the input image.
            pixel_std (Tuple): standard deviations for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        """
        Returns:
            (Any): the device.
        """
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is recommended over calling the network directly.

        Args:
            batched_input (List): a list over input images, each a dictionary with the following keys.
                A prompt key can be excluded if it is not present.
                    'image': the image as a torch tensor in 3 x H x W format, already transformed for input to the network.
                    'original_size' (Tuple): the original size of the image before transformation, as (H, W).
                    'point_coords' (torch.Tensor) batched point prompts for this image, with shape B x N x 2.
                        Already transformed to the input frame of the network.
                    'point_labels' (torch.Tensor): batched labels for point prompts, with shape B x N.
                    'boxes' (torch.Tensor): batched box inputs, with shape B x 4.
                        Already transformed to the input frame of the network.
                    'mask_inputs' (torch.Tensor): batched mask inputs to the network, in the form B x 1 x H x W.
            multimask_output (bool): whether the network should predict multiple disambiguating masks, or return a single mask.
        Returns:
            (List): A list over input images, where each element is as dictionary with the following keys.
                'masks' (torch.Tensor): batched binary mask predictions, with shape B x C x H x W,
                    where B is the number of input prompts, C is determined by multimask_output, and
                    (H, W) is the original size of the image.
                'iou_predictions'(torch.Tensor): the network's predictions of mask quality, in shape B x C.
                'low_res_logits'(torch.Tensor): low resolution logits with shape B x C x H x W, where H = W = 256.
                    Can be passed as mask input to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (torch.Tensor): batched masks from the mask_decoder, in B x C x H x W format.
            input_size (Tuple): the size of the image input to the network, in (H, W) format. Used to remove padding.
            original_size (Tuple): the original size of the image before resizing for input to the network, in (H, W) format.
        Returns:
            (torch.Tensor): batched masks in BxCxHxW format, where (H, W) is given by original_size.
        """
        masks = fn.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = fn.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize pixel values and pad to a square input.

        Args:
            x (torch.Tensor): batched input images, with shape B x C x H x W.
        Returns:
            (torch.Tensor): batched preprocessed images, with shape B x C x H x W.
        """
        # normalize colors.
        x = (x - self.pixel_mean) / self.pixel_std
        # pad.
        h, w = x.shape[-2:]
        pad_h = self.image_encoder.img_size - h
        pad_w = self.image_encoder.img_size - w
        x = fn.pad(x, (0, pad_w, 0, pad_h))
        return x
