from typing import List, Tuple, Type
import torch
from torch import nn
from ..layer import MLP2, LayerNorm2d

__all__ = [
    "MaskDecoder"
]


class MaskDecoder(nn.Module):
    """
    Predicts masks given an image and prompt embeddings, using a transformer architecture.
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Args:
            transformer_dim (int): the channel dimension of the transformer.
            transformer (nn.Module): the transformer architecture used to predict masks.
            num_multimask_outputs (int): the number of masks to predict when disambiguating masks.
            activation (nn.Module): the activation function of the transformer when upscaling masks.
            iou_head_depth (int): the depth of the MLP used to predict masks.
            iou_head_hidden_dim (int): the hidden dimension of the MLP used to predict masks quality.
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP2(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )
        self.iou_prediction_head = MLP2(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (torch.Tensor): the embeddings from the image encoder.
            image_pe (torch.Tensor): positional encoding with the shape of image_embeddings.
            sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes.
            dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs.
        Returns:
            (torch.Tensor): batched predicted masks.
            (torch.Tensor): batched predictions of mask quality.
        """
        # concatenate output tokens.
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # expand per-image data in batch direction to be per-mask.
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        # run the transformer.
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]
        # upscale mask embeddings and predict masks using the mask tokens.
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # generate mask quality predictions.
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_embeddings (torch.Tensor): the embeddings from the image encoder.
            image_pe (torch.Tensor): positional encoding with the shape of image_embeddings.
            sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes.
            dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs.
            multimask_output (bool): Whether to return multiple masks or a single.
        Returns:
            (torch.Tensor): batched predicted masks.
            (torch.Tensor): batched predictions of mask quality.
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        # select the correct mask or masks for output.
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        # prepare output.
        return masks, iou_pred
