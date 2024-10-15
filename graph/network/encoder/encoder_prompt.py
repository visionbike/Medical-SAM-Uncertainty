from typing import Optional, Tuple, Type
import torch
from torch import nn
from ..layer import PositionEmbeddingRandom, LayerNorm2d

__all__ = [
    "PromptEncoder"
]


class PromptEncoder(nn.Module):
    """
    Encodes prompts for input to SAM's mask decoder.
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_channels: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Args:
            embed_dim (int): the prompts' embedding dimension.
            image_embedding_size (Tuple): the spatial size of the image embedding, as (H, W).
            input_image_size (int): the padded size of the image as input to the image encoder, as (H, W).
            mask_in_channels (int): the number of hidden channels used for encoding input masks.
            activation (nn.Module): the activation to use when encoding input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_channels // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_channels // 4),
            activation(),
            nn.Conv2d(mask_in_channels // 4, mask_in_channels, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_channels),
            activation(),
            nn.Conv2d(mask_in_channels, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          (torch.Tensor): Positional encoding with shape 1 x embed_dim x embedding_h x embedding_w.
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    @staticmethod
    def _get_batch_size(
            points: Optional[Tuple[torch.Tensor, torch.Tensor]],
            boxes: Optional[torch.Tensor],
            masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.

        Args:
            points (torch.Tensor): the input points.
            boxes (torch.Tensor): the input boxes.
            masks (torch.Tensor): the input masks.
        Returns:
            (int): the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """
        Embeds point prompts.

        Args:
            points (torch.Tensor): the input points.
            labels (torch.Tensor): the input labels.
            pad (bool): whether to pad the input points.
        Returns:
            (torch.Tensor): the embedded points.
        """
        points = points + 0.5  # shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Embeds bounding boxes.

        Args:
            boxes (torch.Tensor): the input boxes.
        Returns:
            (torch.Tensor): the embedded boxes.
        """
        boxes = boxes + 0.5  # shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Embeds masks inputs.

        Args:
            masks (torch.Tensor): the input masks.
        Returns:
            (torch.Tensor): the embedded masks.
        """
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (Tuple): point coordinates and labels to embed.
            boxes (torch.Tensor): boxes to embed.
            masks (torch.Tensor): masks to embed.
        Returns:
            (torch.Tensor): sparse embeddings for the points and boxes, with shape B x N x embed_dim,
                 where N is determined by the number of input points and boxes.
            (torch.Tensor): dense embeddings for the masks, in the shape B x embed_dim x embed_H x embed_W.
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        return sparse_embeddings, dense_embeddings
