from typing import Tuple, Dict
from abc import ABC
import time
from logging import Logger
from argparse import Namespace
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as fn
from utils.log import create_logger, create_log_directory
from .model_base import ModelBase
from graph.network import get_network
from graph.optimizer import get_optimizer
from graph.loss import get_loss
from graph.metric import get_metrics
from utils import get_device, visualize_images

__all__ = [
    "ModelSAM"
]

class ModelSAM(ModelBase, ABC):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        self.device, self.device_ids = get_device(args.ExpConfig.use_gpu, args.ExpConfig.gpu_device, args.ExpConfig.distributed)
        self.net = get_network(args.NetworkConfig, use_gpu=args.ExpConfig.use_gpu, device=self.device, distributed=self.device_ids, pretrain=args.ExpConfig.pretrain)
        self.optimizer, self.lr_scheduler = get_optimizer(args.OptimConfig, self.net.parameters())
        self.criterion = get_loss(args.LossConfig, device=self.device)
        self.metrics = get_metrics(args.MetricConfig)
        # create log directories
        self.log_paths = create_log_directory("logs", args.ExpConfig.exp_name)
        print(self.log_paths)
        self.writer_train = SummaryWriter(log_dir=f"{self.log_paths['path_run']}/train")
        self.writer_val = SummaryWriter(log_dir=f"{self.log_paths['path_run']}/val")

    def train(self, dataset: str, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # set up logger for training
        logger = create_logger(log_dir=self.log_paths["path_log"])
        logger.info(self.args)

        # init best score
        dice_best = 0.
        loss_best = 1e4

        for epoch in range(self.args.ExpConfig.epochs):
            if epoch and epoch < 5:
                self.validate(epoch, dataset, val_loader, logger, do_writer=False)

            time_start = time.time()
            loss = self.train_epoch(epoch, train_loader)
            logger.info(f"Train loss: {loss} || @ epoch {epoch}.")
            time_end = time.time()
            print(f"Training time: {time_end - time_start}")
            self.writer_train.add_scalar("Loss", loss, epoch)

            if epoch and epoch % self.args.ExpConfig.val_freq == 0 or epoch == self.args.ExpConfig.epochs - 1:
                loss_val, dice_val, iou_val = self.validate(epoch, dataset, val_loader, logger, do_writer=False)
                self.writer_val.add_scalar("Loss", loss_val, epoch)
                self.writer_val.add_scalar("Dice", dice_val, epoch)
                self.writer_val.add_scalar("IOU", iou_val, epoch)

                if dice_val > dice_best:
                    loss_best = loss_val
                    dice_best = dice_val
                    is_best = True
                else:
                    is_best = False

                self.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model": self.args.NetworkConfig.net,
                        "state_dict": self.net.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "loss_best": loss_best,
                        "dice_best": dice_best,
                        "log_paths": self.log_paths,
                    },
                    is_best=is_best,
                    filename=f"checkpoint_epoch_{epoch}.pth"
                )

        self.writer_train.close()
        self.writer_val.close()

    def validate(self, epoch: int, dataset: str, val_loader: DataLoader, logger: Logger, do_writer: bool = False) -> Tuple[torch.Tensor, ...]:
        self.net.eval()
        loss, dice_scores, iou_scores = self.validate_epoch(epoch, val_loader)
        if dataset == "refuge":
            logger.info(f"Loss: {loss}, IOU_CUP: {iou_scores[0]}, IOU_DISC: {iou_scores[1]}, DICE_CUP: {dice_scores[0]}, DICE_DISC: {dice_scores[1]} || @ epoch {epoch}.")
        else:
            logger.info(f"Loss: {loss}, DICE: {dice_scores}, IOU: {iou_scores} || @ epoch {epoch}.")
        dice_mean = dice_scores.mean()
        iou_mean = iou_scores.mean()
        if do_writer:
            self.writer_val.add_scalar("Loss", loss.item(), epoch)
            self.writer_val.add_scalar("Dice", dice_mean.item(), epoch)
            self.writer_val.add_scalar("IOU", iou_mean.item(), epoch)
            self.writer_val.close()
        return loss.item(), dice_mean.item(), iou_mean.item()

    def train_epoch(self, epoch: int, train_loader: DataLoader) -> torch.Tensor:
        self.net.train()
        self.optimizer.zero_grad()
        loss_epoch = 0.
        num_train = len(train_loader)
        with tqdm(total=num_train, desc=f'Epoch {epoch}', unit="image", leave=False) as pbar:
            for i, data in enumerate(train_loader):
                images = data["image"].to(dtype=torch.float32, device=self.device)
                labels = data["label"].to(dtype=torch.float32, device=self.device)
                # check generated points for prompting
                if "point_coord" not in data:
                    from dataset import generate_click_prompt
                    images, point_coords, labels = generate_click_prompt(images, labels)
                else:
                    point_coords = data["point_coord"]
                    point_labels = data["point_label"]

                if point_labels.clone().flatten()[0] != -1:
                    point_coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
                    point_labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
                    if len(point_labels.shape) == 1:
                        # only one point prompt
                        point_coords_torch, point_labels_torch = point_coords_torch[None, :, :], point_labels_torch[None, :]
                    point_coords = (point_coords_torch, point_labels_torch)

                if self.args.NetworkConfig.block == "adapt":
                    for n, value in self.net.image_encoder.named_parameters():
                        if "adapt" not in n:
                            value.requires_grad = False
                        else:
                            value.requires_grad = True
                elif self.args.NetworkConfig.block == "lora" or self.args.NetworkConfig.block == "adalora":
                    from graph.network.layer.lora import mark_only_lora_as_trainable
                    mark_only_lora_as_trainable(self.net.image_encoder)
                    if self.args.NetworkConfig.block == "adalora":
                        from graph.network.layer.lora import RankAllocator
                        # initialize the RankAllocator
                        rank_allocator = RankAllocator(
                            self.net.image_encoder,
                            lora_r=4,
                            target_rank=8,
                            init_warmup=500,
                            final_warmup=1500,
                            mask_interval=10,
                            total_step=3000,
                            beta1=0.85,
                            beta2=0.85,
                        )
                else:
                    for n, value in self.net.image_encoder.named_parameters():
                        value.requires_grad = True

                embed_image =self.net.image_encoder(images)

                if self.args.NetworkConfig.net in ["sam", "mobile_sam_v2"]:
                    with torch.no_grad():
                        embed_sparse, embed_dense = self.net.prompt_encoder(
                            points=point_coords,
                            boxes=None,
                            masks=None
                        )

                    preds, _ = self.net.mask_decoder(
                        image_embeddings=embed_image,
                        image_pe=self.net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=embed_sparse,
                        dense_prompt_embeddings=embed_dense,
                        multimask_output=(self.args.DataConfig.multimask_output > 1),
                    )

                # resize to the ordered output size
                preds = fn.interpolate(preds, size=(self.args.DataConfig.output_size, self.args.DataConfig.output_size))
                loss = self.criterion(preds, labels)

                pbar.set_postfix(**{"loss (batch)": loss.item()})
                loss_epoch += loss.item()

                if self.args.NetworkConfig.block == "adalora":
                    from graph.network.layer.lora import compute_orth_regu
                    (loss + compute_orth_regu(self.net, regu_weight=0.1)).backward()
                    self.optimizer.step()
                    rank_allocator.update_and_mask(self.net, i)
                else:
                    loss.backward()
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # visual image
                # if self.args.ExpConfig.vis_freq and i % self.args.ExpConfig.vis_freq == 0:
                #     visualize_images(
                #         images,
                #         labels,
                #         preds,
                #         save_path=Path(self.log_paths["path_sample"]),
                #         filename=f"{name}_epoch_{epoch}",
                #         prefix="train",
                #         reverse=False
                #     )

                pbar.update()
        return loss.item()
        # return loss_epoch / num_train

    def validate_epoch(self, epoch: int, val_loader: DataLoader, ) -> Tuple[torch.Tensor, ...]:
        self.net.eval()
        num_val = len(val_loader)
        loss_total = 0.
        dice_total = torch.zeros(size=self.args.DataConfig.multimask_output, dtype=torch.float32)
        iou_total = torch.zeros(size=self.args.DataConfig.multimask_output, dtype=torch.float32)
        with tqdm(total=num_val, desc="Validation", unit="batch", leave=False) as pbar:
            for i, data in enumerate(val_loader):
                images = data["image"].to(dtype=torch.float32, device=self.device)
                labels = data["label"].to(dtype=torch.float32, device=self.device)
                # check generated points for prompting
                if ("point_coord" not in data) or self.args.DataConfig.is_3d:
                    from dataset import generate_click_prompt
                    images, point_coords, labels = generate_click_prompt(images, labels)
                else:
                    point_coords = data["point_coord"]
                    point_labels = data["point_label"]
                name = data["filename"][0]

                buoy = 0
                if self.args.DataConfig.eval_chunk:
                    eval_chunk = int(self.args.DataConfig.eval_chunk)
                else:
                    eval_chunk = int(images.size(-1))

                while (buoy + eval_chunk) <= images.size(-1):
                    images_ = images[..., buoy: buoy + eval_chunk]
                    labels_ = labels[..., buoy: buoy + eval_chunk]
                    buoy += eval_chunk

                    i += 1
                    if point_labels.clone().flatten()[0] != -1:
                        point_coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
                        point_labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
                        if len(point_labels.shape) == 1:
                            # only one point prompt
                            point_coords_torch, point_labels_torch = point_coords_torch[None, :, :], point_labels_torch[None, :]
                        point_coords = (point_coords_torch, point_labels_torch)

                    with torch.no_grad():
                        embed_image = self.net.image_encoder(images_)
                        if self.args.NetworkConfig.net in ["sam", "mobile_sam_v2"]:
                            embed_sparse, embed_dense = self.net.prompt_encoder(
                                points=point_coords,
                                boxes=None,
                                masks=None
                            )

                            preds_, _ = self.net.mask_decoder(
                                image_embeddings=embed_image,
                                image_pe=self.net.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=embed_sparse,
                                dense_prompt_embeddings=embed_dense,
                                multimask_output=(self.args.DataConfig.multimask_output > 1),
                            )

                        # resize to the ordered output size
                        preds_ = fn.interpolate(preds_, size=(self.args.DataConfig.output_size, self.args.DataConfig.output_size))
                        loss_total += self.criterion(preds_, labels_)

                        # compute metrics
                        dice_scores, iou_scores, entropies, maes = self.evaluate_results(preds_, labels_)
                        dice_total += dice_scores
                        iou_total += iou_scores

                pbar.update()
            # visual image
            if self.args.ExpConfig.vis_freq and i % self.args.ExpConfig.vis_freq == 0:
                visualize_images(
                    images_,
                    labels_,
                    preds_,
                    entropies,
                    maes,
                    save_path=Path(self.log_paths["path_sample"]),
                    filename=f"{name}_epoch_{epoch}",
                    prefix="val",
                    writer=self.writer_val,
                    reverse=False
                )

        if self.args.DataConfig.eval_chunk:
            num_val *= (images.size(-1) // eval_chunk)

        return loss_total / num_val, dice_total / num_val, iou_total / num_val

    def evaluate_results(self, mask_prd: torch.Tensor, mask_tgt: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B, C, H, W = mask_prd.shape
        dice_scores = torch.zeros(size=C, dtype=torch.float32)
        iou_scores = torch.zeros(size=C, dtype=torch.float32)
        for threshold in self.args.MetricConfig.thresholds:
            mask_prd_p = (mask_prd > threshold).float()
            mask_tgt_p = (mask_tgt > threshold).float()
            dice_scores += self.metrics["dice_coeff"](mask_prd_p, mask_tgt_p)
            iou_scores += self.metrics["iou"](mask_prd_p, mask_tgt_p)
        dice_scores /= len(self.args.MetricConfig.thresholds)
        iou_scores /= len(self.args.MetricConfig.thresholds)

        entropies = self.metrics["entropy"](mask_prd)
        maes = self.metrics["mae"](mask_prd, mask_tgt)

        return iou_scores, dice_scores, entropies, maes

    def save_checkpoint(self, state: Dict, is_best: bool, filename: str = "checkpoint.pth") -> None:
        torch.save(state, (Path(self.log_paths["path_ckpt"]) / filename))
        if is_best:
            torch.save(state, (Path(self.log_paths["path_ckpt"]) / "checkpoint_best.pth"))

    def test(self, **kwargs) -> None:
         pass