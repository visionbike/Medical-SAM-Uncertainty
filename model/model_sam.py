from typing import Tuple, Dict
from abc import ABC
import time
from logging import Logger
from argparse import Namespace
from pathlib import Path
import torch
from tqdm import tqdm
from einops import rearrange
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as vtf
import torch.nn.functional as fn
from utils.log import create_logger, create_log_directory
from .model_base import ModelBase
from graph.network import get_network
from graph.optimizer import get_optimizer
from graph.loss import get_loss
from graph.metric import get_metrics
from utils import get_device, visualize_images


class SAMModel(ModelBase, ABC):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        self.device, self.device_ids = get_device(args.ExpConfig.use_gpu, args.ExpConfig.gpu_device, args.ExpConfig.distributed)
        self.net = get_network(args.NetworkConfig, use_gpu=args.ExpConfig.use_gpu, device=self.device, distributed=self.device_ids)
        self.optimizer, self.lr_scheduler = get_optimizer(args.OptimConfig, self.net.parameters())
        self.criterion = get_loss(args.LossConfig, device=self.device)
        self.metrics = get_metrics(args.MetricConfig)
        # create log directories
        self.log_paths = create_log_directory("logs", args.ExpConfig.exp_name)
        self.writer = SummaryWriter(log_dir=self.log_paths["path_run"])

    def train(self, dataset: str, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # set up logger for training
        logger = create_logger(log_dir=self.log_paths["path_log"])
        logger.info(self.args)

        # init best score
        dice_best = 0.
        loss_best = 1e4

        for epoch in range(self.args.ExpConfig.epochs):
            if epoch and epoch < 5:
                _ = self.validate(epoch, dataset, val_loader, logger)

            time_start = time.time()
            loss = self.train_epoch(epoch, train_loader)
            logger.info(f"Train loss: {loss.item()} || @ epoch {epoch}.")
            time_end = time.time()
            print(f"Training time: {time_end - time_start}")

            if epoch and epoch % self.args.ExpConfig.val_freq == 0 or epoch == self.args.ExpConfig.epochs - 1:
                loss, dice, iou = self.validate(epoch, dataset, val_loader, logger)

                if dice.item() > dice_best:
                    loss_best = loss.item()
                    dice_best = dice.item()
                    is_best = True
                else:
                    is_best = False

                self.save_checkpoint(
                    {
                        "epoch": epoch,
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

        self.writer.close()

    def validate(self, epoch: int, dataset: str, val_loader: DataLoader, logger: Logger) -> Tuple[torch.Tensor, ...]:
        loss, dices, ious = self.validate_epoch(epoch, val_loader)
        if dataset == "refuge":
            logger.info(
                f"Total score: {loss.item()}, DICE_CUP: {dices[0].item()}, DICE_DISC: {dices[1].item()} || @ epoch {epoch}.")
        else:
            logger.info(f"Total score: {loss.item()}, DICE: {dices.item()}, IOU: {ious.item()} || @ epoch {epoch}.")
        return loss, dices.mean(), ious.mean()

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
                if ("point_coord" not in data) or self.args.DataConfig.is_3d:
                    from dataset import generate_click_prompt
                    images, point_coords, labels = generate_click_prompt(images, labels)
                else:
                    point_coords = data["point_coord"]
                    point_labels = data["point_label"]
                name = data["filename"]

                if self.args.DataConfig.is_3d:
                    point_coords = rearrange(point_coords, "b n d -> (b d) n")
                    images = rearrange(images, "b c h w d -> (b d) c h w")
                    labels = rearrange(labels, "b c h w d -> (b d) c h w")
                    images = images.repeat(1, 3, 1, 1)
                    point_labels = torch.ones(images.size(0))
                    images = vtf.Resize((self.args.DataConfig.image_size, self.args.DataConfig.image_size))(images)
                    labels = vtf.Resize((self.args.DataConfig.out_size, self.args.DataConfig.out_size))(labels)

                i += 1
                if point_labels.clone().flatten()[0] != -1:
                    coords_point = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
                    coords_label = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
                    if len(point_labels.shape) == 1:
                        # only one point prompt
                        coords_point, coords_label = coords_point[None, :, :], coords_label[None, :]
                    point_coords = (coords_point, coords_label)

                if self.args.NetworkConfig.block == "adapt":
                    for n, value in self.net.image_encoder.named_parameters():
                        if "Adapter" not in n:
                            value.requires_grad = False
                        else:
                            value.requires_grad = True
                elif self.args.NetworkConfig.block == "lora" or self.args.NetworkConfig.block == "adalora":
                    from graph.network.layer.lora import mark_only_lora_as_trainable
                    mark_only_lora_as_trainable(self.net.image_encoder)
                    if self.args.NetworkConfig.block == "adalora":
                        from graph.network.layer.lora import RankAllocator
                        # Initialize the RankAllocator
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
                        embed_sparse, embed_dense = self.net.promt_encoder(
                            points=point_coords,
                            boxes=None,
                            masks=None
                        )

                    preds, _ = self.net.mask_decoder(
                        image_embeddings=embed_image,
                        image_pe=self.net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=embed_sparse,
                        dense_prompt_embeddings=embed_dense,
                        multimask_output=(self.args.multimask_output > 1),
                    )

                # resize to the ordered output size
                preds = fn.interpolate(preds, size=(self.args.DataConfig.out_size, self.args.Config.out_size))

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

                # visual image
                if i % self.args.ExpConfig.vis == 0:
                    visualize_images(
                        images,
                        preds,
                        labels,
                        save_path=(Path(self.log_paths["path_sample"]) / f"train_{name}_epoch_{epoch}.jpg").__str__(),
                        reverse=False
                    )

                pbar.update()
        return loss_epoch / num_train

    def validate_epoch(self, epoch: int, val_loader: DataLoader, ) -> Tuple[torch.Tensor, ...]:
        self.net.eval()
        num_val = len(val_loader)
        loss_total = 0.
        dice_total = torch.zeros([self.args.DataConfig.multimask_output], dtype=torch.float32)
        iou_total = torch.zeros([self.args.DataConfig.multimask_output], dtype=torch.float32)
        with tqdm(total=num_val, desc="Validation", unit="batch", leave=False) as pbar:
            for i, data in enumerate(val_loader):
                images = data["image"].to(dtype=torch.float32, device=self.device)
                labels = data["label"].to(dtype=torch.float32, device=self.device)
                # check generated points for prompting
                if ("point" not in data) or self.args.DataConfig.is_3d:
                    from dataset import generate_click_prompt
                    images, points, labels = generate_click_prompt(images, labels)
                else:
                    points = data["point"]
                    point_labels = data["point_label"]
                name = data["filename"]

                buoy = 0
                if self.args.DataConfig.eval_chunk:
                    eval_chunk = int(self.args.DataConfig.eval_chunk)
                else:
                    eval_chunk = int(images.size(-1))

                while (buoy + eval_chunk) <= images.size(-1):
                    # prepare point prompting
                    if self.args.DataConfig.is_3d:
                        points_ = points[:, :, buoy: buoy + eval_chunk]
                    else:
                        points_ = points

                    images_ = images[..., buoy: buoy + eval_chunk]
                    labels_ = labels[..., buoy:buoy + eval_chunk]
                    buoy += eval_chunk

                    if self.args.DataConfig.is_3d:
                        points_ = rearrange(points_, "b n d -> (b d) n")
                        images_ = rearrange(images_, "b c h w d -> (b d) c h w")
                        labels_ = rearrange(labels_, "b c h w d -> (b d) c h w")
                        images_ = images_.repeat(1, 3, 1, 1)
                        point_labels = torch.ones(images_.size(0))
                        images_ = vtf.Resize((self.args.DataConfig.image_size, self.args.DataConfig.image_size))(images_)
                        labels_ = vtf.Resize((self.args.DataConfig.out_size, self.args.DataConfig.out_size))(labels_)

                    i += 1
                    if point_labels.clone().flatten()[0] != -1:
                        coords_ = torch.as_tensor(points_, dtype=torch.float, device=self.device)
                        coords_label_ = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
                        if len(point_labels.shape) == 1:
                            # only one point prompt
                            coords_, coords_label__ = coords_[None, :, :], coords_label_[None, :]
                        points_ = (coords_, coords_label_)

                    with torch.no_grad():
                        preds_, _ = self.net(
                            images_,
                            points=points_,
                            boxes=None,
                            masks=None,
                            multimask_output=(self.args.DataConfig.multimask_output > 1)
                        )

                        # resize to the ordered output size
                        preds_ = fn.interpolate(preds_, size=(self.args.DataConfig.out_size, self.args.Config.out_size))
                        loss_total += self.criterion(preds_, labels_)

                        # compute metrics
                        dices, ious, entropies, maes = self.evaluate_results(preds_, labels_)
                        dice_total += dices
                        iou_total += ious

                        # visual image
                        if i % self.args.ExpConfig.vis == 0:
                            visualize_images(
                                images_,
                                preds_,
                                labels_,
                                entropies,
                                maes,
                                (Path(self.log_paths["path_sample"]) / f"test_{name}_epoch_{epoch}.jpg").__str__(),
                                reverse=False
                            )

                pbar.update()

        if self.args.DataConfig.eval_chunk:
            num_val = num_val * (images.size(-1) // eval_chunk)

        # get average values through batches
        loss_total /= num_val
        dice_total /= num_val
        iou_total /= num_val

        return loss_total, dice_total, iou_total


    def evaluate_results(self, mask_prd: torch.Tensor, mask_tgt: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B, C, H, W = mask_prd.shape

        dices = torch.zeros([C], dtype=torch.float32)
        ious = torch.zeros([C], dtype=torch.float32)
        for threshold in self.args.MetricConfig.thresholds:
            mask_prd_p = (mask_prd > threshold).float()
            mask_tgt_p = (mask_tgt > threshold).float()
            dices += self.metrics["dice_coeff"](mask_prd_p, mask_tgt_p)
            ious += self.metrics["iou"](mask_prd_p, mask_tgt_p)
        dices /= len(self.args.MetricConfig.thresholds)
        ious /= len(self.args.MetricConfig.thresholds)

        entropies = self.metrics["entropy"](mask_prd)
        maes = self.metrics["mae"](mask_prd, mask_tgt)

        return ious, dices, entropies, maes

    def save_checkpoint(self, state: Dict, is_best: bool, filename: str = "checkpoint.pth") -> None:
        torch.save(state, (Path(self.log_paths["path_ckpt"]) / filename).__str__())
        if is_best:
            torch.save(state, (Path(self.log_paths["path_ckpt"]) / "checkpoint_best.pth").__str__())
