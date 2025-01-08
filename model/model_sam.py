from typing import Tuple, Dict
from abc import ABC
import time
import shutil
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
        if self.args.ExpConfig.mode == "train":
            self.optimizer, self.lr_scheduler = get_optimizer(args.OptimConfig, self.net.parameters())
        else:
            self.optimizer, self.lr_scheduler = None, None
        self.criterion = get_loss(args.LossConfig, device=self.device)
        self.metrics = get_metrics(args.MetricConfig)
        # create log directories
        self.log_paths = create_log_directory("logs", args.ExpConfig.exp_name)
        if self.args.ExpConfig.mode == "train":
            self.writer_train = SummaryWriter(log_dir=f"{self.log_paths['path_run']}/train")
            self.writer_val = SummaryWriter(log_dir=f"{self.log_paths['path_run']}/val")
        elif self.args.ExpConfig.mode in ["val", "test"]:
            self.writer_val = SummaryWriter(log_dir=f"{self.log_paths['path_run']}/test")
        # load checkpoint weight for resume training/testing
        if self.args.ExpConfig.ckpt is not None:
            print(f"=> Loading checkpoint '{self.args.ExpConfig.ckpt}'")
            if self.args.ExpConfig.mode == "train":
                checkpoint = torch.load(self.args.ExpConfig.ckpt, map_location=self.device, weights_only=False)
                self.start_epoch = checkpoint["epoch"]
                self.loss_best = checkpoint["loss_best"]
                self.dice_best = checkpoint["dice_best"]
            else:
                checkpoint = torch.load(self.args.ExpConfig.ckpt, map_location=self.device, weights_only=True)
            self.net.load_state_dict(checkpoint["state_dict"], strict=False)
            if not (Path(self.log_paths["path_ckpt"]) / "checkpoint_best.pth").exists():
                shutil.copyfile(self.args.ExpConfig.ckpt, f"{self.log_paths['path_ckpt']}/checkpoint_best.pth")
        else:
            self.start_epoch = 0
            self.loss_best = 1e-4
            self.dice_best = 0.


    def train(self, dataset: str, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # set up logger for training
        logger = create_logger(log_dir=self.log_paths["path_log"])
        logger.info(self.args)

        for epoch in range(self.start_epoch, self.args.ExpConfig.epochs):
            if epoch < 5:
                loss, dice_scores, iou_scores, corr_scores = self.validate_epoch(epoch, val_loader, is_vis=False)
                if dataset == "refuge":
                    logger.info(
                        f"Val Loss: {loss}, IOU_CUP: {iou_scores[0]}, IOU_DISC: {iou_scores[1]}, DICE_CUP: {dice_scores[0]}, DICE_DISC: {dice_scores[1]}, CORR_DISC: {corr_scores[0]}, CORR_CUP: {corr_scores[1]} || @ epoch {epoch}."
                    )
                else:
                    logger.info(
                        f"Val Loss: {loss}, DICE: {dice_scores}, IOU: {iou_scores}, CORR: {corr_scores} || @ epoch {epoch}.")

            time_start = time.time()
            loss = self.train_epoch(epoch, train_loader)
            logger.info(f"Train loss: {loss} || @ epoch {epoch}.")
            time_end = time.time()
            print(f"Training time: {time_end - time_start}")
            self.writer_train.add_scalar("Loss", loss, epoch)

            if self.args.ExpConfig.val_freq and (epoch % self.args.ExpConfig.val_freq == 0 or epoch == self.args.ExpConfig.epochs - 1):
                loss, dice_scores, iou_scores, corr_scores = self.validate_epoch(epoch, val_loader, is_vis=True)
                if dataset == "refuge":
                    logger.info(
                        f"Val Loss: {loss}, IOU_CUP: {iou_scores[0]}, IOU_DISC: {iou_scores[1]}, DICE_CUP: {dice_scores[0]}, DICE_DISC: {dice_scores[1]}, CORR_DISC: {corr_scores[0]}, CORR_CUP: {corr_scores[1]} || @ epoch {epoch}."
                    )
                else:
                    logger.info(
                        f"Val Loss: {loss}, DICE: {dice_scores}, IOU: {iou_scores}, CORR: {corr_scores} || @ epoch {epoch}.")

                loss_val = loss.item()
                dice_val = dice_scores.mean().item()
                iou_val = iou_scores.mean().item()
                corr_val = corr_scores.mean().item()

                self.writer_val.add_scalar("Loss", loss_val, epoch)
                self.writer_val.add_scalar("Dice", dice_val, epoch)
                self.writer_val.add_scalar("IOU", iou_val, epoch)
                self.writer_val.add_scalar("Corr", corr_val, epoch)

                if dice_val > self.dice_best:
                    self.loss_best = loss_val
                    self.dice_best = dice_val
                    is_best = True
                else:
                    is_best = False

                self.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model": self.args.NetworkConfig.net,
                        "state_dict": self.net.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "loss_best": self.loss_best,
                        "dice_best": self.dice_best,
                        "log_paths": self.log_paths,
                    },
                    is_best=is_best,
                    filename=f"checkpoint_epoch_{epoch}.pth"
                )

        self.writer_train.close()
        self.writer_val.close()

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

                pbar.update()
        return loss_epoch / num_train

    def validate_epoch(self, epoch: int, val_loader: DataLoader, is_vis: bool = False) -> Tuple[torch.Tensor, ...]:
        self.net.eval()
        num_val = len(val_loader)
        loss_total = 0.
        dice_total = torch.zeros(self.args.DataConfig.multimask_output, dtype=torch.float32)
        iou_total = torch.zeros(self.args.DataConfig.multimask_output, dtype=torch.float32)
        corr_total = torch.zeros(self.args.DataConfig.multimask_output, dtype=torch.float32)
        with tqdm(total=num_val, desc="Validation", unit="batch", leave=False) as pbar:
            for i, data in enumerate(val_loader):
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

                with torch.no_grad():
                    embed_image = self.net.image_encoder(images)
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
                    preds = fn.interpolate(preds_, size=(self.args.DataConfig.output_size, self.args.DataConfig.output_size))
                    loss_total += self.criterion(preds, labels)

                    # compute metrics
                    dice_scores, iou_scores, entropies, maes, corr_scores = self.evaluate_results(preds, labels)
                    dice_total += dice_scores
                    iou_total += iou_scores
                    corr_total += corr_scores

                # visual image
                if is_vis and self.args.ExpConfig.vis_freq and i % self.args.ExpConfig.vis_freq == 0:
                    if self.args.DataConfig.batch_size == 1:
                        name = f"{data['filename'][0]}_epoch{epoch}"
                    else:
                        name = f"batch{i}_epoch{epoch}"

                    visualize_images(
                        images.detach().cpu(),
                        labels.detach().cpu(),
                        preds.detach().cpu(),
                        entropies.detach().cpu(),
                        maes.detach().cpu(),
                        save_path=Path(self.log_paths["path_sample"]),
                        filename=name,
                        prefix="val",
                        writer=self.writer_val,
                        reverse=False,
                        num_rows=self.args.DataConfig.batch_size
                    )

                pbar.update()

        return loss_total / num_val, dice_total / num_val, iou_total / num_val, corr_scores / num_val

    def evaluate_results(self, mask_prd: torch.Tensor, mask_tgt: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B, C, H, W = mask_prd.shape
        dice_scores = torch.zeros(C, dtype=torch.float32)
        iou_scores = torch.zeros(C, dtype=torch.float32)
        for threshold in self.args.MetricConfig.thresholds:
            mask_prd_p = (mask_prd > threshold).float()
            mask_tgt_p = (mask_tgt > threshold).float()
            dice_scores += self.metrics["dice_coeff"](mask_prd_p, mask_tgt_p)
            iou_scores += self.metrics["iou"](mask_prd_p, mask_tgt_p)
        dice_scores /= len(self.args.MetricConfig.thresholds)
        iou_scores /= len(self.args.MetricConfig.thresholds)

        entropies = self.metrics["entropy"](mask_prd)
        maes = self.metrics["mae"](mask_prd, mask_tgt)
        corr_scores = self.metrics["corr_coeff"](entropies, maes)

        return iou_scores, dice_scores, entropies, maes, corr_scores

    def save_checkpoint(self, state: Dict, is_best: bool, filename: str = "checkpoint.pth") -> None:
        torch.save(state, (Path(self.log_paths["path_ckpt"]) / filename))
        if is_best:
            torch.save(state, (Path(self.log_paths["path_ckpt"]) / "checkpoint_best.pth"))

    def test(self, dataset: str, test_loader: DataLoader) -> None:
        # set up logger for training
        logger = create_logger(log_dir=self.log_paths["path_log"])
        logger.info(self.args)
        self.net.eval()
        num_test = len(test_loader)
        loss_total = 0.
        dice_total = torch.zeros(self.args.DataConfig.multimask_output, dtype=torch.float32)
        iou_total = torch.zeros(self.args.DataConfig.multimask_output, dtype=torch.float32)
        corr_total = torch.zeros(self.args.DataConfig.multimask_output, dtype=torch.float32)
        with tqdm(total=num_test, desc="Testing", unit="batch", leave=False) as pbar:
            for i, data in enumerate(test_loader):
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

                with torch.no_grad():
                    embed_image = self.net.image_encoder(images)
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
                    preds = fn.interpolate(preds_,
                                           size=(self.args.DataConfig.output_size, self.args.DataConfig.output_size))
                    loss_total += self.criterion(preds, labels)

                    # compute metrics
                    dice_scores, iou_scores, entropies, maes, corr_scores = self.evaluate_results(preds, labels)
                    dice_total += dice_scores
                    iou_total += iou_scores
                    corr_total += corr_scores

                    # visual image
                    if self.args.DataConfig.batch_size == 1:
                        name = data["filename"][0]
                    else:
                        name = f"batch_{i}"

                    visualize_images(
                        images.detach().cpu(),
                        labels.detach().cpu(),
                        preds.detach().cpu(),
                        entropies.detach().cpu(),
                        maes.detach().cpu(),
                        save_path=Path(self.log_paths["path_sample"]),
                        filename=name,
                        prefix="test",
                        writer=self.writer_val,
                        reverse=False,
                        num_rows=self.args.DataConfig.batch_size,
                    )

                pbar.update()

        loss_total /= num_test
        dice_total /= num_test
        iou_total /= num_test
        corr_total /= num_test

        if dataset == "refuge":
            logger.info(
                f"Testing Loss: {loss_total}, IOU_CUP: {iou_total[0]}, IOU_DISC: {iou_total[1]}, DICE_CUP: {dice_total[0]}, DICE_DISC: {dice_total[1]}, CORR_DISC: {corr_scores[0]}, CORR_CUP: {corr_scores[1]}."
            )
        else:
            logger.info(
                f"Testing Loss: {loss_total}, DICE: {dice_total}, IOU: {iou_total}, CORR: {corr_total}.")

        loss_test = loss_total.item()
        dice_test = dice_scores.mean().item()
        iou_test = iou_scores.mean().item()
        corr_test = corr_scores.mean().item()

        self.writer_val.add_scalar("Loss", loss_test, 1)
        self.writer_val.add_scalar("Dice", dice_test, 1)
        self.writer_val.add_scalar("IOU", iou_test, 1)
        self.writer_val.add_scalar("Corr", corr_test, 1)
