from abc import ABC
from collections import defaultdict
from typing import Any, Dict
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn
from flatdict import FlatDict
from lightning import LightningModule

from src.models.components.preprocess import domain_stack_extract
from src.models.components.layer.loss import GMMLoss

LOG_CONFIG = {
    "prog_bar": True,
    "logger": True,
    "on_step": True,
    "on_epoch": True,
}


def flatten_dict(data_dict, prefix=""):
    data_dict = dict(FlatDict(data_dict, delimiter="/"))
    out_dict = {f"{prefix}{key}": value for key, value in data_dict.items()}
    return out_dict


class AdModule(LightningModule):
    def __init__(
        self,
        net,
        losses=None,
        losses_weights=None,
        metrics=None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        scheduler_conf: Dict = None,
    ):
        super().__init__()
        self.net = net
        self.losses = self.post_check(losses)
        self.losses_weights = losses_weights
        self.metrics = self.post_check(metrics)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_conf = scheduler_conf

    def post_check(self, post_process):
        if post_process is None:
            return None
        out_post = defaultdict(dict)
        for stage, stage_post in post_process.items():
            for post_name, post_func in stage_post.items():
                out_post[stage][post_name] = post_func
            out_post[stage] = nn.ModuleDict(out_post[stage])
        out_post = nn.ModuleDict(out_post)
        return out_post

    def post(self, post, pred_dict, target_dict):
        post_dict = defaultdict(dict)
        for stage, stage_post in post.items():
            for post_name, post_func in stage_post.items():
                post_dict[stage][post_name] = post_func(
                    pred_dict[stage],
                    target_dict[stage],
                )
        return post_dict

    def loss_merge(self, loss_dict, weight_dict):
        loss = torch.tensor(0.0, device=self.device)
        for stage, stage_loss in loss_dict.items():
            for loss_name, loss_value in stage_loss.items():
                loss += weight_dict[stage][loss_name] * loss_value
        return loss

    def forward(self, x):
        return self.net(x)

    def model_step(self, batch, batch_idx):
        feature_dict = domain_stack_extract(batch, "features")
        pred_dict = self.net(feature_dict["audios"])
        label_dict = domain_stack_extract(batch, "labels")
        loss_dict = self.post(self.losses, pred_dict, label_dict)
        loss_dict["total"] = self.loss_merge(loss_dict, self.losses_weights)
        metric_dict = self.post(self.metrics, pred_dict, label_dict)
        return feature_dict, label_dict, pred_dict, loss_dict, metric_dict

    def training_step(self, batch, batch_idx):
        feature_dict, label_dict, pred_dict, loss_dict, metric_dict = self.model_step(
            batch, batch_idx
        )
        self.log_dict(flatten_dict(loss_dict, "train/loss/"), **LOG_CONFIG)
        self.log_dict(flatten_dict(metric_dict, "train/metric/"), **LOG_CONFIG)
        return loss_dict["total"]

    def validation_step(self, batch, batch_idx):
        feature_dict, label_dict, pred_dict, loss_dict, metric_dict = self.model_step(
            batch, batch_idx
        )
        self.log_dict(flatten_dict(loss_dict, "val/loss/"), **LOG_CONFIG)
        self.log_dict(flatten_dict(metric_dict, "val/metric/"), **LOG_CONFIG)

    def test_step(self, batch, batch_idx):
        feature_dict, label_dict, pred_dict, loss_dict, metric_dict = self.model_step(
            batch, batch_idx
        )
        self.log_dict(flatten_dict(loss_dict, "test/loss"), **LOG_CONFIG)
        self.log_dict(flatten_dict(metric_dict, "test/metric"), **LOG_CONFIG)

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **self.scheduler_conf,
                },
            }
        return {"optimizer": optimizer}


class GMMModule(AdModule):
    def __init__(
        self,
        net,
        losses=None,
        losses_weights=None,
        metrics=None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        scheduler_conf: Dict = None,
        gmm_loss: torch.nn.Module = GMMLoss(0.7, 0.3),
    ):
        super().__init__(
            net=net,
            losses=losses,
            losses_weights=losses_weights,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_conf=scheduler_conf,
        )
        self.gmm_loss = gmm_loss

    def model_step(self, batch, batch_idx):
        (
            feature_dict,
            label_dict,
            pred_dict,
            loss_dict,
            metric_dict,
        ) = super().model_step(batch, batch_idx)
        gmm_loss = self.gmm_loss(pred_dict["gmm_output"], pred_dict["gmm_input"])
        loss_dict["gmm"] = gmm_loss
        loss_dict["total"] += gmm_loss * self.losses_weights["gmm"]
        return feature_dict, label_dict, pred_dict, loss_dict, metric_dict


if __name__ == "__main__":
    import hydra
    import rootutils
    from omegaconf import OmegaConf

    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    root = rootutils.find_root(search_from=__file__, indicator=".project-root")

    path = root / "configs" / "model" / "vit_classify.yaml"
    cfg = OmegaConf.load(path)
    model = hydra.utils.instantiate(cfg)
    print(model)
