import pandas as pd
import numpy as np
import torch

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
import torch.nn.functional as F

from torch.optim.lr_scheduler import ExponentialLR
import torchmetrics

import wandb


class LitAbdPainPredictionMLP(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        n_classes,
        #                  learning_rate,
        config=None,
        loss_fn=F.binary_cross_entropy_with_logits,
        layer_size=256,
        dropout=0.5,
    ):
        """Basic MLP to Training Abdominal Pain DDX Prediction

        config is a dictionary from W&B most likely
        """
        super().__init__()

        self.fc1 = nn.Linear(input_dim, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.fc4 = nn.Linear(layer_size, n_classes)

        self.bn1 = nn.BatchNorm1d(num_features=layer_size)
        self.bn2 = nn.BatchNorm1d(num_features=layer_size)
        self.bn3 = nn.BatchNorm1d(num_features=layer_size)

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

        self.config = config
        #         self.lr = learning_rate

        self.loss = loss_fn

    def forward(self, x):
        if self.dropout_p > 0:
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            x = self.dropout(F.relu(self.fc3(x)))
        else:
            x = self.bn1(F.relu(self.fc1(x)))
            x = self.bn2(F.relu(self.fc2(x)))
            x = self.bn3(F.relu(self.fc3(x)))

        logits = self.fc4(x)

        return logits

    def training_step(self, batch, batch_idx):
        data, targets = batch
        out = self(data)
        loss = self.loss(out, targets)
        preds = torch.sigmoid(out).data > 0.5
        preds = preds.to(torch.float32)

        wandb.log(
            {
                "train_loss": loss,
                "train_subset_acc": torchmetrics.functional.accuracy(
                    preds, targets.long(), subset_accuracy=True
                ),
                # "hamming_dist":1-torchmetrics.functional.hamming_distance(preds, targets.long()),
                "train_precision/macro": torchmetrics.functional.precision(
                    preds, targets.long()
                ),
                "train_recall/macro": torchmetrics.functional.recall(
                    preds, targets.long()
                ),
                "train_precision@5": torchmetrics.functional.retrieval_precision(
                    out, targets.long(), k=5
                ),
                "train_recall@5": torchmetrics.functional.retrieval_recall(
                    out, targets.long(), k=5
                ),
            }
        )

        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        # TODO: ValueError: You can not use the `top_k` parameter to calculate accuracy for multi-label inputs.
        #     avg_topk_acc = []
        out = self(data)
        loss = self.loss(out, targets)
        #             self.log({"train_loss": loss})
        preds = torch.sigmoid(out).data > 0.5
        preds = preds.to(torch.float32)

        # avg_topk_acc.append(torchmetrics.functional.accuracy(preds, targets.long(), top_k=5))

        # TODO: create a PR plot for the multilabel case
        #             self.log({"pr" : wandb.plot.pr_curve(targets.cpu(), preds.data.cpu(),
        #                      labels=label_freqs.index.tolist()
        #                                                  )})

        self.log("validation_loss", loss)
        wandb.log(
            {
                "validation_acc": torchmetrics.functional.accuracy(
                    preds, targets.long(), subset_accuracy=True
                )
            }
        )
        #         wandb.log({"top5_validation_acc":sum(avg_topk_acc)/len(avg_topk_acc)})

        return loss

    def configure_optimizers(self):
        #         opt = torch.optim.SGD(self.parameters(),
        #                                     lr=self.lr,
        #                                     weight_decay=self.config["lr_weight_decay"],
        #                                     momentum=0.9)

        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["lr_weight_decay"],
        )

        if self.config["lr_scheduler"]:
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": ExponentialLR(opt, 0.99),
                    "interval": "epoch",  # called after each training step
                },
            }
        else:
            return opt
