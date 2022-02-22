import os
import sys
import string
import random
import time
import ast
import pickle as pkl
import csv
from argparse import ArgumentParser

import numpy as np
import pandas as pd

import sklearn

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import *

import wandb

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer, seed_everything

from pl_bolts.callbacks import TrainingDataMonitor

from utils.data_loader import *
from utils.simple_nn_utils import *
from utils.model_utils import *
from models.simple_nn_lightning import *

import yaml


def train():
    ################
    ### Settings ###
    ################
    seed_everything(314, workers=True)

    # data_fp = f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/unq_pt_enc_clean_multilabel_nomis_dvemb.pkl"
    data_fp = f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/unq_pt_enc_clean_multilabel_nomismatches_CTUS.pkl"
    # data_fp = f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/unq_pt_enc_clean_multilabel_nomismatches.pkl"
    config_fp = (
        f"/home/vs428/Documents/deep-ed-diags/configs/simple_nn_base_config.yaml"
    )

    fast_dev_run = False
    standardize = True
    # impacts how quickly we do earlystopping too by patience
    eval_freq = 2

    wandb.init(
        project="test-project",
        entity="decile",
        config=config_fp,
        allow_val_change=True,
        save_code=True,
    )

    WANDB_RUN_NAME = wandb.run.name

    # just fix some issue of conditional params
    if wandb.config["loss_fn"] == "focal":
        wandb.config.update({"class_weight_type": None}, allow_val_change=True)
    if fast_dev_run:
        wandb.config.update({"drop_sparse_cols": 0}, allow_val_change=True)
    if (
        wandb.config["class_weight_type"] == "None"
        or wandb.config["class_weight_type"] == None
    ):
        wandb.config.update({"class_weight_type": None}, allow_val_change=True)

    print(wandb.config, flush=True)

    torch.set_printoptions(profile="default", sci_mode=False, precision=3, linewidth=75)
    #################
    ### Data Load ###
    #################

    (
        X_train_input,
        y_train_input,
        X_test_input,
        y_test_input,
        class_weights,
    ) = load_multilabel_data(
        data_fp,
        wandb.config,
        fast_dev_run=fast_dev_run,
        run_name=WANDB_RUN_NAME,
        basic_cols=wandb.config["basic_col_subset"],
        test_size=0.2,
        standardize=standardize,
    )

    # we need to store this value for the NN model definition
    INPUT_DIM = X_train_input.shape[1]
    N_CLASSES = y_train_input.shape[1]

    train_dataset = TensorDataset(X_train_input, y_train_input)
    val_dataset = TensorDataset(X_test_input, y_test_input)

    train_loader = DataLoader(
        train_dataset,
        batch_size=wandb.config["batch_size"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=wandb.config["batch_size"],
        pin_memory=True,
    )
    ##########################
    ####### Model Train ######
    ##########################
    try:

        # Callbacks

        lr_monitor = LearningRateMonitor(logging_interval="step")
        # log the histograms of input data sent to LightningModule.training_step
        training_data_monitor = TrainingDataMonitor(log_every_n_steps=25)
        print_callback = PrintCallback()

        early_stopping = EarlyStopping(
            min_delta=0.00001, patience=3, verbose=True, monitor="validation_loss"
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath="/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/models",
            filename=f"{WANDB_RUN_NAME}.model",
            monitor="validation_loss",
        )

        # Predict after trainer callback using 20% of the validation dataset
        after_train_dataset = val_dataset[
            np.random.choice(
                len(val_dataset), int(len(val_dataset) * 0.2), replace=False
            )
        ]
        val_preds_logger = PredictionLogger(after_train_dataset)

        callbacks = [
            lr_monitor,
            training_data_monitor,
            print_callback,
            early_stopping,
            checkpoint_callback,
            val_preds_logger,
        ]

        # Logger
        wandb_logger = WandbLogger(project="test-project")

        trainer = Trainer(
            logger=wandb_logger,
            callbacks=callbacks,
            check_val_every_n_epoch=eval_freq,
            devices="auto",
            accelerator="auto",
            fast_dev_run=fast_dev_run,
        )

        if wandb.config["loss_fn"] == "focal":
            # TODO: the loss balance param seems to have the opposite effect?
            # shouldn't loss go up as gamma becomes bigger
            loss = MultilabelFocalLoss(
                N_CLASSES, gamma=wandb.config["focal_loss_gamma"]
            )
        elif wandb.config["loss_fn"] == "bce":
            loss = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            loss = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

        mlp_system = LitAbdPainPredictionMLP(
            INPUT_DIM,
            N_CLASSES,
            config=wandb.config,
            loss_fn=loss,
            layer_size=wandb.config["layer_size"],
            dropout=wandb.config["dropout"],
        )

        print(mlp_system, flush=True)
        wandb_logger.watch(mlp_system)

        trainer.fit(mlp_system, train_loader, val_loader)

    except Exception as e:
        import traceback

        traceback.print_exc()
    finally:
        wandb.finish()


# def main(args):
#     model = LightningModule()
#     trainer = Trainer.from_argparse_args(args)
#     trainer.fit(model)


if __name__ == "__main__":
    #     parser = ArgumentParser()
    #     parser = Trainer.add_argparse_args(parser)
    #     args = parser.parse_args()

    #     main(args)
    #############
    ### Sweep ###
    #############
    config_yaml = "/home/vs428/Documents/deep-ed-diags/sweepv1.yaml"

    with open(config_yaml) as file:
        try:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    train()
#     sweep_id = wandb.sweep(config_dict,
#                           project="test-project")
#     wandb.agent(sweep_id, function=train)
