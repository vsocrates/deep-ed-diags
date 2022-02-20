import os
import sys
import string
import random


# from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import optuna


import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay

import csv
import ast
from datetime import date, datetime, time, timedelta
from datetime import datetime
import pickle as pkl
import time

from sklearn import preprocessing
from sklearn.metrics import top_k_accuracy_score

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from joblib import dump, load

import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.optim.lr_scheduler import *

from models.simple_nn import *

import wandb
import torchmetrics

import logging


import pandas as pd
import numpy as np
import torch

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger  # newline 1
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from models.simple_nn_lightning import *
from utils.simple_nn_utils import *
from utils.data_loader import *

################
### Settings ###
################

data_fp = f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/unq_pt_enc_single_label_full_clean_label.csv"
logger_fp = f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/model_outputs/simple_nn_logging.txt"

# Create W&B study
wandb.init(project="test-project", entity="decile")
WANDB_RUN_NAME = wandb.run.name

logging.basicConfig(
    filename=logger_fp,
    level=logging.INFO,
    format=f"{WANDB_RUN_NAME} | %(asctime)s | %(levelname)s | %(message)s",
)
logging.info(f"Data File loc: {data_fp}")


device = "cuda" if torch.cuda.is_available() else "cpu"


wandb.config = {
    "learning_rate": 0.00001,
    "lr_weight_decay": 0.0,
    "lr_scheduler": False,
    "epochs": 500,
    "batch_size": 256,
    # drop columns that don't have at least this many non-NA values
    "drop_sparse_cols": 10,
    #     "multilabel":True,
    "dropout": 0.2,
    # can be "inverse", None, "effective_sample", or "balanced"
    "class_weight_type": "inverse",
    # The amount to normalize by in inverse class weighting
    "class_weight_inv_lambda": 10000.0,
    # only used if class_weight_type is "effective sample"
    "weight_beta": 0.999,
}

print(wandb.config)

# load data
(
    X_train_input,
    X_test_input,
    y_train,
    y_test,
    idxs_train,
    idxs_test,
    class_labels,
) = load_data(data_fp)

# we need to store this value for the NN model definition
INPUT_DIM = X_train_input.shape[1]


train_dataset = TensorDataset(X_train_input, torch.tensor(le.transform(y_train)))
test_dataset = TensorDataset(X_test_input, torch.tensor(le.transform(y_test)))

train_loader = DataLoader(
    train_dataset,
    batch_size=wandb.config["batch_size"],
    #  collate_fn=collate_wrapper,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=wandb.config["batch_size"],
    #  collate_fn=collate_wrapper,
    pin_memory=True,
)
logging.info(f"Created train/test loaders")
##########################
### Model Train Funcs. ###
##########################


try:

    model = AbdPainPredictionMLP(INPUT_DIM, N_CLASSES, wandb.config["dropout"]).to(
        device
    )
    print(model, flush=True)
    wandb.watch(model)

    wandb_logger = WandbLogger()  # newline 2
    trainer = Trainer(logger=wandb_logger)

    if wandb.config["lr_scheduler"]:
        scheduler = MultiStepLR(
            opt, milestones=list(range(0, wandb.config["epochs"], 100))[1:], gamma=0.5
        )
    else:
        scheduler = None

    # opt = torch.optim.RMSprop(model.parameters(), lr=0.001)
    if wandb.config["class_weight_type"]:
        loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        print(class_weights.shape)
    else:
        loss = torch.nn.CrossEntropyLoss()

    train(
        model,
        loss,
        opt,
        train_loader,
        test_loader,
        n_epochs=wandb.config["epochs"],
        scheduler=scheduler,
    )

    torch.save(
        model,
        f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/models/{WANDB_RUN_NAME}.model",
    )
    # y_pred = model(test_features)
    # top_k_accuracy_score(y_test, y_pred, labels=le.classes_, k=10)


except Exception as e:
    pass
finally:
    wandb.finish()
