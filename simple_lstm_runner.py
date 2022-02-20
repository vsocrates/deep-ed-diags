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

################
## Functions ###
################
def _effective_num_weighting(beta, samples_per_cls, no_of_classes):
    """
    Compute the effective number sample weighting as per https://doi.org/10.1109/CVPR.2019.00949

    beta [float] : usually between 0.9 and 0.99, and is the hyperparam for the amount of weighting like inverse freq
    samples_per_cls [list-like] : a list-like object with the number of samples per class that has length no_of_classes
    no_of_classes [int] : the number of classes
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    return weights


################
### Settings ###
################
data_fp = f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/unq_pt_enc_single_label_full_clean_label.csv"
logger_fp = f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/model_outputs/simple_nn_logging.txt"


# create UUID for run
alphabet = string.ascii_lowercase + string.digits
study_run_id = "".join(random.choices(alphabet, k=8))

logging.basicConfig(
    filename=logger_fp,
    level=logging.INFO,
    format=f"{study_run_id} | %(asctime)s | %(levelname)s | %(message)s",
)
logging.info(f"Data File loc: {data_fp}")


device = "cuda" if torch.cuda.is_available() else "cpu"
wandb.init(project="test-project", entity="decile")

wandb.config = {
    "learning_rate": 0.0001,
    "lr_weight_decay": 0.0,
    "lr_scheduler": False,
    "epochs": 500,
    "batch_size": 256,
    "multilabel": True,
    "dropout": 0.3,
    # can be "inverse", None, "effective_sample", or "balanced"
    "class_weight_type": "effective_sample",
    # only used if class_weight_type is "effective sample"
    "weight_beta": 0.95,
}

print(wandb.config)

#################
### Data Load ###
#################

should_reweight = wandb.config["class_weight_type"]

le = preprocessing.LabelEncoder()
scaler = StandardScaler()

# data = pd.read_csv(data_fp , nrows=100)
data = pd.read_csv(data_fp)
# data = data.sample(50000)

print(f"dataset size: {data.shape}")

data = data.rename(columns={"WikEM_overalltopic": "label"})
logging.info(f"Data loaded successfully!")

single_support_classes = set(
    data["label"].value_counts()[data["label"].value_counts() == 1].index
)
droppable_rows = data["label"].isin(single_support_classes).sum()
data = data[~data["label"].isin(single_support_classes)]
train_test_stratify = data["label"]
data = data.fillna(0)
data.columns = data.columns.str.replace("[|]|<", "leq_")
# logging.info(f"Dropped {droppable_rows} rows for stratified K-fold with {single_support_classes} classes")

N_CLASSES = data["label"].nunique()  # 65

non_train_col_mask = (
    data.columns[data.columns.str.contains("EdDisposition_")]
    .union(data.columns[:3], sort=False)
    .union(pd.Index(["label"]), sort=False)
)
train_col_mask = data.columns.difference(non_train_col_mask, sort=False)

le.fit(data["label"])
# get the 1/class_size weights for training
if should_reweight:
    #    class_weights = torch.tensor((1/data['label'].value_counts()).reindex(le.classes_).values).float()
    #    class_weights = class_weights.to(device)
    #     from sklearn.utils import compute_class_weight
    #     class_weights = compute_class_weight(
    #                                         class_weight="balanced",
    #                                         classes=le.classes_,
    #                                         y=data['label']
    #                                     )
    #     class_weights = torch.tensor(class_weights).float().to(device)
    if wandb.config["class_weight_type"] == "effective_sample":
        beta = wandb.config["weight_beta"]
        samples_per_cls = data["label"].value_counts()
        no_of_classes = samples_per_cls.size
        class_weights = pd.Series(
            _effective_num_weighting(beta, samples_per_cls, no_of_classes),
            index=samples_per_cls.index,
        )

    elif wandb.config["class_weight_type"] == "balanced":
        class_weights = pd.Series(
            compute_class_weight(
                class_weight="balanced", classes=samples_per_cls.index, y=data["label"]
            ),
            index=samples_per_cls.index,
        )
    elif wandb.config["class_weight_type"] == "inverse":
        samples_per_cls = data["label"].value_counts()
        # multiply by total/2 as per tensorflow core example imbalanced classes
        class_weights = (1 / samples_per_cls) * (samples_per_cls.sum() / 2.0)

    print(class_weights)
    class_weights = torch.tensor(class_weights.values).float().to(device)
    print(class_weights)

X_train, X_test, y_train, y_test = train_test_split(
    data[train_col_mask],
    data["label"],
    stratify=train_test_stratify,  # we would like to stratify by label, but we have a few that only have one instance
    test_size=0.2,
    random_state=314,
)
print("le shape", le.classes_.shape)
logging.info(f"Created training/test sets")
scaler.fit(X_train.values.astype(np.float32))
train_features = torch.tensor(scaler.transform(X_train.values.astype(np.float32)))
test_features = torch.tensor(scaler.transform(X_test.values.astype(np.float32)))

# print(train_features.shape)
# print(le.transform(y_train).shape)
train_dataset = TensorDataset(train_features, torch.tensor(le.transform(y_train)))
test_dataset = TensorDataset(test_features, torch.tensor(le.transform(y_test)))

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
label_freqs = data["label"].value_counts()


def evaluate(model, loss_fn, evaluation_set):
    """
    Evaluates the given model on the given dataset.
    Returns the percentage of correct classifications out of total classifications.
    """
    correct = 0
    total = 0
    losses = 0
    avg_topk_acc = []

    with torch.no_grad():
        for data, targets in evaluation_set:
            targets = targets.to(device)
            out = model(data.to(device))
            loss = loss_fn(out, targets)
            loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += targets.size(0)
            avg_topk_acc.append(torchmetrics.functional.accuracy(out, targets, top_k=5))

            correct += (predicted == targets).sum().item()
            wandb.log(
                {
                    "pr": wandb.plot.pr_curve(
                        targets.cpu(), out.data.cpu(), labels=label_freqs.index.tolist()
                    )
                }
            )

        accuracy = correct / total
        wandb.log({"validation_loss": losses / len(evaluation_set)})
        wandb.log({"validation_acc": accuracy})
        wandb.log({"top5_validation_acc": sum(avg_topk_acc) / len(avg_topk_acc)})

    return accuracy


def train(
    model,
    loss_fn,
    optimizer,
    train_loader,
    test_loader,
    n_epochs=100,
    class_weights=None,
    scheduler=None,
):
    """
    This is a standard training loop, which leaves some parts to be filled in.
    INPUT:
    :param model: an untrained pytorch model
    :param loss_fn: e.g. Cross Entropy loss of Mean Squared Error.
    :param optimizer: the model optimizer, initialized with a learning rate.
    :param training_set: The training data, in a dataloader for easy iteration.
    :param test_loader: The testing data, in a dataloader for easy iteration.
    """

    for epoch in range(n_epochs):
        for data, targets in train_loader:

            optimizer.zero_grad()
            targets = targets.to(device)
            out = model(data.to(device))
            loss = loss_fn(out, targets)
            wandb.log({"train_loss": loss})
            wandb.log({"train_acc": torchmetrics.functional.accuracy(out, targets)})

            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()
            wandb.log({"lr": scheduler.get_lr()[0]})

        if epoch % 20 == 0:
            print(f" EPOCH {epoch}. Progress: {epoch/n_epochs*100}%. ")
            test_acc = evaluate(model, loss_fn, test_loader)
            print(f"Test accuracy: {test_acc}")

    print(f" EPOCH {n_epochs}. Progress: 100%. ")
    print(
        f" Train accuracy: {evaluate(model,loss_fn, train_loader)}. Test accuracy: {evaluate(model,loss_fn, test_loader)}"
    )


model = AbdPainPredictionMLP(N_CLASSES, wandb.config["dropout"]).to(device)
print(model)
wandb.watch(model)

opt = torch.optim.Adam(
    model.parameters(),
    lr=wandb.config["learning_rate"],
    weight_decay=wandb.config["lr_weight_decay"],
)  # lr=1e-5)

if wandb.config["lr_scheduler"]:
    scheduler = MultiStepLR(
        opt, milestones=list(range(0, wandb.config["epochs"], 100))[1:], gamma=0.5
    )
else:
    scheduler = None

# opt = torch.optim.RMSprop(model.parameters(), lr=0.001)
if should_reweight:
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
    "/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/models/simple_nn_v2.model",
)
# y_pred = model(test_features)
# top_k_accuracy_score(y_test, y_pred, labels=le.classes_, k=10)
