import os
import sys
import string
import random

import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import optuna


import matplotlib.pyplot as plt

import csv
import ast
import pickle as pkl
import logging

import numpy as np
import pandas as pd

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.optim.lr_scheduler import *

import wandb

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from utils.simple_nn_utils import *
from utils.model_utils import *
from models.simple_nn import *


################
### Settings ###
################

# data_fp = f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/unq_pt_enc_clean_multilabel_nomis_dvemb.pkl"
data_fp = f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/unq_pt_enc_clean_multilabel_nomismatches.pkl"
logger_fp = f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/model_outputs/simple_nn_logging.txt"
labels_fp = f"/home/vs428/Documents/deep-ed-diags/label_list.txt"

config = {
    ###### DATA CONFIG
    "basic_col_subset": False,
    # drop columns that don't have at least this many non-NA values
    "drop_sparse_cols": 5000,
    # downsample majority class
    "downsample": False,
    ###### CLASS WEIGHT CONFIG
    # can be "inverse", None, "effective_sample", or "balanced", "constant", "bce_weights"
    "class_weight_type": "constant",
    # The amount to normalize by in "inverse" class weighting
    # "class_weight_inv_lambda":10.0,
    # only used if class_weight_type is "effective sample"
    # "weight_beta" : 0.999,
    # only used if class_weight_type is "constant"
    "constant_weight": 1000,
    ###### PROGRAM CONFIG
    # how often do we want to do evaluation
    "eval_freq": 2,
    ###### MODEL CONFIG
    # two options, "focal" or "bce"
    "loss_fn": "bce",
    "learning_rate": 0.0001,
    "lr_weight_decay": 0.0,
    "lr_scheduler": False,
    "focal_loss_gamma": 10.0,
    "layer_size": 128,
    "epochs": 50,
    "batch_size": 128,
    "dropout": 0.0,
}

# just fix this issue of conditional params
if config["loss_fn"] == "focal":
    config["class_weight_type"] = None


wandb.init(project="test-project", entity="decile", config=config, save_code=True)
WANDB_RUN_NAME = wandb.run.name

logging.basicConfig(
    filename=logger_fp,
    level=logging.INFO,
    format=f"{WANDB_RUN_NAME} | %(asctime)s | %(levelname)s | %(message)s",
)
logging.info(f"Data File loc: {data_fp}")


device = "cuda" if torch.cuda.is_available() else "cpu"


print(wandb.config)
torch.set_printoptions(profile="default", sci_mode=False, precision=3, linewidth=75)
#################
### Data Load ###
#################

std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

data = pd.read_pickle(data_fp)
# data = data.sample(50000)

if wandb.config["basic_col_subset"]:
    with open("./basic_col_subset.txt") as f:
        cols = f.read().splitlines()
    # switch out label for multilabel
    cols = cols[:-1]
    cols.append("multilabel")
    data = data[cols]

with open(labels_fp, "r+") as f:
    label_list = f.read().splitlines()


# data = data[test_cols]

print(f"dataset size: {data.shape}")

logging.info(f"Data loaded successfully!")

# This step is already done in the dataset so we skip it
# single_support_classes = set(data['label'].value_counts()[data['label'].value_counts() == 1].index)
# droppable_rows = data['label'].isin(single_support_classes).sum()
# data = data[~data['label'].isin(single_support_classes)]

# TODO: downsample doesn't work for multilabel yet
if wandb.config["downsample"]:
    # we use the avg of the freq of the next 5 classes to downsample abdominal pain
    downsample_rate = int(data["label"].value_counts()[1:5].mean())

    def downsample_grp(grp):
        if grp.name == "Abdominal Pain, general":
            return grp.sample(downsample_rate)
        else:
            return grp

    downsampled = data.groupby("label").apply(downsample_grp)
    data = downsampled.drop("label", axis=1).reset_index().set_index("level_1")
    data.index.name = None
    # TODO: we don't do stratification on multilabel, is that okay?
#     train_test_stratify = data['label']

else:
    pass
#     train_test_stratify = data['label']


# drop all columns that don't have any positive actual values/only have all NaNs
data = data.drop(
    data.columns[((data.shape[0] - data.isnull().sum()) == 0)], axis=1, errors="ignore"
)

# remove columns that don't have at least N (hyperparam) number of non-NaN values
data = data[
    data.columns.intersection(
        data.columns[
            (((data.shape[0] - data.isnull().sum())) > wandb.config["drop_sparse_cols"])
        ]
    )
]
print(f"After dropping sparse columns: {data.shape}")

data.columns = data.columns.str.replace("[|]|<", "leq_")
# logging.info(f"Dropped {droppable_rows} rows for stratified K-fold with {single_support_classes} classes")

# drop EDDisposition, ID, and label columns
non_train_col_mask = (
    data.columns[data.columns.str.contains("EdDisposition_")]
    .union(data.columns[:3], sort=False)
    .union(pd.Index(["multilabel"]), sort=False)
)
train_col_mask = data.columns.difference(non_train_col_mask, sort=False)

#######
# Scale variables either MinMax/Standard depending on if they are Normal-ish or not
# Note: We do the fitting after train_test_split since we only fit on training data
#######

# first separate out the corresponding cols

# all those that take avgs probably are normal
avg_cols = data.columns[data.columns.str.contains("_avg")].union(pd.Index(["age"]))
# same with vitals
vital_cols = pd.Index(
    [
        "last_SpO2",
        "last_Temp",
        "last_Patient Acuity",
        "last_Pulse",
        "last_Pain Score",
        "last_Resp",
        "last_BP_Systolic",
        "last_BP_Diastolic",
        "ed_SpO2",
        "ed_Temp",
        "ed_Patient Acuity",
        "ed_Pulse",
        "ed_Pain Score",
        "ed_Resp",
        "ed_BP_Systolic",
        "ed_BP_Diastolic",
    ]
)
normal_cols = avg_cols.union(vital_cols).union(
    data.columns[data.columns.str.contains("doc2vec_")]
)
# pull in the doc2vec embeddings as well


# all the other columns are not normal, since they are all counts, which have a long tail (likely Poisson)
# except for the purely categorical ones
cat_col_headers = [
    "EdDisposition_",
    "DepartmentName_",
    "Sex_",
    "GenderIdentity_",
    "FirstRace_",
    "Ethnicity_",
    "PreferredLanguage_",
    "SmokingStatus_",
    "AcuityLevel_",
    "FinancialClass_",
    "CC_",
]

cat_cols = pd.Index(
    flatten(
        [
            data.columns[data.columns.str.contains(col)].tolist()
            for col in cat_col_headers
        ]
    )
)

all_other_cols = data.columns.difference(cat_cols, sort=False)
all_other_cols = all_other_cols.difference(normal_cols, sort=False)
# remove label as well, since we don't scale it
all_other_cols = all_other_cols.difference(pd.Index(["multilabel"]), sort=False)


# Create a dummy index variable to get the indices
indices = range(data.shape[0])
X_train, X_test, y_train, y_test, idxs_train, idxs_test = train_test_split(
    data[train_col_mask],
    data["multilabel"],
    indices,
    #                                                     stratify=train_test_stratify, # don't know how this works with multilabel
    test_size=0.2,
    random_state=314,
)
logging.info(f"Created training/test sets")

# the classes can only be of the trained dataset
UNQ_LABEL_LIST = set(flatten(y_train.tolist()))

# get the subset of labels that are in the training class set
label_list = [x for x in label_list if x in UNQ_LABEL_LIST]
N_CLASSES = len(label_list)

# get class weights using only training data
if wandb.config["class_weight_type"]:
    class_weights = get_class_weights(y_train.tolist(), wandb, data.shape[0])
    class_weights = torch.tensor(class_weights).float().to(device)
else:
    class_weights = None
    print(class_weights)
print(data[train_col_mask].shape)
print(X_train.shape)

# get the indices for the pandas column names
cat_col_idxs = column_index(data[train_col_mask], train_col_mask.intersection(cat_cols))
normal_col_idxs = column_index(
    data[train_col_mask], train_col_mask.intersection(normal_cols)
)
all_other_cols_idxs = column_index(
    data[train_col_mask], train_col_mask.intersection(all_other_cols)
)

with open(
    f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/model_metadata/{WANDB_RUN_NAME}_metadata.txt",
    "a+",
) as f:
    f.write(
        f"Categorical columns with no Scaling:\n{train_col_mask.intersection(cat_cols).tolist()}\n"
    )
    time.sleep(2)
    f.write(
        f"Normal columns with Standard Scaling:\n{train_col_mask.intersection(normal_cols).tolist()}\n"
    )
    time.sleep(2)
    f.write(
        f"Non Normal columns with MinMax Scaling:\n{train_col_mask.intersection(all_other_cols).tolist()}\n"
    )
    time.sleep(2)
    f.write(f"labels:\n{label_list}\n")
    time.sleep(2)

# fit the indices by data type
# we check if there are any such columns
# if not, this fails silently by indexing with empty arrays
if len(normal_col_idxs) > 0:
    std_scaler.fit(X_train.iloc[:, normal_col_idxs])
    X_train_std_scaled = std_scaler.transform(X_train.iloc[:, normal_col_idxs])
    X_test_std_scaled = std_scaler.transform(X_test.iloc[:, normal_col_idxs])
else:
    X_train_std_scaled = np.array([])
    X_test_std_scaled = np.array([])

if len(all_other_cols_idxs) > 0:
    minmax_scaler.fit(X_train.iloc[:, all_other_cols_idxs])
    X_train_minmax_scaled = minmax_scaler.transform(
        X_train.iloc[:, all_other_cols_idxs]
    )
    X_test_minmax_scaled = minmax_scaler.transform(X_test.iloc[:, all_other_cols_idxs])
else:
    X_train_minmax_scaled = np.array([])
    X_test_minmax_scaled = np.array([])

# scale/normalize based on the column idxs above
X_train_input = torch.zeros(X_train.values.shape)
X_test_input = torch.zeros(X_test.values.shape)

### TODO: The error is in this line where the transform function expects what we trained on, which we aren't using here. terrible.
X_train_input[:, cat_col_idxs] = torch.tensor(
    np.nan_to_num(X_train.iloc[:, cat_col_idxs]), dtype=torch.float
)

X_train_input[:, normal_col_idxs] = torch.tensor(
    np.nan_to_num(X_train_std_scaled), dtype=torch.float
)
X_train_input[:, all_other_cols_idxs] = torch.tensor(
    np.nan_to_num(X_train_minmax_scaled), dtype=torch.float
)


X_test_input[:, cat_col_idxs] = torch.tensor(
    np.nan_to_num(X_test.iloc[:, cat_col_idxs]), dtype=torch.float
)
X_test_input[:, normal_col_idxs] = torch.tensor(
    np.nan_to_num(X_test_std_scaled), dtype=torch.float
)
X_test_input[:, all_other_cols_idxs] = torch.tensor(
    np.nan_to_num(X_test_minmax_scaled), dtype=torch.float
)

# we need to store this value for the NN model definition
INPUT_DIM = X_train.shape[1]

# For the multilabel case, we have to transform the y_train and y_test datasets ourselves
y_train_input = torch.zeros(X_train_input.shape[0], N_CLASSES, dtype=torch.float)
y_test_input = torch.zeros(X_test_input.shape[0], N_CLASSES, dtype=torch.float)

print("y_train_input", y_train_input.shape)
for row_idx, (_, labels) in enumerate(y_train.items()):
    y_train_input[row_idx, :] = torch.tensor(
        np.isin(label_list, labels, assume_unique=True).astype(float)
    )

for row_idx, (_, labels) in enumerate(y_test.items()):
    y_test_input[row_idx, :] = torch.tensor(
        np.isin(label_list, labels, assume_unique=True).astype(float)
    )


assert (torch.sum(y_train_input, 1) >= 1).all()
# assert((torch.sum(y_test_input, 1) >= 1).all())

train_dataset = TensorDataset(X_train_input, y_train_input)
test_dataset = TensorDataset(X_test_input, y_test_input)

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
# label_freqs = data['label'].value_counts()


try:

    model = AbdPainPredictionMLP(
        INPUT_DIM,
        N_CLASSES,
        layer_size=wandb.config["layer_size"],
        dropout=wandb.config["dropout"],
    ).to(device)
    print(model, flush=True)
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

    if wandb.config["loss_fn"] == "focal":
        loss = MultilabelFocalLoss(N_CLASSES, gamma=wandb.config["focal_loss_gamma"])
    elif wandb.config["loss_fn"] == "bce":
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

    multilabel_train(
        model,
        loss,
        opt,
        train_loader,
        test_loader,
        device,
        wandb,
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
    import traceback

    traceback.print_exc()
finally:
    wandb.finish()
