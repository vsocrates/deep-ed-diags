import os
import sys
import string
import random
import ast
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms, utils
import torchmetrics

import wandb

from .general_utils import *
from .simple_nn_utils import *

#################
### Data Load ###
#################
def load_multilabel_data(data_fp, config, run_name, fast_dev_run=False, basic_cols=False, test_size=0.2, standardize=True):

    std_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    data = pd.read_pickle(data_fp)
    if fast_dev_run:
        data = data.sample(100)

    if config["basic_col_subset"]:
        with open("/home/vs428/Documents/deep-ed-diags/basic_col_subset.txt") as f:
            cols = f.read().splitlines()

        cols = cols[:-1]
        cols.append("multilabel")
        data = data[cols]

    with open(f"/home/vs428/Documents/deep-ed-diags/label_list.txt", "r+") as f:
        label_list = f.read().splitlines()

    print(f"dataset size: {data.shape}")

    # TODO: downsample doesn't work for multilabel yet
#     if wandb.config["downsample"]:
#         # we use the avg of the freq of the next 5 classes to downsample abdominal pain
#         downsample_rate = int(data["label"].value_counts()[1:5].mean())

#         def downsample_grp(grp):
#             if grp.name == "Abdominal Pain, general":
#                 return grp.sample(downsample_rate)
#             else:
#                 return grp

#         downsampled = data.groupby("label").apply(downsample_grp)
#         data = downsampled.drop("label", axis=1).reset_index().set_index("level_1")
#         data.index.name = None
#         # TODO: we don't do stratification on multilabel, is that okay?
#     #     train_test_stratify = data['label']

#     else:
#         pass
#     #     train_test_stratify = data['label']

    # drop all columns that don't have any positive actual values/only have all NaNs
    data = data.drop(
        data.columns[((data.shape[0] - data.isnull().sum()) == 0)], axis=1, errors="ignore"
    )

    # remove columns that don't have at least N (hyperparam) number of non-NaN values
    data = data[
        data.columns.intersection(
            data.columns[
                (((data.shape[0] - data.isnull().sum())) > config["drop_sparse_cols"])
            ]
        )
    ]
    print(f"After dropping sparse columns: {data.shape}")

    data.columns = data.columns.str.replace("[|]|<", "leq_")

    # drop EDDisposition, ID, and label columns
    non_train_col_mask = (
        data.columns[data.columns.str.contains("EdDisposition_")]
        .union(data.columns[:3], sort=False)
        .union(pd.Index(["multilabel"]), sort=False)
    )
    # get only those training columns that exist in the dataset after dropping those sparse ones
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
        # stratify=train_test_stratify, # TODO: don't know how this works with multilabel
        test_size=test_size,
        random_state=314,
    )

    # the classes can only be of the trained dataset
    UNQ_LABEL_LIST = set(flatten(y_train.tolist()))

    # get the subset of labels that are in the training class set
    label_list = [x for x in label_list if x in UNQ_LABEL_LIST]
    N_CLASSES = len(label_list)

    # get class weights using only training data
    class_weights = None
    if config["class_weight_type"]:
        class_weights = multilabel_get_class_weights(y_train.tolist(), config, data.shape[0], label_list)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights).float()
        print(class_weights)
        
    print(data[train_col_mask].shape)
    print(X_train.shape)
    if standardize:
        # get the indices for the pandas column names
        cat_col_idxs = column_index(data[train_col_mask], train_col_mask.intersection(cat_cols))
        normal_col_idxs = column_index(
            data[train_col_mask], train_col_mask.intersection(normal_cols)
        )
        all_other_cols_idxs = column_index(
            data[train_col_mask], train_col_mask.intersection(all_other_cols)
        )

        with open(
            f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/model_metadata/{run_name}_metadata.txt",
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
    else:
        # scale/normalize based on the column idxs above
        X_train_input = torch.tensor(np.nan_to_num(X_train.values)).float()
        X_test_input = torch.tensor(np.nan_to_num(X_test.values)).float()
        

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

    # TODO: put this in the right compute_class_weight function, but works for now
    if config["class_weight_type"] == "pos_weight":
        num_positives = torch.sum(y_train_input, dim=0)
        num_negatives = len(y_train_input) - num_positives
        pos_weight  = num_negatives / (num_positives + 1e-5)
        class_weights = pos_weight
        print(class_weights)
        
    # make sure that we have at least positive class per data example
    assert (torch.sum(y_train_input, 1) >= 1).all()
    
    return X_train_input, y_train_input, X_test_input, y_test_input, class_weights

    