import os
import sys
import string 
import random 

import ast
import pickle as pkl

from sklearn import preprocessing
from sklearn.metrics import top_k_accuracy_score

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

from models.simple_nn import * 

import wandb
import torchmetrics

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


#################
### Data Load ###
#################
def load_data(data_fp, fast_dev_run=False):
    le = preprocessing.LabelEncoder()
    std_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    if fast_dev_run:
        data = pd.read_csv(data_fp , nrows=100)
    else:
        data = pd.read_csv(data_fp)
    # data = data.sample(50000)

    print(f"dataset size: {data.shape}")

    data = data.rename(columns={"WikEM_overalltopic" : "label"})
    logging.info(f"Data loaded successfully!")

    single_support_classes = set(data['label'].value_counts()[data['label'].value_counts() == 1].index)
    droppable_rows = data['label'].isin(single_support_classes).sum()
    data = data[~data['label'].isin(single_support_classes)]
    train_test_stratify = data['label']

    # drop all columns that don't have any positive actual values/only have all NaNs
    data = data.drop( data.columns[((data.shape[0]- data.isnull().sum()) == 0)], 
                     axis=1, errors='ignore' )

    # remove columns that don't have at least N (hyperparam) number of non-NaN values
    data = data[ data.columns.intersection(data.columns[(((data.shape[0]- data.isnull().sum())) > wandb.config['drop_sparse_cols'])]) ]
    print(f"After dropping sparse columns: {data.shape}")

    data.columns = data.columns.str.replace("[|]|<", "leq_")
    # logging.info(f"Dropped {droppable_rows} rows for stratified K-fold with {single_support_classes} classes")

    N_CLASSES = data['label'].nunique() # 65

    # drop EDDisposition, ID, and label columns
    non_train_col_mask = data.columns[data.columns.str.contains("EdDisposition_")].union(data.columns[:3], sort=False).union(pd.Index(['label']), sort=False)
    train_col_mask = data.columns.difference(non_train_col_mask, sort=False)

    le.fit(data['label'])
    # get the 1/class_size weights for training
    if wandb.config['class_weight_type']:
        samples_per_cls = data['label'].value_counts()

    #    class_weights = torch.tensor((1/data['label'].value_counts()).reindex(le.classes_).values).float()
    #    class_weights = class_weights.to(device)
    #     from sklearn.utils import compute_class_weight
    #     class_weights = compute_class_weight(
    #                                         class_weight="balanced",
    #                                         classes=le.classes_,
    #                                         y=data['label']                                                
    #                                     )
    #     class_weights = torch.tensor(class_weights).float().to(device)
        if wandb.config['class_weight_type'] == "effective_sample":
            beta = wandb.config['weight_beta']
            no_of_classes = samples_per_cls.size
            class_weights = pd.Series(_effective_num_weighting(beta, samples_per_cls, no_of_classes),
                                index=samples_per_cls.index)

        elif wandb.config['class_weight_type'] == "balanced":
            class_weights = pd.Series(compute_class_weight(class_weight="balanced",classes=samples_per_cls.index, y=data['label']),
                                index=samples_per_cls.index )
        elif wandb.config['class_weight_type'] == "inverse":
            # multiply by total/2 as per tensorflow core example imbalanced classes
            class_weights = (1 / samples_per_cls) * (samples_per_cls.sum() / wandb.config['class_weight_inv_lambda'])

        print(class_weights)
        class_weights = torch.tensor(class_weights.values).float().to(device)

    #######
    # Scale variables either MinMax/Standard depending on if they are Normal-ish or not
    # Note: We do the fitting after train_test_split since we only fit on training data
    #######

    # first separate out the corresponding cols
    # all those that take avgs probably are normal
    avg_cols = data.columns[data.columns.str.contains("_avg")].union(pd.Index(["age"]))
    # same with vitals
    vital_cols = pd.Index(['last_SpO2', 'last_Temp', 'last_Patient Acuity', 'last_Pulse', 'last_Pain Score', 'last_Resp', 
     'last_BP_Systolic', 'last_BP_Diastolic', 'ed_SpO2', 'ed_Temp', 
     'ed_Patient Acuity', 'ed_Pulse', 'ed_Pain Score', 'ed_Resp', 'ed_BP_Systolic', 'ed_BP_Diastolic'])
    normal_cols = avg_cols.union(vital_cols)

    # all the other columns are not normal, since they are all counts, which have a long tail (likely Poisson)
    all_other_cols = data.columns.difference(avg_cols, sort=False)
    all_other_cols = all_other_cols.difference(vital_cols, sort=False)
    # remove label as well, since we don't scale it
    all_other_cols = all_other_cols.difference(pd.Index(['label']), sort=False)


    # Create a dummy indices variable to get the indices
    indices = range(data.shape[0])
    X_train, X_test, y_train, y_test, idxs_train, idxs_test = train_test_split(data[train_col_mask],
                                                        data['label'],
                                                        indices,
                                                        stratify=train_test_stratify, # we would like to stratify by label, but we have a few that only have one instance
                                                       test_size=0.2, random_state=314, 
                                                       )
    print("le shape", le.classes_.shape)
    logging.info(f"Created training/test sets")

    print(data[train_col_mask].shape)
    print(X_train.shape)

    # get the indices for the pandas column names 
    normal_col_idxs = column_index(data[train_col_mask], train_col_mask.intersection(normal_cols))
    all_other_cols_idxs = column_index(data[train_col_mask], train_col_mask.intersection(all_other_cols))
    print("Non Normal columns with MinMax Scaling: ", train_col_mask.intersection(normal_cols).tolist())
    print("Non Normal columns with MinMax Scaling: ", train_col_mask.intersection(all_other_cols).tolist())

    # fit the indices by data type
    std_scaler.fit(X_train.iloc[:, normal_col_idxs])
    minmax_scaler.fit(X_train.iloc[:, all_other_cols_idxs])

    # scale/normalize based on the column idxs above
    X_train_input = torch.zeros(X_train.values.shape)
    X_test_input = torch.zeros(X_test.values.shape)

    ### TODO: The error is in this line where the transform function expects what we trained on, which we aren't using here. terrible. 
    X_train_input[:, normal_col_idxs] = torch.tensor(np.nan_to_num(std_scaler.transform(X_train.iloc[:, normal_col_idxs])),
                                                     dtype=torch.float)
    X_train_input[:, all_other_cols_idxs] = torch.tensor(np.nan_to_num(minmax_scaler.transform(X_train.iloc[:, all_other_cols_idxs])),
                                                         dtype=torch.float)
    X_test_input[:, normal_col_idxs] = torch.tensor(np.nan_to_num(std_scaler.transform(X_test.iloc[:, normal_col_idxs])),
                                                    dtype=torch.float)
    X_test_input[:, all_other_cols_idxs] = torch.tensor(np.nan_to_num(minmax_scaler.transform(X_test.iloc[:, all_other_cols_idxs])),
                                                        dtype=torch.float)

    return X_train_input, X_test_input,  y_train, y_test, idxs_train, idxs_test, class_weights