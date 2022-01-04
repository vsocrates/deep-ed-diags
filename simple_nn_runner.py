import os
import sys


# from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


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

import numpy as np
import pandas as pd
import sklearn
from joblib import dump, load

import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from models.simple_nn import * 


################
### Settings ###
################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#################
### Data Load ###
#################
data_fp = f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/unq_pt_enc_single_label_full_clean_label.csv"
N_CLASSES = 65

le = preprocessing.LabelEncoder()

data = pd.read_csv(data_fp, nrows=10000)
data = data.rename(columns={"WikEM_overalltopic" : "label"})
single_support_classes = set(data['label'].value_counts()[data['label'].value_counts() == 1].index)
droppable_rows = data['label'].isin(single_support_classes).sum()
data = data[~data['label'].isin(single_support_classes)]
train_test_stratify = data['label']
data = data.fillna(0)
data.columns = data.columns.str.replace("[|]|<", "leq_")
# logging.info(f"Dropped {droppable_rows} rows for stratified K-fold with {single_support_classes} classes")

non_train_col_mask = data.columns[data.columns.str.contains("EdDisposition_")].union(data.columns[:3], sort=False).union(pd.Index(['label']), sort=False)
train_col_mask = data.columns.difference(non_train_col_mask, sort=False)

le.fit(data['label'])
X_train, X_test, y_train, y_test = train_test_split(data[train_col_mask],
                                                    data['label'],
                                                    stratify=train_test_stratify, # we would like to stratify by label, but we have a few that only have one instance
                                                   test_size=0.2, random_state=314, 
                                                   )


train_features = torch.tensor(X_train.values.astype(np.float32))
test_features = torch.tensor(X_test.values.astype(np.float32))

# print(train_features.shape)
# print(le.transform(y_train).shape)
train_dataset = TensorDataset(train_features, torch.tensor(le.transform(y_train)))
test_dataset = TensorDataset(test_features, torch.tensor(le.transform(y_test)))

train_loader = DataLoader(train_dataset, batch_size=64,
#  collate_fn=collate_wrapper,
                    pin_memory=True)

test_loader = DataLoader(test_dataset, batch_size=64,
#  collate_fn=collate_wrapper,
                    pin_memory=True)

##########################
### Model Train Funcs. ###
##########################
def evaluate(model, evaluation_set):
    """
    Evaluates the given model on the given dataset.
    Returns the percentage of correct classifications out of total classifications.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in evaluation_set:
            out = model(data)
            print(type(out))
            print(out.shape)
            _, predicted = torch.max(out.data, 1)
            print(predicted.shape)
            print(type(predicted))

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        accuracy = (correct/total)*100
    return accuracy




def train(model, loss_fn, optimizer, train_loader, test_loader,
            n_epochs=100):
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
            out = model(data)

            loss = loss_fn(out, targets)

            loss.backward()
            optimizer.step()


        if epoch % 5 == 0:
            print(f" EPOCH {epoch}. Progress: {epoch/n_epochs*100}%. ")
            print(f" Train accuracy: {evaluate(model,train_loader)}. Test accuracy: {evaluate(model,test_loader)}") 

    print(f" EPOCH {n_epochs}. Progress: 100%. ")
    print(f" Train accuracy: {evaluate(model,train_loader)}. Test accuracy: {evaluate(model,test_loader)}") 


model = AbdPainPredictionMLP(N_CLASSES).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-6)
loss = torch.nn.CrossEntropyLoss()

train(model, loss, opt, train_loader, test_loader)
