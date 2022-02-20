import os
import sys
import string 
import random 

import time

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

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils.simple_nn_utils import *


################
### Settings ###
################

data_fp = f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/unq_pt_enc_single_label_clean_label_nomismatches.csv"
logger_fp = f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/model_outputs/simple_nn_logging.txt"

# Create W&B study
wandb.init(project="test-project", entity="decile")
WANDB_RUN_NAME = wandb.run.name

logging.basicConfig(filename=logger_fp, 
                    level=logging.INFO,
                   format=f'{WANDB_RUN_NAME} | %(asctime)s | %(levelname)s | %(message)s')
logging.info(f'Data File loc: {data_fp}')


device = 'cuda' if torch.cuda.is_available() else 'cpu'


wandb.config = {
    # how often do we want to do evaluation
    "eval_freq":5,
    "learning_rate": 0.0001,
    "lr_weight_decay":0.0,
    "lr_scheduler":False,
    "epochs": 100,
    "batch_size": 256,
    # drop columns that don't have at least this many non-NA values
    "drop_sparse_cols":5,
    # downsample majority class
    "downsample":True,
    "dropout":0.0,
    # can be "inverse", None, "effective_sample", or "balanced"
    "class_weight_type":None,
    # The amount to normalize by in "inverse" class weighting
    "class_weight_inv_lambda":10.0,
    # only used if class_weight_type is "effective sample"
    "weight_beta" : 0.999,
}

print(wandb.config)

#################
### Data Load ###
#################

le = preprocessing.LabelEncoder()
std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

data = pd.read_csv(data_fp , nrows=50000)
# data = pd.read_csv(data_fp)
# data = data.sample(50000)

# test_cols = ['Sex_Female', 'Sex_Male', 'Sex_Unknown', 'GenderIdentity_Choose not to disclose', 
# 'GenderIdentity_Female', 'GenderIdentity_Gender non-conforming', 'GenderIdentity_Genderqueer',
# 'GenderIdentity_Intersex', 'GenderIdentity_Male', 'GenderIdentity_Other', 
# 'GenderIdentity_Transgender Female / Male-to-Female', 
# 'GenderIdentity_Transgender Male / Female-to-Male', 'FirstRace_Asian', 
# 'FirstRace_Black or African American', 'FirstRace_Native Hawaiian', 
# 'FirstRace_Native Hawaiian or Other Pacific Islander', 'FirstRace_Other', 
# 'FirstRace_Other Pacific Islander', 'FirstRace_Other/Not Listed', 
# 'FirstRace_Patient Refused', 'FirstRace_Unknown', 'FirstRace_White or Caucasian',
# 'Ethnicity_Hispanic or Latino', 'Ethnicity_Non-Hispanic', 'Ethnicity_Patient Refused',
# 'Ethnicity_Unknown','SmokingStatus_*Unspecified', 'SmokingStatus_Current Every Day Smoker', 
#  'SmokingStatus_Current Some Day Smoker', 'SmokingStatus_Former Smoker', 
#  'SmokingStatus_Heavy Tobacco Smoker', 'SmokingStatus_Light Tobacco Smoker',
#  'SmokingStatus_Never Assessed', 'SmokingStatus_Never Smoker', 'SmokingStatus_Never Smoker ', 
#  'SmokingStatus_Passive Smoke Exposure - Never Smoker', 
#  'SmokingStatus_Smoker, Current Status Unknown',
#  'SmokingStatus_Unknown If Ever Smoked', 'AcuityLevel_Emergent', 
#  'AcuityLevel_Immediate', 'AcuityLevel_Less Urgent', 'AcuityLevel_Non-Urgent',
#  'AcuityLevel_Urgent','CC_ABDOMINAL DISTENTION', 'CC_ABDOMINAL PAIN',
#  'CC_ABDOMINAL PAIN PREGNANT', 'CC_EPIGASTRIC PAIN', 'CC_FLANK PAIN',
#  'CC_PELVIC PAIN', 'CC_PELVIC PAIN-PREGNANT', 'CC_SIDE PAIN','ed_SpO2',
#  'ed_Temp', 'ed_Patient Acuity', 'ed_Pulse', 'ed_Pain Score', 'ed_Resp', 
#  'ed_BP_Systolic', 'ed_BP_Diastolic', 'label']

# data = data[test_cols]

print(f"dataset size: {data.shape}")

logging.info(f"Data loaded successfully!")

single_support_classes = set(data['label'].value_counts()[data['label'].value_counts() == 1].index)
# droppable_rows = data['label'].isin(single_support_classes).sum()
data = data[~data['label'].isin(single_support_classes)]

if wandb.config['downsample']:
    # we use the avg of the freq of the next 5 classes to downsample abdominal pain
    downsample_rate = int(data['label'].value_counts()[1:5].mean())
    def downsample_grp(grp):
        if grp.name == "Abdominal Pain, general":
            return grp.sample(downsample_rate)
        else:
            return grp
    downsampled = data.groupby("label").apply(downsample_grp)    
    data = downsampled.drop("label", axis=1).reset_index().set_index("level_1")
    data.index.name = None
    # reorder the cols to put the 'label' col back to the end
    cols = data.columns.tolist()
    cols = cols[1:] + [cols[0]]
    data = data[cols]
    
    train_test_stratify = data['label']

else:
    train_test_stratify = data['label']

    
# drop all columns that don't have any positive actual values/only have all NaNs
data = data.drop( data.columns[((data.shape[0]- data.isnull().sum()) == 0)], 
                 axis=1, errors='ignore' )

# remove columns that don't have at least N (hyperparam) number of non-NaN values
data = data[ data.columns.intersection(data.columns[((data.shape[0]- data.isnull().sum()) > wandb.config['drop_sparse_cols'])], sort=False) ]
print(f"After dropping sparse columns: {data.shape}")

data.columns = data.columns.str.replace("[|]|<", "leq_")
# logging.info(f"Dropped {droppable_rows} rows for stratified K-fold with {single_support_classes} classes")

# drop EDDisposition, ID, and label columns
non_train_col_mask = data.columns[data.columns.str.contains("EdDisposition_")].union(data.columns[:3], sort=False).union(pd.Index(['label']), sort=False)
train_col_mask = data.columns.difference(non_train_col_mask, sort=False)

le.fit(data['label'])
# get the 1/class_size weights for training
if wandb.config['class_weight_type']:
    samples_per_cls = data['label'].value_counts()

    if wandb.config['class_weight_type'] == "effective_sample":
        beta = wandb.config['weight_beta']
        no_of_classes = samples_per_cls.size
        class_weights = pd.Series(_effective_num_weighting(beta, samples_per_cls, no_of_classes),
                            index=samples_per_cls.index)
    
    elif wandb.config['class_weight_type'] == "balanced":
        class_weights = data.shape[0]/(len(samples_per_cls) * samples_per_cls)
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
vital_cols = pd.Index(['last_SpO2', 'last_Temp', 'last_Patient Acuity', 'last_Pulse', 'last_Pain Score', 'last_Resp', 'last_BP_Systolic', 'last_BP_Diastolic', 'ed_SpO2', 
                       'ed_Temp', 'ed_Patient Acuity', 'ed_Pulse', 'ed_Pain Score', 'ed_Resp', 'ed_BP_Systolic', 'ed_BP_Diastolic'])
normal_cols = avg_cols.union(vital_cols)

# all the other columns are not normal, since they are all counts, which have a long tail (likely Poisson)
# except for the purely categorical ones
cat_col_headers = ["EdDisposition_","DepartmentName_","Sex_",
                     "GenderIdentity_","FirstRace_","Ethnicity_",
                     "PreferredLanguage_","SmokingStatus_",
                     "AcuityLevel_","FinancialClass_","CC_"]

cat_cols = pd.Index(flatten([data.columns[data.columns.str.contains(col)].tolist() for col in cat_col_headers]))

all_other_cols = data.columns.difference(cat_cols, sort=False)
all_other_cols = all_other_cols.difference(normal_cols, sort=False)
# remove label as well, since we don't scale it
all_other_cols = all_other_cols.difference(pd.Index(['label']), sort=False)


# Create a dummy index variable to get the indices
indices = range(data.shape[0])
X_train, X_test, y_train, y_test, idxs_train, idxs_test = train_test_split(data[train_col_mask],
                                                    data['label'],
                                                    indices,
                                                    stratify=train_test_stratify,
                                                   test_size=0.2, random_state=314, 
                                                   )

print("le shape", le.classes_.shape)
logging.info(f"Created training/test sets")

# the classes can only be of the trained dataset
N_CLASSES = y_train.nunique() # 65

print(data[train_col_mask].shape)
print(X_train.shape)

# get the indices for the pandas column names 
cat_col_idxs = column_index(data[train_col_mask], train_col_mask.intersection(cat_cols))
normal_col_idxs = column_index(data[train_col_mask], train_col_mask.intersection(normal_cols))
all_other_cols_idxs = column_index(data[train_col_mask], train_col_mask.intersection(all_other_cols))

with open(f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/model_metadata/{WANDB_RUN_NAME}_metadata.txt", "a+") as f:
    f.write(f"Categorical columns with no Scaling:\n{train_col_mask.intersection(cat_cols).tolist()}\n")
    time.sleep(2)
    f.write(f"Normal columns with Standard Scaling:\n{train_col_mask.intersection(normal_cols).tolist()}\n")
    time.sleep(2)
    f.write(f"Non Normal columns with MinMax Scaling:\n{train_col_mask.intersection(all_other_cols).tolist()}\n")
    time.sleep(2)
    f.write(f"labels:\n{label_list}\n")
    time.sleep(2)

# fit the indices by data type
# we check if there are any such columns
# if not, this fails silently
if len(normal_col_idxs) > 0:
    std_scaler.fit(X_train.iloc[:, normal_col_idxs])
    X_train_std_scaled = std_scaler.transform(X_train.iloc[:, normal_col_idxs])
    X_test_std_scaled = std_scaler.transform(X_test.iloc[:, normal_col_idxs])
else:
    X_train_std_scaled = np.array([])
    X_test_std_scaled = np.array([])
    
if len(all_other_cols_idxs) > 0:
    minmax_scaler.fit(X_train.iloc[:, all_other_cols_idxs])
    X_train_minmax_scaled = minmax_scaler.transform(X_train.iloc[:, all_other_cols_idxs])
    X_test_minmax_scaled = minmax_scaler.transform(X_test.iloc[:, all_other_cols_idxs])
else:
    X_train_minmax_scaled = np.array([])
    X_test_minmax_scaled = np.array([])
    
# scale/normalize based on the column idxs above
X_train_input = torch.zeros(X_train.values.shape)
X_test_input = torch.zeros(X_test.values.shape)

### TODO: The error is in this line where the transform function expects what we trained on, which we aren't using here. terrible. 
X_train_input[:, cat_col_idxs] = torch.tensor(np.nan_to_num(X_train.iloc[:, cat_col_idxs]),
                                                 dtype=torch.float)

X_train_input[:, normal_col_idxs] = torch.tensor(np.nan_to_num(X_train_std_scaled),
                                                 dtype=torch.float)
X_train_input[:, all_other_cols_idxs] = torch.tensor(np.nan_to_num(X_train_minmax_scaled),
                                                     dtype=torch.float)


X_test_input[:, cat_col_idxs] = torch.tensor(np.nan_to_num(X_test.iloc[:, cat_col_idxs]),
                                                dtype=torch.float)
X_test_input[:, normal_col_idxs] = torch.tensor(np.nan_to_num(X_test_std_scaled),
                                                dtype=torch.float)
X_test_input[:, all_other_cols_idxs] = torch.tensor(np.nan_to_num(X_test_minmax_scaled),
                                                    dtype=torch.float)

# we need to store this value for the NN model definition
INPUT_DIM = X_train.shape[1]

train_dataset = TensorDataset(X_train_input, torch.tensor(le.transform(y_train)))
test_dataset = TensorDataset(X_test_input, torch.tensor(le.transform(y_test)))

train_loader = DataLoader(train_dataset, batch_size=wandb.config['batch_size'],
#  collate_fn=collate_wrapper,
                    pin_memory=True)

test_loader = DataLoader(test_dataset, batch_size=wandb.config['batch_size'],
#  collate_fn=collate_wrapper,
                    pin_memory=True)
logging.info(f"Created train/test loaders")
##########################
### Model Train Funcs. ###
##########################
label_freqs = data['label'].value_counts()

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
#             wandb.log({"train_loss": loss})
            loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += targets.size(0)
            avg_topk_acc.append(torchmetrics.functional.accuracy(out, targets, top_k=5))
            

            correct += (predicted == targets).sum().item()
            wandb.log({"pr" : wandb.plot.pr_curve(targets.cpu(), out.data.cpu(),
                     labels=label_freqs.index.tolist()
                                                 )})        

        accuracy = (correct/total)
        wandb.log({"validation_loss": loss/len(evaluation_set)})
        wandb.log({"validation_acc":accuracy})
        wandb.log({"top5_validation_acc":sum(avg_topk_acc)/len(avg_topk_acc)}) 
        
        
    return accuracy

def train(model, loss_fn, optimizer, train_loader, test_loader,
            n_epochs=100, class_weights=None, scheduler=None):
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
            wandb.log({"train_acc":torchmetrics.functional.accuracy(out, targets)})
            
            loss.backward()
            optimizer.step()
        
        if scheduler:
            scheduler.step()
            wandb.log({"lr": scheduler.get_lr()[0]})


        if epoch % wandb.config['eval_freq'] == 0:
            print(f" EPOCH {epoch}. Progress: {epoch/n_epochs*100}%. ", flush=True)
            test_acc = evaluate(model,loss_fn, test_loader)
            print(f"Test accuracy: {test_acc}", flush=True) 
            
    print(f" EPOCH {n_epochs}. Progress: 100%. ", flush=True)
    print(f" Train accuracy: {evaluate(model,loss_fn, train_loader)}. Test accuracy: {evaluate(model,loss_fn, test_loader)}", flush=True) 


    
try:

    model = AbdPainPredictionMLP(INPUT_DIM, N_CLASSES, wandb.config['dropout']).to(device)
    print(model, flush=True)
    wandb.watch(model)

    opt = torch.optim.Adam(model.parameters(), lr=wandb.config['learning_rate'],
                          weight_decay = wandb.config['lr_weight_decay'])# lr=1e-5)

    if wandb.config['lr_scheduler']:
        scheduler = MultiStepLR(opt, milestones=list(range(0,wandb.config['epochs'], 100))[1:], gamma=0.5)
    else:
        scheduler = None

    # opt = torch.optim.RMSprop(model.parameters(), lr=0.001)
    if wandb.config['class_weight_type']:
        loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        print(class_weights.shape) 
    else:
        loss = torch.nn.CrossEntropyLoss()


    train(model, loss, opt, train_loader, test_loader, n_epochs=wandb.config['epochs'], scheduler=scheduler)

    torch.save(model, f"/gpfs/milgram/project/rtaylor/shared/ABDPain_EarlyDiags/models/{WANDB_RUN_NAME}.model")
    # y_pred = model(test_features)
    # top_k_accuracy_score(y_test, y_pred, labels=le.classes_, k=10)
    
    
except Exception as e:
    print(e)
finally:
    wandb.finish()
    
