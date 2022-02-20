import pandas as pd
import numpy as np
import torch 

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class AbdPainPredictionMLP(nn.Module):
    def __init__(self, input_dim, n_classes, layer_size=256, dropout=0.5):
        super(AbdPainPredictionMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.fc4 = nn.Linear(layer_size, n_classes)
        
        self.bn1 = nn.BatchNorm1d(num_features=layer_size)
        self.bn2 = nn.BatchNorm1d(num_features=layer_size)
        self.bn3 = nn.BatchNorm1d(num_features=layer_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.dropout(F.relu(self.fc2(x)))
#         x = self.dropout(F.relu(self.fc3(x)))

        logits = self.fc4(x)
        
        return logits
