import pandas as pd
import numpy as np
import torch 

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class AbdPainPredictionMLP(nn.Module):
    def __init__(self, n_classes):
        super(AbdPainPredictionMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4083, 1024),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(p=0.5), 
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )

        # self.bert_embedding = 

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
