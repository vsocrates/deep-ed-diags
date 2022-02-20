import pandas as pd
import numpy as np
import torch 

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

class AbdPainPredictionMLP(pl.LightningModule):
    def __init__(self, *args, **kwarg):
        '''Basic MLP to Training Abdominal Pain DDX Prediction
        input_dim, n_classes, config=None, loss_fn=F.MSELoss, layer_size=256, dropout=0.5
        config is a dictionary from W&B most likely
        '''
        super(AbdPainPredictionMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.fc4 = nn.Linear(layer_size, n_classes)
        
        self.bn1 = nn.BatchNorm1d(num_features=layer_size)
        self.bn2 = nn.BatchNorm1d(num_features=layer_size)
        self.bn3 = nn.BatchNorm1d(num_features=layer_size)

        self.dropout = nn.Dropout(p=dropout)
        
        self.config = config
        self.loss_fn = loss_fn
        
        self.save_hyperparameters()

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.dropout(F.relu(self.fc2(x)))
#         x = self.dropout(F.relu(self.fc3(x)))

        logits = self.fc4(x)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        data, targets = batch
        out = model(data)
        loss = self.loss_fn(out, targets)
        preds = torch.sigmoid(out).data > 0.5
        preds = preds.to(torch.float32)
#             print("in train - pred:\n", preds)
#             print("in train - out:\n", torch.sigmoid(out))
#             print("in train - target:\n", targets)
#             print("# correct?:\n", (preds==targets).sum())
#             print(loss,  "\n")

        if (data_idx % 5000 == 0):
            idxs = torch.nonzero(torch.sum(targets, 1) > 1)
#                 print(targets[idxs[0],:])
#                 print(preds[idxs[0],:])
#                 print(torch.sigmoid(out[idxs[0],:]))
#                 print(out[idxs[0],:])
            print("# correct?:\n", (preds==targets).sum())
            print(loss)
        if wandb:
            wandb.log({"train_loss": loss,
                      "train_subset_acc":torchmetrics.functional.accuracy(preds, targets.long(), subset_accuracy=True),
#                           "hamming_dist":1-torchmetrics.functional.hamming_distance(preds, targets.long()),
                       "train_precision/macro":torchmetrics.functional.precision(preds, targets.long()),
                       "train_recall/macro":torchmetrics.functional.recall(preds, targets.long()),
                       "train_precision@5":torchmetrics.functional.retrieval_precision(out, targets.long(), k=5),
                       "train_recall@5":torchmetrics.functional.retrieval_recall(out, targets.long(), k=5),

                      })


        loss.backward()
        optimizer.step()
        
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss        

    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.wandb.config['learning_rate'],
                              weight_decay = wandb.config['lr_weight_decay'])
        return opt

    def training_step(self, batch, batch_idx):
        
