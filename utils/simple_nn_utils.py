import numpy as np
import pandas as pd
from collections import Counter
import torch
from sklearn.metrics import *
import torchmetrics
import wandb
from torchmetrics.functional import hamming_distance



################
## Functions ###
################
def _effective_num_weighting(beta, samples_per_cls, no_of_classes):
    '''
    Compute the effective number sample weighting as per https://doi.org/10.1109/CVPR.2019.00949

    beta [float] : usually between 0.9 and 0.99, and is the hyperparam for the amount of weighting like inverse freq
    samples_per_cls [list-like] : a list-like object with the number of samples per class that has length no_of_classes
    no_of_classes [int] : the number of classes
    '''
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    return weights

def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]


def flatten(t):
    return [item for sublist in t for item in sublist]


def get_class_weights(labels, wandb, n_samples, label_list=None):
    '''Returns class weights given a list of labels and the label_list we want the 
    order to be in. 
    
    Also takes in the wandb config
    '''
    samples_per_cls = pd.Series(Counter(flatten(labels)))
    if label_list:
        samples_per_cls = samples_per_cls.reindex(label_list)
        
        # make sure we don't have any labels not in label_list
        assert(samples_per_cls.isna().sum() == 0)

    if wandb.config['class_weight_type'] == "effective_sample":
        beta = wandb.config['weight_beta']
        no_of_classes = samples_per_cls.size
        class_weights = pd.Series(_effective_num_weighting(beta, samples_per_cls, no_of_classes),
                            index=samples_per_cls.index).values
    
    elif wandb.config['class_weight_type'] == "balanced":
        class_weights = n_samples/(len(samples_per_cls) * samples_per_cls).values
    elif wandb.config['class_weight_type'] == "inverse":
        # multiply by total/2 as per tensorflow core example imbalanced classes
        class_weights = (1 / samples_per_cls) * (samples_per_cls.sum() / wandb.config['class_weight_inv_lambda']).values
        
    elif wandb.config['class_weight_type'] == "constant":
        class_weights = np.full((1, len(samples_per_cls)), wandb.config['constant_weight'])
    elif wandb.config['class_weight_type'] == "bce_weights":
        class_weights = ((samples_per_cls.sum() - samples_per_cls) / (samples_per_cls)).values
    else:
        class_weights = None
    
    return class_weights
    
    
    
# Use threshold to define predicted labels and invoke sklearn's metrics with different averaging strategies.
def calculate_metrics(pred, target, threshold=0.5):
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro', zero_division=0.0),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro', zero_division=0.0),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro', zero_division=0.0),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro', zero_division=0.0),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro', zero_division=0.0),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro', zero_division=0.0),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples', zero_division=0.0),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples', zero_division=0.0),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples', zero_division=0.0),
            }




def multilabel_evaluate(model, loss_fn, evaluation_set, device, wandb=None):
    """
    Evaluates the given model on the given dataset.
    Returns the percentage of correct classifications out of total classifications.
    """
    correct = 0
    total = 0
    losses = 0
    # TODO: ValueError: You can not use the `top_k` parameter to calculate accuracy for multi-label inputs.
#     avg_topk_acc = []
    model.eval()
    with torch.no_grad():
        for data, targets in evaluation_set:
            targets = targets.to(device)
            out = model(data.to(device))
            loss = loss_fn(out, targets)
#             wandb.log({"train_loss": loss})
            loss += loss.item()
            preds = torch.sigmoid(out).data > 0.5
            preds = preds.to(torch.float32)

            total += torch.numel(targets)
#             avg_topk_acc.append(torchmetrics.functional.accuracy(preds, targets.long(), top_k=5))
            
            
            correct += (preds == targets).sum().item()
            # TODO: create a PR plot for the multilabel case
#             wandb.log({"pr" : wandb.plot.pr_curve(targets.cpu(), preds.data.cpu(),
#                      labels=label_freqs.index.tolist()
#                                                  )})        
        
        accuracy = (correct/total)
        if wandb:
            wandb.log({"validation_loss": loss/len(evaluation_set)})
            wandb.log({"validation_acc":accuracy})
    #         wandb.log({"top5_validation_acc":sum(avg_topk_acc)/len(avg_topk_acc)}) 
        
        
    return accuracy

def multilabel_train(model, loss_fn, optimizer, train_loader, test_loader, device, wandb=None,
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
    model.train()
    for epoch in range(n_epochs):
        for data_idx, (data, targets) in enumerate(train_loader):
                        
            optimizer.zero_grad()
            targets = targets.to(device)
            out = model(data.to(device))
            loss = loss_fn(out, targets)
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
            

        if scheduler:
            scheduler.step()
            if wandb:            
                wandb.log({"lr": scheduler.get_lr()[0]})

        if wandb:
            if epoch % wandb.config['eval_freq'] == 0:
                print(f" EPOCH {epoch}. Progress: {epoch/n_epochs*100}%. ", flush=True)
                test_acc = multilabel_evaluate(model, loss_fn, test_loader, device, wandb)
#                 print(f"Test accuracy: {test_acc}", flush=True) 
                model.train()
        else:
            # only used in synthetic test dataset
            print(f" EPOCH {epoch}. Progress: {epoch/n_epochs*100}%. ", flush=True)
            test_acc = multilabel_evaluate(model, loss_fn, test_loader, device, wandb)
            
    print(f" EPOCH {n_epochs}. Progress: 100%. ", flush=True)
    print(f" Train accuracy: {multilabel_evaluate(model,loss_fn, train_loader, device, wandb)}. Test accuracy: {multilabel_evaluate(model,loss_fn, test_loader, device, wandb)}", flush=True) 

