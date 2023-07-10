import torch
from torch import optim
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import pandas as pd
import numpy as np
import time

"""
References: 
    1. https://github.com/pytorch/vision/blob/v0.3.0/references/detection/engine.py
    2. https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/
    3. https://pseudo-lab.github.io/Tutorial-Book-en/chapters/en/object-detection/Ch4-RetinaNet.html
"""

def train_one_epoch(model, data_loader, optimizer, device, warmup_scheduler = None):
    """
    Trains a given model for one epoch using the provided data loader, criterion, and optimizer.

    Args:
        model (nn.Module): The model to be trained.
        data_loader (DataLoader): The data loader providing the training data.
        optimizer (torch.optim.Optimizer): The optimizer to be used for updating the model's parameters.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').
        warmup_scheduler (torch.optim.lr_scheduler): warm-up learning rate scheduler (Default: None)
        
    Returns:
        loss_per_epoch(float): The average loss per batch for the entire epoch.
        time_per_epoch(float): The training time for the entire epoch.
    """
    model.train(True)
    running_loss = 0  
    progress_bar = tqdm(data_loader, total=len(data_loader))      
    for (i, data) in enumerate(progress_bar):
        images, targets = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        running_loss += losses.item()
        progress_bar.set_description(desc=f"Loss: {losses.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.4f}")
    loss_per_epoch = running_loss / len(data_loader)
    return loss_per_epoch


def validate_one_epoch(model, data_loader, device):
    """
    Tests a given model for one epoch using the provided data loader and criterion.

    Args:
        model (nn.Module): The model to be tested.
        data_loader (DataLoader): The data loader providing the testing data.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').

    Returns:
        float: The average loss per batch for the entire epoch.
    """
    val_loss = 0
    target = []
    predict = []
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        val_loss += losses.item()
        model.eval()
        outputs = model(images, targets)
        # For mAP calculation using Torchmetrics.
        for i in range(len(images)):
            true_dict = dict()
            pred_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            pred_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            pred_dict['scores'] = outputs[i]['scores'].detach().cpu()
            pred_dict['labels'] = outputs[i]['labels'].detach().cpu()
            predict.append(pred_dict)
            target.append(true_dict)
        model.train()
    metric = MeanAveragePrecision()
    metric.update(predict, target)
    metric_summary = metric.compute()
    loss_per_epoch = val_loss / len(data_loader)
    return loss_per_epoch, metric_summary

def train_and_validate_model(model, train_loader, test_loader, optimizer, num_epochs, device, warmup = False, lr_scheduler=None):
    """
    Trains a given model for a specified number of epochs using the provided data loader, criterion,
    and optimizer, and tracks the loss for each epoch.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The data loader providing the training data.
        test_loader (DataLoader): The data loader providing the testing data.
        optimizer (torch.optim.Optimizer): The optimizer to be used for updating the model's parameters.
        num_epochs (int): The number of epochs to train the model.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').
        warmup (boolean): whether to activate the warm-up learning rate scheduler (default is False).
        lr_scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler is implemented. (default is False).

    Returns:
        list: A list of the average train loss per batch for each epoch.
        list: A list of the average validation loss per batch for each epoch.
    """
    train_losses = []
    val_losses = []
    mAP_50 = []
    mAP = []
    for e in range(num_epochs):
        start = time.time()
        # train(...)
        warmup_scheduler = None
        if e == 0 and warmup:
            warmup_iters = min(1000, len(train_loader) - 1)
            warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters)
        train_per_epoch  = train_one_epoch(model, train_loader, optimizer, device, warmup_scheduler)
        # update the learning rate if scheduler is implemented
        if lr_scheduler:
            lr_scheduler.step()
        # validate(...)
        validate_per_epoch, metric_summary = validate_one_epoch(model, test_loader, device)
        end = time.time()
        print(f'\nEpoch {e+1} of {num_epochs}')
        print(f"Training Loss: {train_per_epoch:.3f} \t\t Validation Loss: {validate_per_epoch} \
              \t\t mAP: {metric_summary['map']} \t\t Time: {((end - start) / 60):.3f} mins")
        train_losses.append(train_per_epoch)
        val_losses.append(validate_per_epoch)
        mAP_50.append(metric_summary['map_50'])
        mAP.append(metric_summary['map'])
    return train_losses, val_losses, mAP_50, mAP

def save_results_csv(model_name, train_losses, val_losses, mAP):
    """
    Write the training and validation results to a .csv file, enabling the creation of new plots.
    This feature is essential when running multiple training configurations on different machines, as you will need to merge the various .csv files and generate the training plots.
    Args:
        model_name (str): The name of the model (for the legend)
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        mAP (list): Validation mean average precision per epoch
    """
    result_dict = {'name': model_name, 'train_losses': train_losses, 
                   'test_losses': val_losses, 'mAP': mAP}
    df = pd.DataFrame.from_dict(result_dict) 
    savepath = "%s.csv" %(model_name)
    df.to_csv (savepath, index=False, header=True)
    
def warmup_lr_scheduler(optimizer, warmup_iters):
    """
    Define a warm-up learning rate sheduler
    References: 
        1. https://drive.google.com/drive/folders/1VtJF-zPbXc-V-UDl2bDgWJp05DnKZpQH
        2. https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863
        3. https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
        4. https://github.com/developer0hye/Learning-Rate-WarmUp
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_iters: number of iterations for warm-up learning rate.
    Return: 
        custom warmup scheduler
    """
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-03, total_iters=warmup_iters)
    return warmup_scheduler


