"""
References: 
    1. https://github.com/pytorch/vision/blob/v0.3.0/references/detection/engine.py
    2. https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/
    3. https://pseudo-lab.github.io/Tutorial-Book-en/chapters/en/object-detection/Ch4-RetinaNet.html
"""

import matplotlib.pyplot as plt
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import pandas as pd
import time
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import numpy as np
import json

def train_one_epoch(model, data_loader, optimizer, device, lr_scheduler = None):
    """
    Trains a given model for one epoch using the provided data loader, criterion, and optimizer.

    Args:
        model (nn.Module): The model to be trained.
        data_loader (DataLoader): The data loader providing the training data.
        optimizer (torch.optim.Optimizer): The optimizer to be used for updating the model's parameters.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').
        lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler (Default: None)
        
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
        if lr_scheduler:
            lr_scheduler.step()
        running_loss += losses.item()
        progress_bar.set_description(desc=f"Loss: {losses.item():.4f}")
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

def train_and_validate_model(model, train_loader, test_loader, optimizer, num_epochs, device, lr_scheduler=None):
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
        scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler is implemented. (default is False).

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
        train_per_epoch  = train_one_epoch(model, train_loader, optimizer, device, lr_scheduler)
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
    
if __name__ == "__main__":
    from utils import *
    from models import *
    train_img_path = "../archive"
    train_annot_filename = "train"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = retinaNet(num_classes=20, device=device)
    train_transformers = get_transform(moreAugmentations=True)
    dataset = TexBigDataset(train_img_path, train_annot_filename, train_transformers)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, 
                                              collate_fn=collate_fn)
    for (i, data) in enumerate(data_loader):
        images, targets = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

def plot_multiple_losses_and_accuracies(model_data_list):
    """
    Plots training and testing losses and mAP for multiple models.

    Args:
        model_data_list (list of dict): A list of dictionaries containing the following keys:
            - 'name' (str): The name of the model (for the legend)
            - 'train_losses' (list): Training losses per epoch
            - 'test_losses' (list): Testing losses per epoch
            - 'test_mAP' (list): Testing mean average precision per epoch
    """
    fig = plt.figure(figsize=(21,15))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    for i, model_data in enumerate(model_data_list):
        train_losses = model_data.get("train_losses")
        test_losses = model_data.get("test_losses")
        test_mAP = model_data.get("test_mAP")
        line, = ax0.plot(train_losses, label=model_data.get("name")+" (Train)", linestyle = '-')
        ax0.plot(test_losses, label=model_data.get("name")+" (Test)", 
                 color = line.get_color(), linestyle = ':')
        ax1.plot(test_top1_accuracies, label=model_data.get("name"))
        ax0.legend(fontsize="14")
        ax1.legend(fontsize="14")
        ax0.set_title("Loss", fontsize=26)
        ax1.set_title("mAP", fontsize=26)
        ax0.set_xlabel("epoch", fontsize=20)
        ax0.set_ylabel("loss", fontsize=20)
        ax1.set_xlabel("epoch", fontsize=20)
        ax1.set_ylabel("mAP", fontsize=20)
        ax0.tick_params(axis='x', labelsize=18)
        ax0.tick_params(axis='y', labelsize=18)
        ax1.tick_params(axis='x', labelsize=18)
        ax1.tick_params(axis='y', labelsize=18)
    fig.suptitle("Performance for different model architectures", fontsize=36)
    plt.show()