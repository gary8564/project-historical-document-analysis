"""
References: 
    1. https://github.com/pytorch/vision/blob/v0.3.0/references/detection/engine.py
    2. https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/
    3. https://pseudo-lab.github.io/Tutorial-Book-en/chapters/en/object-detection/Ch4-RetinaNet.html
"""

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import pandas as pd
import time
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import numpy as np
# from utils import prepare_for_evaluation, cosine_warmup_lr_scheduler

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
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        val_loss += losses.item()
    loss_per_epoch = val_loss / len(data_loader)
    return loss_per_epoch

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
    for e in range(num_epochs):
        start = time.time()
        # train(...)
        train_per_epoch  = train_one_epoch(model, train_loader, optimizer, device, lr_scheduler)
        # validate(...)
        validate_per_epoch = validate_one_epoch(model, test_loader, device)
        end = time.time()
        print(f'\nEpoch {e+1} of {num_epochs}')
        print(f'Training Loss: {train_per_epoch:.3f} \t\t Validation Loss: {validate_per_epoch} \t\t Time: {((end - start) / 60):.3f} mins')
        train_losses.append(train_per_epoch)
        val_losses.append(validate_per_epoch)
    return train_losses, val_losses

def save_results_csv(model_name, train_losses, val_losses):
    """
    Write the training and validation results to a .csv file, enabling the creation of new plots.
    This feature is essential when running multiple training configurations on different machines, as you will need to merge the various .csv files and generate the training plots.
    Args:
        model_name (str): The name of the model (for the legend)
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        val_top1_acc (list): top-1 accuracies per epoch
        val_top5_acc (list): top-1 accuracies per epoch
    """
    result_dict = {'name': model_name, 'train_losses': train_losses, 'test_losses': val_losses}
    df = pd.DataFrame.from_dict(result_dict) 
    savepath = "%s.csv" %(model_name)
    df.to_csv (savepath, index=False, header=True)
    
def evaluate(model, data_loader, device, gt_annotFilePath):
    """
    Evaluate the test dataset. Using built-in pycocotools for evaluation. 
    Args:
        model (nn.Module)
        data_loader (DataLoader)
        device (torch.device)
        gt_annotFilePath (str): the annotation file path for the ground-truth test dataset.

    """
    cpu_device = torch.device("cpu")
    model.eval()
    coco_gt = COCO(gt_annotFilePath)
    for image, targets in data_loader:
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(image)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        img_ids = list(np.unique(list(res.keys())))
        results, output_filepath = prepare_for_evaluation(res)
        coco_dt = coco_gt.loadRes(output_filepath)
        cocoEval = COCOeval(coco_gt, coco_dt, 'bbox') 
        cocoEval.params.imgIds = img_ids 
        cocoEval.evaluate() 
    cocoEval.accumulate()
    cocoEval.summarize()
        

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
