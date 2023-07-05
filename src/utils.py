import json
import torch
import torchvision
import torchvision.transforms.v2 as transforms
torchvision.disable_beta_transforms_warning()
from torch import nn, optim
from torchvision.transforms.v2 import functional as F
from torchvision import models
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import pandas as pd

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))
    
# Data augmentation: image transformation
def get_pretrained_model_transform(pretrained_model_weights):
    """
    get the transformers from pretrained model to ensure custom data is 
    transformed/formatted in the same way the data the original model was 
    trained on.

    Parameters: 
    ----------
    pretrained_model_weights : 
        torchvision.models.pretrained_model_weights.

    Returns
    -------
    pretrained model transformers
    """
    pretrained_model_transforms= pretrained_model_weights.transforms()
    return pretrained_model_transforms

def get_transform(moreAugmentations):
    """
    define the transforms for data augmentation

    Parameters
    ----------
    moreAugmentations : boolean
        If true, more data transformations are operated to avoid overfitting.
    
    Returns
    -------
    Pytorch transformers
    """
    transformList = []
    if moreAugmentations:
        #transformList.append(transforms.RandomPhotometricDistort())
        #transformList.append(transforms.RandomZoomOut(
        #    fill=defaultdict(lambda: 0, {Image.Image: (123, 117, 104)})
        #))
        #transformList.append(transforms.RandomIoUCrop())
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ColorJitter(contrast=0.5))
        #transformList.append(transforms.RandomRotation([-15,15]))
        transformList.append(transforms.SanitizeBoundingBox())
    transformList.append(transforms.PILToTensor())
    transformList.append(transforms.ConvertImageDtype(torch.float32))
    return transforms.Compose(transformList)

def show_bbox_image(data_sample):
    image, target = data_sample
    if isinstance(image, Image.Image):
        image = F.to_image_tensor(image)
    image = transforms.functional.convert_dtype(image, torch.uint8)
    annotated_img = draw_bounding_boxes(image, target["boxes"], colors = "lime", width = 5)
    fig, ax = plt.subplots()
    ax.imshow(annotated_img.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()

def draw_predict_bbox(boxes, pred_classes, gt_classes, image):
    COLORS = np.random.uniform(0, 255, size=(len(gt_classes), 3))
    for i, box in enumerate(boxes):
        color = COLORS[gt_classes.index(pred_classes[i])]
        cv2.rectangle(image, 
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      color, 2
        )
        cv2.putText(image, pred_classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image

def prepare_for_evaluation(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue
        boxes = prediction["boxes"]
        # convert (xmin, ymin, xmax, ymax) back to COCO format (x, y, w, h)
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        boxes = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()
        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

def predict(image, model, device, detection_threshold, classes, save_filename):
    # Create a BGR copy of the image for annotation.
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # transform the image to tensor
    test_transform = get_transform(moreAugmentations=False)
    image = test_transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    with torch.no_grad():
        outputs = model(image) # get the predictions on the image
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        # get all the scores
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]
        result = draw_predict_bbox(draw_boxes, pred_classes, classes, image_bgr)
        cv2.imshow('Prediction', result)
        cv2.waitKey(10) 
        cv2.imwrite(f"../images/{save_filename}.png", result)
        print("Predicted image saved!")

def load_pretrained_weights(model, weights_path, device):
    """
    Loads pretrained weights into the specified model and returns the model with the loaded weights.

    Args:
        model (nn.Module)
        weights_path (str or pathlib.Path): The path to the file containing the pretrained weights.
        classes (int): The number of classes in the original classification task.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').
        new_classes (int): The number of classes in the new classification task.

    Returns:
        model (nn.Module): The model with the pretrained weights loaded.

    Raises:
        AssertionError: If the model_name is not in ["SimpleConvNet", "TwoLayerFullyConnectedNetwork"].
    """
    current_model_dict = model.state_dict()
    loaded_state_dict = torch.load(weights_path, map_location=device)
    new_state_dict = {k: v if v.size() == current_model_dict[k].size() 
                    else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    return model

def freeze_layers(model, frozen_layers):
    """
    Freezes the specified layers of a neural network model. Freezing a layer means that its
    parameters will not be updated during training.

    Args:
        model (nn.Module): The neural network model.
        frozen_layers (list): A list of layer names to be frozen.
    """
    for name, param in model.named_parameters():
        if param.requires_grad and any(layer in name for layer in frozen_layers):
            param.requires_grad = False
    print('After frozen, require grad parameter names:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    return model

def aspectRatioStats(annot_filename):
    """
    Compute the statistical information of the distribution of bounding boxes' aspect ratios in the dataset
    
    Args:
        annot_filename (str): jsonl fileaname of annotation in COCO format
    Return:
        statistical summary of widths, heighst, and aspect ratios.
        histogram plot.
    """
    with open(annot_filename, "r") as read_file:
        annot_data = json.load(read_file)
    annots = annot_data['annotations']
    ar_dict = {"width": [], "height": [], "aspect_ratio": []}
    for i, annot in enumerate(annots):
        bbox = annot['bbox'] 
        w = bbox[2]
        h = bbox[3]
        ar = h / w
        ar_dict["width"].append(w)
        ar_dict["height"].append(h)
        ar_dict["aspect_ratio"].append(ar)
    df = pd.DataFrame.from_dict(ar_dict)
    df_ar = df["aspect_ratio"]
    df_ar_adjusted = df_ar[df_ar.between(df_ar.quantile(.25), df_ar.quantile(.75))]
    sns.set()
    plt.figure()
    plt.hist(df_ar_adjusted, bins=20)
    plt.title("Distribution of aspect ratios of bounding boxes between Q1 and Q3")
    plt.xlabel("aspect ratio")
    plt.show()
    sns.reset_orig()
    return df_ar.describe()

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
        test_mAP = model_data.get("mAP")
        line, = ax0.plot(train_losses, label=model_data.get("name")+" (Train)", linestyle = '-')
        ax0.plot(test_losses, label=model_data.get("name")+" (Test)", 
                 color = line.get_color(), linestyle = ':')
        ax1.plot(test_mAP, label=model_data.get("name"))
        ax0.legend(fontsize="14")
        ax1.legend(fontsize="14")
        ax0.set_title("Loss", fontsize=26)
        ax1.set_title("Validation mAP", fontsize=26)
        ax0.set_xlabel("epoch", fontsize=20)
        ax0.set_ylabel("loss", fontsize=20)
        ax1.set_xlabel("epoch", fontsize=20)
        ax1.set_ylabel("mAP", fontsize=20)
        ax0.tick_params(axis='x', labelsize=18)
        ax0.tick_params(axis='y', labelsize=18)
        ax1.tick_params(axis='x', labelsize=18)
        ax1.tick_params(axis='y', labelsize=18)
    fig.suptitle("Performance for different model configurations", fontsize=36)
    plt.show()
        
    
if __name__ == "__main__":
    from engines import warmup_lr_scheduler
    from dataset import *
    # Loading dataset
    train_img_path = "../archive"
    train_annot_filename = "train"
    pretrained_vit_weights = models.ViT_B_16_Weights.DEFAULT 
    pretrained_vit_transforms = get_pretrained_model_transform(pretrained_vit_weights)
    train_transformers = get_transform(moreAugmentations=True)
    train_data = TexBigDataset(train_img_path, train_annot_filename)
    train_sample = train_data[50]
    image, target = train_sample
    print(type(image))
    print(type(target), list(target.keys()))
    print(type(target["boxes"]), type(target["labels"]))
    show_bbox_image(train_sample)
    torch.manual_seed(1046)
    transformed_train_data = TexBigDataset(train_img_path, train_annot_filename, train_transformers)
    transformed_train_sample = transformed_train_data[50]
    transformed_image, transformed_target = transformed_train_sample
    print(type(transformed_image))
    print(type(transformed_target), list(transformed_target.keys()))
    print(type(transformed_target["boxes"]), type(transformed_target["labels"]))
    show_bbox_image(transformed_train_sample)
    
    # Test data bounding box
    CLASSES = [
        'background', 'Advertisement', 'Author', 'Caption', 'Column title', 'Decoration', 
        'Editorial note', 'Equation', 'Footer', 'Footnote', 'Frame', 'Header', 'Heading', 
        'Image', 'Logo', 'Noise', 'Page Number', 'Paragraph', 'Sub-heading', 'Table'
    ]
    annot_filename = 'val'
    test = TexBigDataset(train_img_path, annot_filename)
    summary = aspectRatioStats(test.annot_path)
    print(summary)
    
    with open(test.annot_path, "r") as read_file:
        annot_data = json.load(read_file)
    img_path = '../archive/val/14688302_1881_Seite_002.tiff'
    filename = img_path.split('/')[-1]
    image = Image.open(img_path).convert('RGB')
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    images = annot_data['images']
    annots = annot_data['annotations']
    save_filename = 'ground-truth'
    index = 0
    for i, img in enumerate(images):
        if (img['file_name'] == filename):
            index = img['id']
    for i, annot in enumerate(annots):
        if (annot['image_id'] == index):
            bbox = annot['bbox'] 
            label = CLASSES[annot['category_id']]
            cv2.rectangle(
                image_bgr, 
                (int(bbox[0]), int(bbox[1])), (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])),
                (0, 255, 0), 2
            )
            cv2.putText(
                image_bgr, label, (int(bbox[0]), int(bbox[1]-5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
    cv2.imshow('Ground-truth', image_bgr)
    cv2.waitKey(0)
    cv2.imwrite(f"../images/{save_filename}.png", image_bgr)
    print("Ground-truth image saved!")
            
        
    # Plotting cosine warm-up learning rate scheduler
    # Needed for initializing the lr scheduler
    p = nn.Parameter(torch.empty(4, 4))
    epochs = 20
    batches = 1000
    total_iters = epochs * batches
    #optimizer = optim.Adam([p], lr=initial_lr) 
    optimizer = optim.SGD([p], lr=1e-03,
                          momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.75)
    sns.set()
    x = [] 
    y = []
    iters = 0
    for i in range(epochs): 
        scheduler = None
        if (i == 0):
            warmup_iters = 1000
            scheduler = warmup_lr_scheduler(optimizer, warmup_iters)
        for j in range(batches):
            iters += 1
            optimizer.step() 
            x.append(iters) 
            y.append(optimizer.param_groups[0]['lr'])
            if scheduler is not None:
                scheduler.step()
        lr_scheduler.step()
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.ylabel("Learning rate")
    plt.xlabel("Iterations")
    plt.title("Linearly Warm-up Step Decay Learning Rate Scheduler")
    plt.show()
    sns.reset_orig()
    

    