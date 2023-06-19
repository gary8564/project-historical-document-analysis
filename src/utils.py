import json
import os
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch import nn, optim
torchvision.disable_beta_transforms_warning()
from torchvision.transforms.v2 import functional as F
from torchvision import models, datapoints
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import Dataset
import glob
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup

class TexBigDataset(Dataset):
    """
    Define a custom dataset. The dataset should inherit from the standard 
    torch.utils.data.Dataset class, and implement `__len__` and `__getitem__`.
    
    __getitem__ is required to return:
        * image: a PIL Image of size (H, W)
        * target: a dict containing the following fields:
            * `boxes` (`FloatTensor[N, 4]`): the coordinates of the N bounding boxes
               in `[x0, y0, x1, y1]` format, ranging from 0 to W and 0 to H.
            * `labels` (`Int64Tensor[N]`): the label for each bounding box
            * `image_id` (`Int64Tensor[1]`): an image identifier. It should be unique
               between all the images in the dataset, and is used during evaluation.
            * `area` (`Tensor[N]`): The area of the bounding box. This is used 
               during evaluation with the COCO metric, to separate the metric scores
               between small, medium and large boxes.
            * `iscrowd` (`UInt8Tensor[N]`): instances with `iscrowd=True` will be 
               ignored during evaluation.
            * (optionally) `masks` (`UInt8Tensor[N, H, W]`): The segmentation masks 
              for each one of the objects
            * (optionally) `keypoints` (`FloatTensor[N, K, 3]`): `K` keypoints in 
              `[x, y, visibility]` format, defining the object.
    Note that the model considers class 0 as background. 
    
    To use aspect ratio grouping during training (so that each batch only contains
    images with similar aspect ratio) and meanwhile expedite image loading in memory,
    `get_height_and_width` is additionally implemented.
    
    References: 
        1. https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        2. https://github.com/pytorch/vision/tree/v0.3.0/references/detection
        3. https://pytorch.org/blog/extending-torchvisions-transforms-to-object-detection-segmentation-and-video-tasks/
        4. https://blog.paperspace.com/data-augmentation-for-object-detection-building-input-pipelines/
    """
    def __init__(self, root, annot_filename, transforms=None):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.annot_filename = annot_filename
        self.imgs_path = glob.glob(f"{self.root}/{annot_filename}/*")
        self.imgs = [image_path.split('/')[-1] for image_path in self.imgs_path]
        self.imgs = sorted(self.imgs)
        self.annot_path = os.path.join(self.root, f"{self.annot_filename}.json")
        with open(self.annot_path, "r") as read_file:
           self.annot_data = json.load(read_file) 

    def __getitem__(self, index):
        # load images
        img_filename = self.imgs[index]
        img_path = os.path.join(self.root, self.annot_filename, img_filename)
        img = Image.open(img_path).convert("RGB")
        
        # get bounding box coordinates
        labels = []
        boxes = []
        iscrowd = []
        area = []
        matched_img = list(filter(lambda x: x["file_name"] == img_filename, self.annot_data["images"]))
        image_id = np.unique([image["id"] for image in matched_img])
        # COCO format: [xmin, ymin, width, height]
        # PyTorch format: [xmin, ymin, xmax, ymax]
        matched_annot_idx = []
        annotations = self.annot_data["annotations"]
        for i, annot in enumerate(annotations):
            if (annot["image_id"] in image_id):
                matched_annot_idx.append(i)
        for idx in matched_annot_idx:
            xmin = annotations[idx]['bbox'][0]
            ymin = annotations[idx]['bbox'][1]
            xmax = xmin + annotations[idx]['bbox'][2]
            ymax = ymin + annotations[idx]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(annotations[idx]["category_id"])
            iscrowd.append(annotations[idx]["iscrowd"])
            area.append(annotations[idx]["area"])
        # convert everything into a torch.Tensor
        # store required return data in the dict
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if (len(boxes) > 0):
            boxes = datapoints.BoundingBox(boxes, 
                                       format=datapoints.BoundingBoxFormat.XYXY,
                                       spatial_size=F.get_spatial_size(img),)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor(image_id)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
            area = torch.as_tensor(area, dtype=torch.float32)
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = torch.tensor(image_id)
            target["area"] = area
            target["iscrowd"] = iscrowd
        else:
            # negative example, ref: https://github.com/pytorch/vision/issues/2144
            target = {"boxes": datapoints.BoundingBox(torch.zeros((0, 4), dtype=torch.float32), 
                                       format=datapoints.BoundingBoxFormat.XYXY,
                                       spatial_size=F.get_spatial_size(img),),
                      "labels": torch.zeros(0, dtype=torch.int64),
                      "image_id": torch.tensor(image_id),
                      "area": torch.zeros(0, dtype=torch.float32),
                      "iscrowd": torch.zeros((0,), dtype=torch.int64)}
            
        # apply the image transforms        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
    def get_height_and_width(self, idx):
        return self.annot_data['image'][idx]['height'], self.annot_data['image'][idx]['width']


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

def cosine_warmup_lr_scheduler(optimizer, warmup_iters, total_iters, initial_lr, warmup_initial_lr):
    """
    Define the custom learning rate sheduler with a warm-up.
    The currently most popular scheduler is the cosine warm-up scheduler, 
    which combines warm-up with a cosine-shaped learning rate decay.
    References: 
        1. https://drive.google.com/drive/folders/1VtJF-zPbXc-V-UDl2bDgWJp05DnKZpQH
        2. https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863
        3. https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
        4. https://github.com/developer0hye/Learning-Rate-WarmUp
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_iters: number of iterations for warm-up learning rate.
        total_iters: maximum number of iterations.
        initial_lr: learning rate for the initial scheduler.
        warnup_initial_lr: learning rate in the beginning of warmup.
    Return: 
        cosine warmup scheduler
    """
    lr_scheduler = create_lr_scheduler_with_warmup(
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters-warmup_iters),
        warmup_start_value=warmup_initial_lr,
        warmup_duration=warmup_iters,
        warmup_end_value=initial_lr)
    return lr_scheduler
    
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
    transformList.append(transforms.PILToTensor())
    transformList.append(transforms.ConvertImageDtype(torch.float32))
    if moreAugmentations:
        transformList.append(transforms.RandomPhotometricDistort())
        transformList.append(transforms.RandomZoomOut(
            fill=defaultdict(lambda: 0, {Image.Image: (123, 117, 104)})
        ))
        transformList.append(transforms.RandomIoUCrop())
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ColorJitter(contrast=0.5))
        transformList.append(transforms.RandomRotation([-15,15]))
        transformList.append(transforms.SanitizeBoundingBox())
    return transforms.Compose(transformList)

def show_bbox_image(data_sample):
    image, target = data_sample
    if isinstance(image, Image.Image):
        image = F.to_image_tensor(image)
    image = transforms.functional.convert_dtype(image, torch.uint8)
    annotated_img = draw_bounding_boxes(image, target["boxes"], colors = "lime", width = 3)
    fig, ax = plt.subplots()
    ax.imshow(annotated_img.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()

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
    output_filepath = "./archive/val_pred.json"
    with open(output_filepath, "w") as final:
        json.dump(coco_results, final)

    return coco_results, output_filepath

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
    
if __name__ == "__main__":
    # Loading dataset
    train_img_path = "../archive"
    train_annot_filename = "train"
    pretrained_vit_weights = models.ViT_B_16_Weights.DEFAULT 
    pretrained_vit_transforms = get_pretrained_model_transform(pretrained_vit_weights)
    train_transformers = get_transform(moreAugmentations=True)
    train_data = TexBigDataset(train_img_path, train_annot_filename)
    train_sample = train_data[0]
    image, target = train_sample
    print(type(image))
    print(type(target), list(target.keys()))
    print(type(target["boxes"]), type(target["labels"]))
    show_bbox_image(train_sample)
    torch.manual_seed(1046)
    transformed_train_data = TexBigDataset(train_img_path, train_annot_filename, train_transformers)
    transformed_train_sample = transformed_train_data[0]
    transformed_image, transformed_target = transformed_train_sample
    print(type(transformed_image))
    print(type(transformed_target), list(transformed_target.keys()))
    print(type(transformed_target["boxes"]), type(transformed_target["labels"]))
    show_bbox_image(transformed_train_sample)    
    # Plotting cosine warm-up learning rate scheduler
    epochs = list(range(2000))
    # Needed for initializing the lr scheduler
    p = nn.Parameter(torch.empty(4, 4))
    total_iters = 10000
    warmup_iters = 1000
    initial_lr = 1e-3
    warmup_initial_lr = 1e-5
    optimizer = optim.Adam([p], lr=initial_lr) 
    lr_scheduler = cosine_warmup_lr_scheduler(optimizer, warmup_iters, total_iters, initial_lr, warmup_initial_lr)
    sns.set()
    x = [] 
    y = [] 
    for i in range(total_iters): 
        lr_scheduler(None)
        x.append(i) 
        y.append(optimizer.param_groups[0]['lr']) 
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.ylabel("Learning rate")
    plt.xlabel("Iterations (in batches)")
    plt.title("Cosine Warm-up Learning Rate Scheduler")
    plt.show()
    sns.reset_orig()
    

    