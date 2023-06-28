import json
import os
import torch
import torchvision
import torchvision.transforms.v2 as transforms
torchvision.disable_beta_transforms_warning()
from torch import nn, optim
from torchvision.transforms.v2 import functional as F
from torchvision import models, datapoints
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import Dataset
import glob
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO
import numpy as np
import cv2
import pandas as pd

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
        #self.imgs_path = glob.glob(f"{self.root}/{annot_filename}/*")
        #self.imgs = [image_path.split('/')[-1] for image_path in self.imgs_path]
        #self.imgs = sorted(self.imgs)
        self.annot_path = os.path.join(self.root, f"{self.annot_filename}.json")
        self.coco = COCO(self.annot_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        #with open(self.annot_path, "r") as read_file:
        #    self.annot_data = json.load(read_file) 

    def __getitem__(self, index):
        # load images
        #img_filename = self.imgs[index]
        #matched_img = list(filter(lambda x: x["file_name"] == img_filename, self.annot_data["images"]))
        #image_id = np.unique([image["id"] for image in matched_img])
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        img_filename = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, self.annot_filename, img_filename)
        img = Image.open(img_path).convert("RGB")
        num_objs = len(coco_annotation) # number of objects in the image
        
        # get bounding box coordinates
        labels = []
        boxes = []
        iscrowd = []
        area = []
        # COCO format: [xmin, ymin, width, height]
        # PyTorch format: [xmin, ymin, xmax, ymax]
        #matched_annot_idx = []
        #annotations = self.annot_data["annotations"]
        #for i, annot in enumerate(annotations):
        #    if (annot["image_id"] in image_id):
        #        matched_annot_idx.append(i)
        #for idx in matched_annot_idx:
        for i, annot in enumerate(coco_annotation):
            xmin = annot['bbox'][0]
            ymin = annot['bbox'][1]
            xmax = xmin + annot['bbox'][2]
            ymax = ymin + annot['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(annot["category_id"])
            iscrowd.append(annot["iscrowd"])
            area.append(annot["area"])
        # convert everything into a torch.Tensor
        # store required return data in the dict
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if (len(boxes) > 0):
            boxes = datapoints.BoundingBox(boxes, 
                                       format=datapoints.BoundingBoxFormat.XYXY,
                                       spatial_size=F.get_spatial_size(img),)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            img_id = torch.tensor([img_id])
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
            area = torch.as_tensor(area, dtype=torch.float32)
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = img_id
            target["area"] = area
            target["iscrowd"] = iscrowd
        else:
            # negative example, ref: https://github.com/pytorch/vision/issues/2144
            target = {"boxes": datapoints.BoundingBox(torch.zeros((0, 4), dtype=torch.float32), 
                                       format=datapoints.BoundingBoxFormat.XYXY,
                                       spatial_size=F.get_spatial_size(img),),
                      "labels": torch.zeros(0, dtype=torch.int64),
                      "image_id": torch.tensor([img_id]),
                      "area": torch.zeros(0, dtype=torch.float32),
                      "iscrowd": torch.zeros((0,), dtype=torch.int64)}
            
        # apply the image transforms        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
    
    def get_height_and_width(self, idx):
        return self.annot_data['image'][idx]['height'], self.annot_data['image'][idx]['width']

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

def get_transform(moreAugmentations, isTransformersBackbone=False):
    """
    define the transforms for data augmentation

    Parameters
    ----------
    moreAugmentations : boolean
        If true, more data transformations are operated to avoid overfitting.
    isTransformersBackbone : boolean
        If true, resize the image to 224. Default: fase.

    Returns
    -------
    Pytorch transformers
    """
    transformList = []
    transformList.append(transforms.PILToTensor())
    transformList.append(transforms.ConvertImageDtype(torch.float32))
    if isTransformersBackbone:
        transformList.append(transforms.RandomCrop(224))
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
        result = draw_predict_bbox(boxes, pred_classes, classes, image_bgr)
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
        
    
if __name__ == "__main__":
    from engines import warmup_lr_scheduler
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
    '''
    with open(test.annot_path, "r") as read_file:
        annot_data = json.load(read_file)
    img_path = '../archive/val/14688302_1881_Seite_002.tiff'
    filename = img_path.split('/')[-1]
    image = Image.open(img_path).convert('RGB')
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    images = annot_data['images']
    annots = annot_data['annotations']
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
                (0, 255, 0), 1
            )
            cv2.putText(
                image_bgr, label, (int(bbox[0]), int(bbox[1]-5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            cv2.imshow('Ground-truth', image_bgr)
            cv2.waitKey(0)
        
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
    '''

    