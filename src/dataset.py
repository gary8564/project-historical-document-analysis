import os
import torch
from torchvision.transforms.v2 import functional as F
from torchvision import datapoints
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO


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
        5. https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
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
        