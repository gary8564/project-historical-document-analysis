[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/7EW3yjxG)
# An Evaluation of RetinaNet on TexBig dataset
An ablation study on RetinaNet for historical document layout analysis

## Introduction
In the realm of document layouts analysis, the results of previous works<sup>[[1]](#1)</sup> <sup>[[2]](#2)</sup> have pointed out that CV-based approaches performs better than NLP-based approaches.
This project tries to implement a state-of-the-art and efficient object detector to document layout analysis on [TexBig dataset](https://zenodo.org/record/6885144) with the consideration of training constraints on Kaggle notebook. 

## Table of Contents
* [Problem Statement](#problem-statement)
* [Proposed Solutions](#proposed-solutions)
* [Dataset Description](#dataset-description)
* [Installation](#installation)
* [Usage](#usage)
* [Code Structure](#code-structure)
* [Experiment Results](#experiment-results)
* [Discussion and Analysis](#discussion-and-analysis)
* [Outlook and Future Work](#outlook-and-future-work)
* [Citation](#citation)

## Problem Statement
Create a well-suited object detection model for TexBig dataset, a domain-specifically dataset for historical document layout analysis under the constraints of training a batch size of at least 2 on a NVIDIA Tesla P100 with 12 GB VRAM within an 9-hour window.

## Proposed Solutions
Considering the constraints of runtime sessions and the limited GPU resources, a [pretrained RetinaNet](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py)<sup>[[3]](#3)</sup> model provided in [Pytorch model zoo](https://pytorch.org/vision/stable/models.html#) is the candidate of my baseline approach. 
Based on the textual features of the dataset, a [pretrained ViT](https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py)<sup>[[4]](#4)</sup> and [SwinT](https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py)<sup>[[5]](#5)</sup> are chosen to be the backbones of RetinaNet.

## Dataset Description
TexBig<sup>[[6]](#6)</sup> is a high-quality document layouts dataset in historical digital humanities domain. The dataset provides fine-grained annotations with 19 classes as well as instance segmentation level ground truth data in COCO format. 

It has 1922 training samples and 335 validation samples. Mean average precision(mAP) in the validation samples are used for the evaluation metric in ablation study of different model configurations.

To generalize the validation performance, data argumentation by using [Pytorch Transforms v2 library](https://pytorch.org/vision/main/auto_examples/plot_transforms_v2_e2e.html) is experimented in this study. Further details of the data augmentation will be discussed in section [Experiment Results](#experiment-results) and [Discussion and Analysis](#discussion-and-analysis). The visualization example of the ground-truth bounding boxes from one of the training samples is shown below:
| Without image transformations  |  After image transformations |
|:------------------------------:|:----------------------------:|
|![](/Users/kyle_lee/Desktop/Bauhaus/DL4CV/final-project-gary8564/images/ground-truth-no-transform.png)  |  ![](/Users/kyle_lee/Desktop/Bauhaus/DL4CV/final-project-gary8564/images/ground-truth-transforms.png)|

To fine-tune the aspect ratios of anchor boxes, the distribution of the aspect ratios of all annotated bounding boxes in the dataset is analyzed as below:
|               | aspect ratio  | 
|:-------------:|:-------------:| 
| mean          | 4.21          | 
| std           | 16.96         |  
| min           | 0.0044        |
| max           | 233.18        |
| 25%           | 0.092         |
| 50%           | 0.24          |
| 75%           | 0.86          |

![texBig_aspectRatioHist](https://github.com/BUW-CV/final-project-gary8564/assets/54540162/d56d722e-79ec-42d1-b3d8-1499ad4460bd)

From the above analysis, most of the aspect ratios are in the range of 0.1 to 0.2. Therefore, in the later fine-tuning experiments, the smaller aspect ratios of anchor boxes are targeted. 

## Installation
Install the repository in editable mode. Example for MacOS/Linux(Ubuntu):

    python3 -m venv dlcv_final_project  
    source dlcv_final_project/bin/activate  
    pip install torch torchvision
    pip install cython
    pip install pycocotools
    pip install torchmetrics
    pip install -e .  


## Usage
(a) Training
Run the training script from [Kaggle notebook]() or open your terminal/command line from in the `src` directory and execute the following command: 

```
python train.py --datapath ... --batchsize ... --epochs ... --modelname ... --frozen ... --scheduler ... --warmup ...
```
* `--datapath` is to input the root folder location where the TexBig dataset is stored.
* `-batchsize` is to input the desired batch size. It is an optional input argument. By default, it is set to 2.
* `--epochs` is to input the desired epochs. It is an optional input argument. By default, it is set to 10.
* `--modelname` is to input the model name for training. It is an optional and limited input argument. Choices of the model name are "baseline", "EfficientNetFPN", "ResNeXT101FPN", "SwinTFPN", "SwinT", and "ViT". By default, it is set to "ResNeXT101FPN".
* `--frozen` is to input the desired frozen layer names. It is an optional input argument. By default, it is set to None.
* `--scheduler` is to input whether to activate the StepLR learning rate scheduler (True/False). It is an optional input argument. By default, it is set to False.
* `--warmup` is to input whether to activate the warmup learning rate (True/False). It is an optional input argument. By default, it is set to True.  

(b) Testing
Run the testing script to output the prediction results as a json file in [Kaggle notebook]() or open your terminal/command line from in the `src` directory and execute the following command:
```
python test.py --backbone ... --weights ...
```
* `--backbone` is to input the desired backbone model name. Choices are limited to 'baseline', 'EfficientNetFPN', and 'ResNeXT101FPN'. It is an optional input argument. By default, it is set to 'ResNeXT101FPN'.
* `--weights` is to input the trained model weights path.

(c) Inference:
To run inference of the trained models on a new data, open your terminal/command line from in the `src` directory and execute the following command: 
```
python inference.py --input ... --threshold ... --model ... --weights ...
```
* `--input` is to input the location of the new image data. 
* `--threshold` is to input the minimum confidence score for detection. It is an optional input argument. By default, it is set to 0.5.
* `--model` is to input the desired backbone model name. Choices are limited to 'baseline', 'EfficientNetFPN', 'ResNeXT101FPN'. It is an optional input argument. By default, it is set to 'ResNeXT101FPN'.
* `--weights` is to input the trained model weights path. 

## Code Structure
This project is built by using Setuptools together with `pyproject.toml`. The codebase directory structure of this project is as follows:

```
final-project-gary8564/
├── src
│   ├── __init__.py
│   ├── utils.py
│   ├── engines.py
│   ├── models.py
│   ├── dataset.py
│   └── train.py
│   ├── test.py
│   └── inference.py 
├── pyproject.toml
├── setup.py
├── README.md
└── .gitignore
```
All of the source code for this project can be found in `src` folder:
  1. `utils.py` contains all of the utility code and helper functions needed in this project.
  2. `engines.py` contains code blocks supported for training.
  3. `models.py` creates RetinaNet model and pretrained backbone models.
  4. `dataset.py` creates the custom Dataset class
  5. `train.py` contains the executable training script.
  6. `test.py` contains the executable test script to test a trained model version with corresponding 
      weights on a validation or test dataset and output the object detection results in a json file.
  7. `inference.py` contains executable test inference of the trained models on a new data.

## Experiment Results
### 1. Fine-tuning the baseline model - [pretrained RetinaNet](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html#torchvision.models.detection.retinanet_resnet50_fpn_v2) 
#### Grid search hyperparameters of batch size, learning rate, and anchor boxes
Warm-up StepLR scheduler first linearly increases the learning rate from initial learning rate of 0.0005 to 0.001 in the first 1000 iterations. After 1000 iterations, the learning rate decays by 0.75 after every 5 epochs. The visualization of the warm-up StepLR scheduler is shown as below:
![lr_scheduler](/Users/kyle_lee/Desktop/Bauhaus/DL4CV/final-project-gary8564/images/learning rate scheduler.png)
Three different configurations are considered:
* batch size = 4; optimizer = SGD; warm-up SetpLR scheduler
* batch size = 2; optimizer = SGD; change parameters of anchor boxes
* batch size = 2; optimizer = SGD; ; warm-up SetpLR scheduler; change parameters of anchor boxes
The comparison results of three model configurations is shown as below:

(1) Training and validation loss history 

![fine-tune comparison](/Users/kyle_lee/Desktop/Bauhaus/DL4CV/final-project-gary8564/images/finetune_baseline.png)

(2) mAP

| model configs | mAP           | mAP<sub>50</sub>| mAP<sub>75</sub>| mAP<sub>s</sub>| mAP<sub>m</sub>| mAP<sub>l</sub>|
|:-------------:|:-------------:|:---------------:|:---------------:|:--------------:|:--------------:|:--------------:|
| baseline      | 0.447         | 0.655           | 0.484           | 0.334          | 0.293          | 0.423          |
| config1       | 0.454         | 0.647           | 0.478           | 0.285          | 0.268          | 0.429          |  
| config2       | 0.494         | 0.727           | 0.530           | 0.340          | 0.344          | 0.460          |
| config3       | 0.478         | 0.695           | 0.504           | 0.316          | 0.311          | 0.439          |

### 2. Ablation study
The comparison results of different pretrained baackbone models are shown as below:
(1) Training and validation loss history 
![fine-tune comparison](/Users/kyle_lee/Desktop/Bauhaus/DL4CV/final-project-gary8564/images/finetune_baseline.png)
(2) mAP
| backbones     | mAP           | mAP<sub>50</sub>| mAP<sub>75</sub>| mAP<sub>s</sub>| mAP<sub>m</sub>| mAP<sub>l</sub>|
|:-------------:|:-------------:|:---------------:|:---------------:|:--------------:|:--------------:|:--------------:|
| no feature pyramids |||||||
| [Vit](https://pytorch.org/vision/stable/models/vision_transformer.html)| 0.220| 0.324 | 0.224 | 0.004 | 0.015 | 0.236 |
| [SwinT](https://pytorch.org/vision/stable/models/swin_transformer.html)| 0.214 | 0.377 | 0.207 | 0.223| 0.145| 0.178 |
| with feature pyramids |||||||
| [SwinT](https://pytorch.org/vision/stable/models/swin_transformer.html) | 0.242 | 0.398 | 0.224 | 0.244| 0.160| 0.183 |   
| [EfficientNetV2](https://pytorch.org/vision/main/models/efficientnetv2.html)| 0.441 | 0.637 | 0.478  | 0.231 | 0.225 | 0.435  |
| [ResNeXT101](https://pytorch.org/vision/main/models/resnext.html) |0.512 | 0.693 | 0.562 | 0.313 | 0.358 | 0.494|           

### 3. Data Augmentation
From the ablation study, ResNeXT101 as backbone yields the most promising result. Therefore, in this section, only ResNeXT101 backbone is considered.
To improve the generalization, data augmentation using several image transformation techniques is implemented. In this study, RandomHorizontalFlip and ColorJitter are implemented. 
The result of mAP is shown below:
| backbone      | mAP           | mAP<sub>50</sub>| mAP<sub>75</sub>| mAP<sub>s</sub>| mAP<sub>m</sub>| mAP<sub>l</sub>|
|:-------------:|:-------------:|:---------------:|:---------------:|:--------------:|:--------------:|:--------------:|
| [ResNeXT101](https://pytorch.org/vision/main/models/resnext.html) | 0.546 |   0.775 |  0.608  | 0.341  |  0.347  |  0.528 |

#### 4. Final results
Retrain the best configuration for 8 epochs (ResNeXT101-backbone; batch size=2; SGD with learning rate=0.001; warmup StepLR scheduler)
The comparison between the prediction of the final model and the ground-truth annotations is visualized below:
| Groud-Truths  |  Predictions |
|:------------------------------:|:----------------------------:|
|![](/Users/kyle_lee/Desktop/Bauhaus/DL4CV/final-project-gary8564/images/ground-truth1.png)  |  ![](/Users/kyle_lee/Desktop/Bauhaus/DL4CV/final-project-gary8564/images/prediction1.png)|
|![](/Users/kyle_lee/Desktop/Bauhaus/DL4CV/final-project-gary8564/images/ground-truth2.png)  |  ![](/Users/kyle_lee/Desktop/Bauhaus/DL4CV/final-project-gary8564/images/prediction2.png)|
|![](/Users/kyle_lee/Desktop/Bauhaus/DL4CV/final-project-gary8564/images/ground-truth3.png)  |  ![](/Users/kyle_lee/Desktop/Bauhaus/DL4CV/final-project-gary8564/images/prediction3.png)|


## Discussion and Analysis
1. Learning rates:
If the learning rate is set above 0.005, the model tends to diverge. It’s common to use a smaller learning rate for pretrained model, in comparison to the (randomly-initialized) weights. This is because we expect that the pretrained weights are relatively good, so we don’t wish to distort them too quickly and too much. <sup>[[7]](#7)</sup>

2. Anchor boxes:
Anchor boxes are one of the most influential hyperparameters to fine-tune. This can be proved in the baseline fine-tuning stage. Since most of the data contain smaller aspect ratios, I chose to add more anchor boxes and set smaller aspect ratios. The result of mAP is surprisingly improved by 10%.

3. Optimizers:
At the first stage of fine-tuning, Adam-based optimizers such as Adam, AdamW, or RAdam are chosen as optimizers. However, during the training process, the validation loss of using Adam-based optimizers is worse than using SGD with Nesterov momentum. Numerous paper<sup>[8](#8),[9](#9),[10](#10)</sup> has also pointed out that Adam's generalization performance is worse than SGD, especially on image classification problems. More recent paper<sup>[11](#11)</sup> further clarified that fine-tuned Adam is always better than SGD, while there exists a performance gap between Adam and SGD when using default hyperparameters. Since it might be difficult to find the optimal hyperparameters and the original paper of RetinaNet<sup>[12](#12)</sup> also used SGD optimizer, I therefore focused only on SGD optimizer. 

3. Backbones:
(1) Transformers-based backbones:
In order to fit in the constraints of training capacity, most of the encoder layers are frozen. However, freezing large portion of layers also led to unpromising results. It may be difficult to learn and fit into this complex and domain-specific large dataset if freezing most parts of the model architecture. 

Even though the result is not promising, the above mAP results can still get another interesting observation: SwinT transformers have more learning capacity to detect smaller objects.

(2) ResNeXT and EfficientNet:
In order to speed up the training process, `nn.DataParallel` is utilized to fit with the Kaggle GPU-T4x2 accelerator. The above ablation study indicates that both EfficientNet and ResNeXT yield outstanding performances. Still, mAP of ResNeXT-backbone model is 
 

## Outlook and Future Work
In conclusion, despite the complexity of historical documents dataset, by fine-tuning hyperparameters and increase backbone model complexities, RetinaNet is still able to detect most of the annotations. Even though mAP on the test dataset leaderboard can only achieve 0.21, the performance can be improved by training more epochs if more powerful computing units can be accessed. More laborious fine-tuning with anchor boxes might also lead to more promising results.
In future work, unfreezing layers of ViT and SwinT backbone can be further experimented to check for the improvement of results. Future studies can also try to implement other more recent methodology such as VitDet<sup>[13](#13)</sup>, which utilized plain ViT-backbone with simple feature pyramid maps structure. In ViTDet paper, the author also points out that the results can be benefited from using the readily available pretrained transformer models from Masked Autoencoder(MAE). Therefore, using the pretrained model from MAE can also be further discussed.

## Citation
<a id="1">[[1]](https://arxiv.org/abs/2212.13924)</a> N.-M. Sven and R. Matteo, “Page layout analysis of text-heavy historical documents: A comparison of textual and visual approaches,” arXiv [cs.IR], 2022.

<a id="2">[[2]](https://arxiv.org/abs/2105.06220)</a> Zhang, P. (2021, May 13). VSR: A Unified Framework for Document Layout Analysis combining Vision, Semantics and Relations. arXiv.org.

<a id="3">[[3]](https://arxiv.org/abs/1708.02002)</a> Lin, T. (2017). Focal Loss for Dense Object Detection. arXiv.org.

<a id="4">[[4]](https://arxiv.org/abs/2010.11929)</a> Alexey Dosovitskiy, et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv.

<a id="5">[[5]](https://arxiv.org/abs/2103.14030)</a> Ze Liu Li, et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. arXiv.

<a id="6">[[6]](https://doi.org/10.1007/978-3-031-16788-1_22)</a> Tschirschwitz, D., Klemstein, F., Stein, B., Rodehorst, V. (2022). A Dataset for Analysing Complex Document Layouts in the Digital Humanities and Its Evaluation with Krippendorff’s Alpha. In: Andres, B., Bernard, F., Cremers, D., Frintrop, S., Goldlücke, B., Ihrke, I. (eds) Pattern Recognition. DAGM GCPR 2022. Lecture Notes in Computer Science, vol 13485. Springer, Cham.

<a id="7">[[7]](https://cs231n.github.io/transfer-learning/)</a> Transfer Learning. Stanford CS231n: Convolutional Neural Networks for Visual Recognition.

https://arxiv.org/pdf/1712.07628.pdf

https://opt-ml.org/papers/2021/paper53.pdf

https://arxiv.org/pdf/1509.01240.pdf

https://arxiv.org/pdf/1910.05446.pdf

https://arxiv.org/pdf/1708.02002v2.pdf

https://arxiv.org/pdf/2203.16527.pdf