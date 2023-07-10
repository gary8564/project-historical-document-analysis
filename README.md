# An Evaluation of RetinaNet on TexBig dataset
Deep Learning for Computer Vision course project: An ablation study on RetinaNet for historical document layout analysis

## Introduction
In the realm of document layouts analysis, the results of previous works<sup>[[1]](#1)</sup> <sup>[[2]](#2)</sup> have pointed out that CV-based approaches perform better than NLP-based approaches.
This project tries to implement a state-of-the-art and efficient object detector to document layout analysis on [TexBig dataset](https://zenodo.org/record/6885144) with the consideration of training constraints on the Kaggle notebook. 

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
Create a well-suited object detection model for the TexBig dataset, a domain-specifically dataset for historical document layout analysis under the constraints of training a batch size of at least 2 on an NVIDIA Tesla P100 with 12 GB VRAM within a 9-hour window.

## Proposed Solutions
Considering the constraints of runtime sessions and the limited GPU resources, a [pretrained RetinaNet](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py)<sup>[[3]](#3)</sup> model provided in [Pytorch model zoo](https://pytorch.org/vision/stable/models.html#) is the candidate of my baseline approach. 
Based on the textual features of the dataset, a [pretrained ViT](https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py)<sup>[[4]](#4)</sup> and [SwinT](https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py)<sup>[[5]](#5)</sup> are chosen to be the backbones of RetinaNet.

## Dataset Description
TexBig<sup>[[6]](#6)</sup> is a high-quality document layout dataset in the historical digital humanities domain. The dataset provides fine-grained annotations with 19 classes as well as instance segmentation level ground truth data in COCO format. 

It has 1922 training samples and 335 validation samples. Mean average precision(mAP) in the validation samples is used for the evaluation metric in the ablation study of different model configurations.

To generalize the validation performance, data argumentation by using [Pytorch Transforms v2 library](https://pytorch.org/vision/main/auto_examples/plot_transforms_v2_e2e.html) experiments in this study. Further details of the data augmentation will be discussed in sections [Experiment Results](#experiment-results) and [Discussion and Analysis](#discussion-and-analysis). The visualization example of the ground-truth bounding boxes from one of the training samples is shown below:
| Without image transformations  |  After image transformations |
|:------------------------------:|:----------------------------:|
|![ground-truth-no-transform](https://github.com/BUW-CV/final-project-gary8564/assets/54540162/b2093636-ef13-4e3e-b5be-b0fc87b0faf7)| ![ground-truth-transforms](https://github.com/BUW-CV/final-project-gary8564/assets/54540162/5daa1b8d-e67c-4c0a-8c1a-189bc824963a)|

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
### 1. Training
Run the training script from Kaggle notebook or open your terminal/command line from in the `src` directory and execute the following command: 

```
python train.py --datapath ... --savepath ...
```
* `--datapath` is to input the root folder location where the TexBig dataset is stored.
* `--savepath` is to input the desired save location of the trained model. 

### 2. Testing
Run the testing script to output the prediction results as a json file in Kaggle notebook or open your terminal/command ····line from the `src` directory and execute the following command:
```
python test.py --backbone ... --weights ... --savepath ...
```
* `--datapath` is to input the root folder location where the test dataset is stored.
* `--backbone` is to input the desired backbone model name. Choices are limited to 'baseline', 'EfficientNetFPN', and 'ResNeXT101FPN'. It is an optional input argument. By default, it is set to 'ResNeXT101FPN'.
* `--weights` is to input the trained model weights path.
* `--savepath` is to input the desired save location of the output json file.

### 3. Inference
To run inference of the trained models on new data, open your terminal/command line from the `src` directory and execute ····the following command: 
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
│   └── config.py
│   ├── train.py
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
  5. `config.py` stores all training configurations. 
  6. `train.py` contains the executable training script.
  7. `test.py` contains the executable test script to test a trained model version with corresponding 
      weights on a validation or test dataset and output the object detection results in a json file.
  8. `inference.py` contains executable test inference of the trained models on new data.

## Experiment Results
### 1. Fine-tuning the baseline model - [pretrained RetinaNet](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html#torchvision.models.detection.retinanet_resnet50_fpn_v2) 
#### Grid search hyperparameters of batch size, learning rate, and anchor boxes
Warm-up StepLR scheduler first linearly increases the learning rate from an initial learning rate of 0.0005 to 0.001 in the first 1000 iterations. After 1000 iterations, the learning rate decays by 0.75 after every 5 epochs. The visualization of the warm-up StepLR scheduler is shown below:\
![learing rate scheduler](https://github.com/BUW-CV/final-project-gary8564/assets/54540162/961b7894-af94-41b3-949f-535efdd1e4d4)


Three different configurations are considered:
* batch size = 4; optimizer = SGD; warm-up SetpLR scheduler
* batch size = 2; optimizer = SGD; change parameters of anchor boxes
* batch size = 2; optimizer = SGD; ; warm-up SetpLR scheduler; change parameters of anchor boxes

The comparison results of three model configurations are shown below:\
(1) Training and validation loss history
![finetune_baseline](https://github.com/BUW-CV/final-project-gary8564/assets/54540162/60ffd218-807e-43d6-a53a-632333b2538e)

(2) mAP

| model configs | mAP           | mAP<sub>50</sub>| mAP<sub>75</sub>| mAP<sub>*s*</sub>| mAP<sub>*m*</sub>| mAP<sub>*l*</sub>|
|:-------------:|:-------------:|:---------------:|:---------------:|:--------------:|:--------------:|:--------------:|
| baseline      | 0.447         | 0.655           | 0.484           | 0.334          | 0.293          | 0.423          |
| config1       | 0.454         | 0.647           | 0.478           | 0.285          | 0.268          | 0.429          |  
| config2       | 0.494         | 0.727           | 0.530           | 0.340          | 0.344          | 0.460          |
| config3       | 0.478         | 0.695           | 0.504           | 0.316          | 0.311          | 0.439          |

### 2. Ablation study
The comparison results of different pre-trained backbone models are shown as follows:\
(1) Training and validation loss history 
![backbones](https://github.com/BUW-CV/final-project-gary8564/assets/54540162/4bc13b28-a0a5-4cc3-9244-cfaebecef559)

(2) mAP
| backbones     | mAP           | mAP<sub>50</sub>| mAP<sub>75</sub>| mAP<sub>*s*</sub>| mAP<sub>*m*</sub>| mAP<sub>*l*</sub>|
|:-------------:|:-------------:|:---------------:|:---------------:|:--------------:|:--------------:|:--------------:|
| no feature pyramids |||||||
| [Vit](https://pytorch.org/vision/stable/models/vision_transformer.html)| 0.220| 0.324 | 0.224 | 0.004 | 0.015 | 0.236 |
| [SwinT](https://pytorch.org/vision/stable/models/swin_transformer.html)| 0.214 | 0.377 | 0.207 | 0.223| 0.145| 0.178 |
| with feature pyramids |||||||
| [SwinT](https://pytorch.org/vision/stable/models/swin_transformer.html) | 0.242 | 0.398 | 0.224 | 0.244| 0.160| 0.183 |   
| [EfficientNetV2](https://pytorch.org/vision/main/models/efficientnetv2.html)| 0.441 | 0.637 | 0.478  | 0.231 | 0.225 | 0.435  |
| [ResNeXT101](https://pytorch.org/vision/main/models/resnext.html) |0.492 | 0.693 | 0.532 | 0.263 | 0.298 | 0.484|           

### 3. Data Augmentation
From the ablation study, ResNeXT101 as backbone yields the most promising result. Therefore, in this section, only ResNeXT101 backbone is considered. 

To improve the generalization, data augmentation using several image transformation techniques is implemented. In this study, RandomHorizontalFlip and ColorJitter are implemented. 

The result of mAP is shown below:
| backbone      | mAP           | mAP<sub>50</sub>| mAP<sub>75</sub>| mAP<sub>*s*</sub>| mAP<sub>*m*</sub>| mAP<sub>*l*</sub>|
|:-------------:|:-------------:|:---------------:|:---------------:|:--------------:|:--------------:|:--------------:|
| [ResNeXT101](https://pytorch.org/vision/main/models/resnext.html) | 0.546 |   0.775 |  0.608  | 0.341  |  0.347  |  0.528 |

The above results show that data augmentation can increase mAP by 6.6%.

#### 4. Final results
Retrain the best configuration (ResNeXT101-backbone; batch size=2; SGD with learning rate=0.001; warmup StepLR scheduler). 
The model weights can be downloaded [here](https://www.kaggle.com/datasets/gary8564/texbigdataset-trained-models).
The result of mAP is shown as follows:
| retrain epochs| mAP           | mAP<sub>50</sub>| mAP<sub>75</sub>| mAP<sub>*s*</sub>| mAP<sub>*m*</sub>| mAP<sub>*l*</sub>| download |
|:-------------:|:-------------:|:---------------:|:---------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| 8 <sup>[a](#note)</sup> | 0.610 |   0.802 |  0.654  | 0.361  |  0.428  |  0.599 | [link](https://drive.google.com/file/d/1hhoe8fKb2BuQbXcQ1llpOqRZZ-8M_xjz/view?usp=sharing)|
| 16 | 0.624 |   0.827 |  0.680  | 0.374  |  0.444  |  0.611 | [link](https://drive.google.com/file/d/15Fo95F_36xolUSnw3OvQm36mCiHPTBrm/view?usp=sharing)|

<a id="note">Note<sup>[a]</sup></a>: the retrained model of 8 epochs is the result on the leaderboard.


The comparison between the prediction of the final model and the ground-truth annotations is visualized below:
| Groud-Truths  |  Predictions |
|:------------------------------:|:----------------------------:|
|![ground-truth1](https://github.com/BUW-CV/final-project-gary8564/assets/54540162/7a633d50-fafd-438f-b683-3d55b7133c4b) | ![prediction1](https://github.com/BUW-CV/final-project-gary8564/assets/54540162/d67e999e-e71a-4dac-8266-0ef0b98c797e) |
|![ground-truth2](https://github.com/BUW-CV/final-project-gary8564/assets/54540162/5689ea3f-0a80-4ce4-adbe-2ed68ae4240a)| ![prediction2](https://github.com/BUW-CV/final-project-gary8564/assets/54540162/619fcb4d-67d7-48f3-a523-82e2c851ef44) |
|![ground-truth3](https://github.com/BUW-CV/final-project-gary8564/assets/54540162/8fcb1bb6-889f-473c-be23-f770c5c51fee) | ![prediction3](https://github.com/BUW-CV/final-project-gary8564/assets/54540162/808b6aaf-4e96-48f1-9466-1f1f8fadc514) |


## Discussion and Analysis
1. Learning rates:\
If the learning rate is set above 0.005, the model tends to diverge. It’s common to use a smaller learning rate for pre-trained models, in comparison to the (randomly initialized) weights. This is because we expect that the pre-trained weights are relatively good, so we don’t wish to distort them too quickly and too much. <sup>[[7]](#7)</sup>

2. Anchor boxes:\
Anchor boxes are one of the most influential hyperparameters to fine-tune. This can be proved in the baseline fine-tuning stage. Since most of the data contain smaller aspect ratios, I chose to add more anchor boxes and set smaller aspect ratios. The result of mAP is surprisingly improved by 10%.

3. Optimizers:\
At the first stage of fine-tuning, Adam-based optimizers such as Adam, AdamW, or RAdam are chosen as optimizers. However, during the training process, the validation loss of using Adam-based optimizers is worse than using SGD with Nesterov momentum. Numerous paper<sup>[[8]](#8),[[9]](#9),[[10]](#10)</sup> has also pointed out that Adam's generalization performance is worse than SGD, especially on image classification problems. A more recent paper<sup>[[11]](#11)</sup> further clarified that fine-tuned Adam is always better than SGD, while there exists a performance gap between Adam and SGD when using default hyperparameters. Since it might be difficult to find the optimal hyperparameters and the original paper of RetinaNet also used SGD optimizer, I, therefore, focused only on SGD optimizer. 

3. Backbones:\
   (1) Transformers-based backbones:\
   In order to fit in the constraints of training capacity, most of the encoder layers are frozen. However, freezing large portions of layers also led to unpromising results. It may be difficult to learn and fit into this complex domain-specific large dataset if freezing most parts of the model architecture.\
   Even though the result is not promising, the above mAP results can still get another interesting observation: SwinT transformers have more learning capacity to detect smaller objects.
   
   (2) ResNeXT and EfficientNet:\
   In order to speed up the training process, `nn.DataParallel` is utilized to fit with the Kaggle GPU-T4x2 accelerator. The above ablation study indicates that both EfficientNet and ResNeXT yield outstanding performances. In particular, ResNeXT-backbone model exceptionally outperforms others.
 

## Outlook and Future Work
In conclusion, despite the complexity of the historical documents dataset, by fine-tuning hyperparameters and increasing backbone model complexities, RetinaNet is still able to detect most of the annotations. Even though mAP on the test dataset leaderboard can only achieve 0.21, the performance can be improved by training more epochs if more powerful computing units can be accessed. More laborious fine-tuning with anchor boxes might also lead to more promising results.

In future work, unfreezing layers of ViT and SwinT backbone can be further experimented with to check for the improvement of results. Future studies can also try to implement other more recent methodologies such as VitDet<sup>[[12]](#12)</sup>, which utilized plain ViT-backbone with simple feature pyramid maps. In the ViTDet paper, the author also points out that the results can be benefited from using the readily available pre-trained transformer models from Masked Autoencoder(MAE). Therefore, using the pre-trained model from MAE can also be further discussed.

## Citation
<a id="1">[[1]](https://arxiv.org/abs/2212.13924)</a> N.-M. Sven and R. Matteo, “Page layout analysis of text-heavy historical documents: A comparison of textual and visual approaches,” arXiv [cs.IR], 2022.

<a id="2">[[2]](https://arxiv.org/abs/2105.06220)</a> Zhang, P. (2021, May 13). VSR: A Unified Framework for Document Layout Analysis combining Vision, Semantics, and Relations. arXiv.org.

<a id="3">[[3]](https://arxiv.org/abs/1708.02002)</a> Lin, T. (2017). Focal Loss for Dense Object Detection. arXiv.org.

<a id="4">[[4]](https://arxiv.org/abs/2010.11929)</a> Alexey Dosovitskiy, et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv.

<a id="5">[[5]](https://arxiv.org/abs/2103.14030)</a> Ze Liu Li, et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. arXiv.

<a id="6">[[6]](https://doi.org/10.1007/978-3-031-16788-1_22)</a> Tschirschwitz, D., Klemstein, F., Stein, B., Rodehorst, V. (2022). A Dataset for Analysing Complex Document Layouts in the Digital Humanities and Its Evaluation with Krippendorff’s Alpha. In: Andres, B., Bernard, F., Cremers, D., Frintrop, S., Goldlücke, B., Ihrke, I. (eds) Pattern Recognition. DAGM GCPR 2022. Lecture Notes in Computer Science, vol 13485. Springer, Cham.

<a id="7">[[7]](https://cs231n.github.io/transfer-learning/)</a> Transfer Learning. Stanford CS231n: Convolutional Neural Networks for Visual Recognition.

<a id="8">[[8]](https://arxiv.org/abs/1712.07628)</a> Keskar, N. S. (2017). Improving Generalization Performance by Switching from Adam to SGD. arXiv.org. 

<a id="9">[[9]](https://arxiv.org/abs/1712.07628)</a> Keskar, N. S. (2017b, December 20). Improving Generalization Performance by Switching from Adam to SGD. arXiv.org. 

<a id="10">[[10]](https://arxiv.org/abs/1509.01240)</a> Hardt, M. (2015, September 3). Train faster, generalize better: Stability of stochastic gradient descent. arXiv.org. 

<a id="11">[[11]](https://arxiv.org/abs/1910.05446)</a> Choi, D. (2019, October 11). On Empirical Comparisons of Optimizers for Deep Learning. arXiv.org. 

<a id="12">[[12]](https://arxiv.org/abs/2203.16527)</a> Li, Y. (2022, March 30). Exploring Plain Vision Transformer Backbones for Object Detection. arXiv.org. 
