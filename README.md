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
To fine-tune the aspect ratios of anchor boxes, the distribution of the aspect ratios of all annotated bounding boxes in the dataset is analyzed as below:
|               | Aspect ratio  | 
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

    ```
    python3 -m venv dlcv_final_project  
    source dlcv_final_project/bin/activate  
    pip3 install torch torchvision
    pip install pycocotools
    pip install torchmetrics
    pip install -e .  
    ```

## Usage


## Code Structure

## Experiment Results

## Discussion and Analysis

## Outlook and Future Work

## Citation
<a id="1">[[1]](https://arxiv.org/abs/2212.13924)</a> N.-M. Sven and R. Matteo, “Page layout analysis of text-heavy historical documents: A comparison of textual and visual approaches,” arXiv [cs.IR], 2022.

<a id="2">[[2]](https://arxiv.org/abs/2105.06220)</a> Zhang, P. (2021, May 13). VSR: A Unified Framework for Document Layout Analysis combining Vision, Semantics and Relations. arXiv.org.

<a id="3">[[3]](https://arxiv.org/abs/1708.02002)</a> Lin, T. (2017). Focal Loss for Dense Object Detection. arXiv.org.

<a id="4">[[4]](https://arxiv.org/abs/2010.11929)</a> Alexey Dosovitskiy, et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv.

<a id="5">[[5]](https://arxiv.org/abs/2103.14030)</a> Ze Liu Li, et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. arXiv.

<a id="6">[[6]](https://doi.org/10.1007/978-3-031-16788-1_22)</a> Tschirschwitz, D., Klemstein, F., Stein, B., Rodehorst, V. (2022). A Dataset for Analysing Complex Document Layouts in the Digital Humanities and Its Evaluation with Krippendorff’s Alpha. In: Andres, B., Bernard, F., Cremers, D., Frintrop, S., Goldlücke, B., Ihrke, I. (eds) Pattern Recognition. DAGM GCPR 2022. Lecture Notes in Computer Science, vol 13485. Springer, Cham.
