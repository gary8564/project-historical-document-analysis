import torch
import argparse
import cv2
import detect_utils
import numpy as np
from PIL import Image
from model import get_model

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', default='/image_1.jpg', 
    help='path to input image'
)
parser.add_argument(
    '-t', '--threshold', default=0.5, type=float,
    help='detection threshold for non-maximum supression of the bounding boxes'
)
parser.add_argument(
    '-m', '--model', default='baseline', 
    help='baseline model(retinanet_resnet50_fpn_v2) or retinanet with ViT backbone or retinanet with SwinT backbone',
    choices=['baseline', 'ViT backbone', 'SwinTbackbone']
)
args = vars(parser.parse_args())