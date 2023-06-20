import torch
import argparse
import cv2
import numpy as np
from PIL import Image

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='./archive/val/14688302_1881_Seite_002.tiff', 
                    help='path to input image')
parser.add_argument('-t', '--threshold', default=0.5, type=float, 
                    help='minimum confidence score for detection')
parser.add_argument('-m', '--model', default='baseline', 
                    help='baseline model(retinanet_resnet50_fpn_v2) or retinanet with ViT backbone or retinanet with SwinT backbone',
                    choices=['baseline', 'ViT backbone', 'SwinTbackbone'])
args = vars(parser.parse_args())