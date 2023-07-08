import torch
import torchvision
import argparse
import cv2
import numpy as np
from PIL import Image

if __name__ == "__main__":
    # Construct the argument parser.
    parser = argparse.ArgumentParser() 
    parser.add_argument('-r', '--root', default="/Users/kyle_lee/Desktop/Bauhaus/DL4CV/final-project-gary8564",
                        help='root path of the project')
    parser.add_argument('-i', '--input', default='../archive/val/14688302_1881_Seite_002.tiff', 
                    help='path to input image') 
    parser.add_argument('-t', '--threshold', default=0.5, type=float, 
                    help='minimum confidence score for detection')
    parser.add_argument('-m', '--model', default='ResNeXT101FPN', 
                    help='baseline model(retinanet_resnet50_fpn_v2), RetinaNet with EfficientNet FPN, \
                        RetinaNet with ResNeXT101 FPN',
                    choices=['baseline', 'EfficientNetFPN', 'ResNeXT101FPN'])
    parser.add_argument('-w', '--weights', default='../pretrained/final_model.pt',
                    help='trained model weight path')
    args = vars(parser.parse_args())
    # append system path
    import sys
    repo_name = args['root']
    sys.path.append(repo_name)
    from src import *
    # classes: 0 index is reserved for background
    CLASSES = [
        'background', 'Advertisement', 'Author', 'Caption', 'Column title', 'Decoration', 
        'Editorial note', 'Equation', 'Footer', 'Footnote', 'Frame', 'Header', 'Heading', 
        'Image', 'Logo', 'Noise', 'Page Number', 'Paragraph', 'Sub-heading', 'Table'
        ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device, args['model'])
    weights_path = args['weights']
    model = load_pretrained_weights(model, weights_path, device)
    image = Image.open(args['input']).convert('RGB')
    save_filename = f"{args['input'].split('/')[-1].split('.')[0]}_t{int(args['threshold']*100)}_model{args['model']}"
    predict(image, model, device, args['threshold'], CLASSES, save_filename)
