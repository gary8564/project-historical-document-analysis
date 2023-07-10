import json
import glob
from PIL import Image
import torch
import os
import argparse
from .config import *

if __name__ == "__main__":
    
    # Construct the argument parser.
    parser = argparse.ArgumentParser() 
    parser.add_argument('-m', '--backbone', default='ResNeXT101FPN', 
                    help='baseline(ResNet50), EfficientNet wtih FPN, ResNeXT with FPN',
                    choices=['baseline', 'EfficientNetFPN', 'ResNeXTFPN'])
    parser.add_argument('-w', '--weights', default='../pretrained/final_model.pt',
                    help='trained model weight path')
    parser.add_argument('-s', '--savepath', default=REPO_NAME,
                    help='test json file save path')
    args = vars(parser.parse_args())
    
    # append system path
    import sys
    repo_name = REPO_NAME
    sys.path.append(repo_name)
    from src import *
    
    # load test dataset
    data_dir = os.path.join(repo_name, "test")
    test_images = glob.glob(f"{data_dir}/*") 
    print(f"Test instances: {len(test_images)}")
    
    # model evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_pretrained_model(device=device, model_name='baseline')
    weights_path = repo_name + args['weights']
    model = load_pretrained_weights(model, weights_path, device)
    model.eval()
    coco_results = []
    for i in range(len(test_images)):
        # get the image file name for saving output later on
        image_name = test_images[i].split('/')[-1] 
        image = Image.open(test_images[i]).convert('RGB')
        # transform the image to tensor
        test_transform = get_transform(moreAugmentations=False)
        image = test_transform(image).to(device)
        image = image.unsqueeze(0) # add a batch dimension
        with torch.no_grad():
            outputs = model(image) # get the predictions on the image
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes']
            scores = outputs[0]['scores'].tolist()
            labels = outputs[0]["labels"].tolist()
            xmin, ymin, xmax, ymax = boxes.unbind(1)
            boxes = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1).tolist()
            coco_results.extend(
                [
                    {
                        "file_name": image_name,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
    result_filepath = os.join(args['savepath'], "test.json")
    with open(result_filepath, "w") as outfile:
        json.dump(coco_results, outfile)
        
        
        
    
    