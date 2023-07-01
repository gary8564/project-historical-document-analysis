import json
import glob
from PIL import Image
import torch
import os

if __name__ == "__main__":
    # append system path
    import sys
    repo_name = "/Users/kyle_lee/Desktop/Bauhaus/DL4CV/final-project-gary8564"
    sys.path.append(repo_name)
    from src import *
    data_dir = os.path.join(repo_name, "test")
    test_images = glob.glob(f"{data_dir}/*") 
    print(f"Test instances: {len(test_images)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    model = get_model(device='cpu', model_name='baseline')
    weights_path = repo_name + '/pretrained/model_baseline_batch2_SGD_changeAnchorBoxes.pt'
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
            thresholded_preds_inidices = [scores.index(i) for i in scores if i >= 0.5]
            coco_results.extend(
                [
                    {
                        "file_name": image_name,
                        "category_id": labels[index],
                        "bbox": boxes[index],
                        "score": scores[index],
                    }
                    for index in thresholded_preds_inidices
                ]
            )
    result_filepath = repo_name + "/test.json"
    with open(result_filepath, "w") as outfile:
        json.dump(coco_results, outfile)
        
        
        
    
    