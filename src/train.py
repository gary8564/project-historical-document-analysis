import argparse
import os
from config import *
 
if __name__ == '__main__':
    
    # Construct the argument parser.
    parser = argparse.ArgumentParser() 
    parser.add_argument('-d', '--datapath',
                    help='TexBig dataset root folder')
    parser.add_argument('-s', '--savepath', default='../pretrained',
                    help='trained model save path')
    
    args = vars(parser.parse_args())
    
    # append system path
    import sys
    repo_name = REPO_NAME
    sys.path.append(repo_name)
    from src import *
    import torch
    from torch.utils.data import DataLoader, random_split
    

    # load the dataset
    data_dir = args['datapath']
    annot_filename = 'train'
    train_transformers = get_transform(moreAugmentations=DATA_AUG)
    train = TexBigDataset(data_dir, annot_filename, train_transformers)
    train_size = int(0.8 * len(train))
    val_size = len(train) - train_size
    train_dataset, val_dataset = random_split(train, [train_size, val_size])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = BATCH_SIZE
    num_epochs = EPOCH
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)
    model_name = MODEL_NAME[2]
    model = retinaNet(num_classes=NUM_CLASSES, device=device, backbone=model_name, anchor_sizes=ANCHOR_SIZES, aspect_ratios=ASPECT_RATIOS)
    frozen_layers = FROZEN_LAYERS
    if frozen_layers is not None:
        model = freeze_layers(model, frozen_layers).to(device)
    params = [p for p in model.parameters() if p.requires_grad]        
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, nesterov=True, weight_decay=WEIGHT_DECAY)
    lr_scheduler = None
    if SCHEDULER:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=int(num_epochs/3),
                                               gamma=0.75)
    train_losses, val_losses, mAP_50, mAP = train_and_validate_model(model, train_loader, val_loader, optimizer, num_epochs, device, WARMPUP, lr_scheduler)
    # save trained models and configuration results
    save_results_csv(model_name, train_losses, val_losses, mAP)
    PATH = os.join(args['savepath'], "model_%s.pt" %(model_name))
    torch.save(model.state_dict(), PATH)
