if __name__ == '__main__':
    # append system path
    import sys
    repo_name = "/Users/kyle_lee/Desktop/Bauhaus/DL4CV/final-project-gary8564"
    sys.path.append(repo_name)
    from src import *
    import torch
    from torch.utils.data import DataLoader, random_split
    
    # Construct the argument parser.
    parser.add_argument('-d', '--datapath',
                    help='TexBig dataset root folder')
    parser.add_argument('-b', '--batchsize', default=2,
                    help='batch size')
    parser.add_argument('-e', '--epochs', default=10,
                    help='epochs')
    parser.add_argument('-n', '--modelname', default="ResNeXT101FPN",
                    help='model name',
                    choices=["baseline", "EfficientNetFPN", "ResNeXT101FPN", 
                             "SwinTFPN", "SwinT", "ViT"])
    parser.add_argument('-f', '--frozen', default=None,
                    help='frozen layers name')
    parser.add_argument('-s', '--scheduler', default=False,
                    help='whether to activate learning rate schedueler (boolean: true/false)')
    parser.add_argument('-w', '--warmup', default=True,
                    help='whether to activate learning rate warmup (boolean: true/false)')
    
    args = vars(parser.parse_args())

    # load the dataset
    data_dir = args['datapath']
    annot_filename = 'train'
    train_transformers = get_transform(moreAugmentations=False)
    train = TexBigDataset(data_dir, annot_filename, train_transformers)
    train_size = int(0.8 * len(train))
    val_size = len(train) - train_size
    train_dataset, val_dataset = random_split(train, [train_size, val_size])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args['batchsize']
    num_epochs = args['epochs']
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)
    model_name = args['modelname']
    get_model(device=device, model_name=model_name)
    frozen_layers = args['frozen']
    if frozen_layers is not None:
        model = freeze_layers(model, frozen_layers).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=1e-03, momentum=0.9, nesterov=True, weight_decay=1e-05)
    lr_scheduler = None
    if args['scheduler']:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=int(num_epochs/3),
                                               gamma=0.75)
    train_losses, val_losses, mAP_50, mAP = train_and_validate_model(model, train_loader, val_loader, optimizer, num_epochs, device, args['warmup'], lr_scheduler)
    # save trained models and configuration results
    save_results_csv(model_name, train_losses, val_losses, mAP)
    PATH = "./model_%s.pt" %(model_name)
    torch.save(model.state_dict(), PATH)
