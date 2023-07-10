REPO_NAME = "../"
NUM_CLASSES = 20
EPOCH = 20
BATCH_SIZE = 2
MODEL_NAME_LIST = ["baseline", "EfficientNetFPN", "ResNeXTFPN", 
                   "SwinTFPN", "SwinT", "ViT"]
LEARNING_RATE = 1e-03
WEIGHT_DECAY = 1e-05
# anchor boxes parameters
ANCHOR_SIZES = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
ASPECT_RATIOS = ((0.33, 0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)
# frozen layer names
FROZEN_LAYERS = None
# whether to activate StepLR learning rate schedueler (boolean: true/false)
SCHEDULER = False
# whether to activate learning rate warmup (boolean: true/false)
WARMUP = True
# Whether to perform data augmentation before training
DATA_AUG = True
