import torchvision
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, ResNeXt101_32X8D_Weights
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, RetinaNet_ResNet50_FPN_V2_Weights, retinanet_resnet50_fpn_v2
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

def retinaNet(num_classes, device, backbone=None, anchor_sizes=None, aspect_ratios=None):
    """
    Build the RetinaNet model.
    
    Parameters
    ----------
    num_classes : int
        number of classification labels.
    backbone : str
        The name of the customized backbone model for which the pretrained
        weights are to be loaded.
        None by default: pretrained RetinaNet baseline model from Pytorch.
        If not None, must be one of ["ViT", "SwinT"] which
        ViT: VisionTransformer, SwinT: SwinTransformer
    anchor_sizes : tuple
    aspect ratios : tuple
        aspect ratios = height / width for each anchor. 
        sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
        and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
        per spatial location for feature map i.

    Returns
    -------
    model : torchvisions.models
        RetinaNet training model 
    """
    if any(item is not None for item in [backbone, anchor_sizes, aspect_ratios]):
        backboneModel = resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.DEFAULT, 
                                            returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256))
        anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes) 
        if backbone:
            assert backbone in ["ResNet_FPN", "ViT", "SwinT"]
            backboneModel = resnet_fpn_backbone('resnext101_32x8d', weights=ResNeXt101_32X8D_Weights.DEFAULT,
                                                 returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256))
            if (backbone == "ViT"):
                backboneModel.body = viTBackBone(device)
            elif (backbone == "SwinT"):
                backboneModel.body = swinTBackBone(device)
        if anchor_sizes and aspect_ratios:
            pass
        elif anchor_sizes:
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes) 
        elif aspect_ratios:
            anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
        model = RetinaNet(
            backbone=backboneModel,
            num_classes=num_classes,
            anchor_generator=anchorGenerator(anchor_sizes, aspect_ratios),
        )
        #print(model)
        return model.to(device)    
    else:
        model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        # get number of input features and anchor boxed for the classifier
        in_features = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        # replace the pre-trained head with a new one
        model.head = RetinaNetHead(in_features, num_anchors, num_classes)
        #print(model)
        return model.to(device)
        
def viTBackBone(device):
    # Get pretrained weights for ViT-Base
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 
    # Setup a ViT model instance with pretrained weights
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)
    # print(pretrained_vit)
    # Load the pretrained ViT backbone.
    class_token = pretrained_vit.class_token
    conv_proj = pretrained_vit.conv_proj
    print(pretrained_vit.seq_length)
    print(pretrained_vit.hidden_dim)
    print(pretrained_vit.patch_size)
    print(pretrained_vit.image_size)
    print(class_token.size)
    print(conv_proj)
    encoder = pretrained_vit.encoder
    backbone = nn.Sequential(conv_proj, encoder)
    # Retinanet needs to know the number of output channels in a backbone.
    # For vit_b_16, it's 768
    backbone.out_channels = 768
    return backbone

def swinTBackBone(device):
    # Get pretrained weights for ViT-Base
    pretrained_swin_weights = torchvision.models.Swin_V2_B_Weights.DEFAULT 
    # Setup a ViT model instance with pretrained weights
    pretrained_swin = torchvision.models.swin_v2_b(weights=pretrained_swin_weights).to(device)
    # print(pretrained_swin)
    # Load the pretrained ViT backbone.
    backbone = pretrained_swin.features
    # Retinanet needs to know the number of output channels in a backbone.
    # For vit_b_16, it's 1024
    backbone.out_channels = 1024
    return backbone

def anchorGenerator(anchor_sizes, aspect_ratios):
    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Anchors with 5 different sizes and 3 different aspect ratios(h/w).
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    return anchor_generator
    
def roIPooler():
    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to be [0]. 
    # More generally, the backbone should return an OrderedDict[Tensor], 
    # and in featmap_names we can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], 
        output_size=7, 
        sampling_ratio=2
    )
    return roi_pooler

def get_model(device='cpu', model_name='baseline'):
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [16, 32, 64, 128, 256]) 
    aspect_ratios=((0.33, 0.5, 1.0, 1.33, 2.0),) * len(anchor_sizes) 
    if model_name == 'baseline':
        model = retinaNet(num_classes=20, device=device, anchor_sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    elif model_name == 'ViT backbone':
        model = retinaNet(num_classes=20, device=device, backbone="ViT", anchor_sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    else:
        model = retinaNet(num_classes=20, device=device, backbone="SwinT", anchor_sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    model = model.eval().to(device)
    return model


if __name__ == "__main__":
    from utils import freeze_layers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = "ViT"
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [16, 32, 64, 128, 256])
    aspect_ratios=((0.33, 0.5, 1.0, 1.33, 2.0),) * len(anchor_sizes)
    model = retinaNet(num_classes=20, device=device, backbone=backbone, anchor_sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    frozen_layers = ["backbone.1"] 
    model = freeze_layers(model, frozen_layers) 
    print(model)