import torchvision
import torch
from torch import nn
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, RetinaNet_ResNet50_FPN_V2_Weights, retinanet_resnet50_fpn_v2
from torchvision.models.detection.rpn import AnchorGenerator


def retinaNet(num_classes, device, backbone=None):
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

    Returns
    -------
    model : torchvisions.models
        RetinaNet training model 
    """
    if backbone:
        assert backbone in ["ViT", "SwinT"]
        if (backbone == "ViT"):
            backboneModel = viTBackBone(device)
        else:
            backboneModel = swinTBackBone(device)
        # Final customized RetinaNet model.
        model = RetinaNet(
            backbone=backboneModel,
            num_classes=num_classes,
            rpn_anchor_generator=anchorGenerator(),
            box_roi_pool=roIPooler()
        )
        print(model)
        return model
    else:
        model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        # get number of input features and anchor boxed for the classifier
        in_features = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        # replace the pre-trained head with a new one
        model.head = RetinaNetHead(in_features, num_anchors, num_classes)
        print(model)
        return model
        
def viTBackBone(device):
    # Get pretrained weights for ViT-Base
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 
    # Setup a ViT model instance with pretrained weights
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)
    # print(pretrained_vit)
    # Load the pretrained ViT backbone.
    conv_proj = pretrained_vit.conv_proj
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

def anchorGenerator():
    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Anchors with 5 different sizes and 3 different aspect ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), 
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
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



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = retinaNet(num_classes=19, device=device)
