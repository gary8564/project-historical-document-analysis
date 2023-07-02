import torchvision
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, ResNeXt101_32X8D_Weights
from torchvision.models.efficientnet import EfficientNet, EfficientNet_B3_Weights, efficientnet_b3
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, RetinaNet_ResNet50_FPN_V2_Weights, retinanet_resnet50_fpn_v2
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, BackboneWithFPN
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.models.detection.transform import GeneralizedRCNNTransform
    
class EfficientNet_FPN(nn.Module):
    def __init__(self,):
        super(EfficientNet_FPN, self).__init__()
        efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT).features
        in_channels_list = [232, 384, 1536]
        return_layers = {'6': '0', '7': '1', '8': '2'}
        self.out_channels = 256
        self.backbone = BackboneWithFPN(efficientnet, return_layers, in_channels_list, self.out_channels, extra_blocks=LastLevelP6P7(256, 256))
        if torch.cuda.device_count() > 1:
            self.backbone = nn.DataParallel(self.backbone)
        
    def forward(self, x):
        x = x.cuda()
        x = self.backbone(x)
        return x
    
class ViT(nn.Module):
    def __init__(self, device):
        super(ViT, self).__init__()
        # Get a ViT backbone
        vit = pretrained_ViT(device)
        self.body = create_feature_extractor(vit, return_nodes=['encoder'])
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 3, 224, 224).cuda()
        with torch.no_grad():
            out = self.body(inp)
            batch_size = out['encoder'].size(0)
            out = out['encoder'].view(batch_size, -1, 16, 16)
        self.out_channels = out.shape[1]

    def forward(self, x):
        x = x.cuda()
        x = self.body(x)
        x = x['encoder'].view(x['encoder'].size(0), -1, 16, 16)
        return x

class SwinT(nn.Module):
    def __init__(self, device):
        super(SwinT, self).__init__()
        # Get a ViT backbone
        swint = pretrained_swinT(device)
        self.body = nn.Sequential(swint.features, swint.norm, swint.permute)
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 3, 256, 256).cuda()
        with torch.no_grad():
            out = self.body(inp)
        self.out_channels = out.shape[1]
  
    def forward(self, x):
        x = x.cuda()
        x = self.body(x)
        return x

class SwinTWithFPN(nn.Module):
    def __init__(self, device):
        super(SwinTWithFPN, self).__init__()
        # Get a ViT backbone
        swint = pretrained_swinT(device).features
        return_layers = {'3': '0',
                         '5': '1',
                         '7': '2'}
        in_channels_list = [192, 384, 768]
        self.out_channels = 256
        self.backbone = BackboneWithFPN(swint, return_layers, in_channels_list, self.out_channels, extra_blocks=LastLevelP6P7(256, 256))
        if torch.cuda.device_count() > 1:
            self.backbone = nn.DataParallel(self.backbone)
        
    def forward(self, x):
        x = x.cuda()
        x = self.backbone(x)
        return x

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
        anchorSizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
        aspectRatios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        
        if backbone:
            assert backbone in ["ResNeXt_FPN", "EfficientNet_FPN", "ViT", "SwinT", "SwinT_FPN"]
            if (backbone == "ViT"):
                backboneModel = ViT(device)
                anchorSizes = ((32, 64, 128, 256, 512),)
                aspectRatios = ((0.5, 1.0, 2.0),)
            elif (backbone == "SwinT"):
                backboneModel = SwinT(device)
                anchorSizes = ((32, 64, 128, 256, 512),)
                aspectRatios = ((0.5, 1.0, 2.0),)
            elif (backbone == "SwinT_FPN"):
                backboneModel = SwinTWithFPN(device)
            elif (backbone == "EfficientNet_FPN"):
                backboneModel = EfficientNet_FPN()
            else:
                backboneModel = resnet_fpn_backbone('resnext101_32x8d', weights=ResNeXt101_32X8D_Weights.DEFAULT,
                                                    returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256))
                if torch.cuda.device_count() > 1:
                    backboneModel = nn.DataParallel(backboneModel)
                    backboneModel.out_channels = 256
        
        if anchor_sizes:
            anchorSizes = anchor_sizes
            
        if aspect_ratios: 
            aspectRatios = aspect_ratios
            
        model = RetinaNet(
            backbone=backboneModel,
            num_classes=num_classes,
            anchor_generator=anchorGenerator(anchorSizes, aspectRatios),
        )
        if backbone == "ViT":
            model.transform = GeneralizedRCNNTransform(min_size=224, max_size=256, image_mean=[0.485, 0.456, 0.406], 
                                     image_std=[0.229, 0.224, 0.225], fixed_size=(224, 224))
        if backbone == "SwinT":
            model.transform = GeneralizedRCNNTransform(min_size=256, max_size=272, image_mean=[0.485, 0.456, 0.406], 
                                     image_std=[0.229, 0.224, 0.225], fixed_size=(256, 256))
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
        
def pretrained_ViT(device):
    # Get pretrained weights for ViT-Base
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 
    # Setup a ViT model instance with pretrained weights
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)
    #print(pretrained_vit)
    #print(get_graph_node_names(pretrained_vit))
    return pretrained_vit

def pretrained_swinT(device):
    # Get pretrained weights for ViT-Base
    pretrained_swin_weights = torchvision.models.Swin_V2_S_Weights.DEFAULT 
    # Setup a ViT model instance with pretrained weights
    pretrained_swin = torchvision.models.swin_v2_s(weights=pretrained_swin_weights).to(device)
    #print(pretrained_swin)
    return pretrained_swin

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
    if model_name == 'baseline':
        backboneModel = resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.DEFAULT)
        anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        model = RetinaNet(
                    backbone=backboneModel,
                    num_classes=20,
                    anchor_generator=anchorGenerator(anchor_sizes, aspect_ratios),
                )
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
    frozen_layers = ["backbone"] 
    model = freeze_layers(model, frozen_layers) 
    print(model)