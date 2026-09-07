import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet, resnet50
from .se_block import SEBlock


class SEBottleneck(Bottleneck):
    """
    ResNet50, 101, 152用のSEブロック付きBottleneck
    構造: 1x1 conv -> 3x3 conv -> 1x1 conv -> SE -> Add -> ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.se = SEBlock(self.conv3.out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def se_resnet50(num_classes=1000):
    model = ResNet(block=SEBottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)
    return model


def load_pretrained_weights(model):
    pretrained = resnet50(weights="IMAGENET1K_V1")
    model_dict = model.state_dict()
    
    pretrained_dict = {
        k: v for k, v in pretrained.state_dict().items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    
    model.load_state_dict(pretrained_dict, strict=False)
    return model