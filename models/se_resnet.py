from torchvision.models.resnet import BasicBlock, ResNet
import torch.nn as nn
from .se_block import SEBlock
from torchvision.models import resnet18

class SEBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.se = SEBlock(self.conv2.out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def se_resnet18(num_classes=23):
    model = ResNet(block=SEBasicBlock, layers=[2, 2, 2, 2])
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model


def load_pretrained_weights(model):
    pretrained = resnet18(weights="IMAGENET1K_V1")
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in pretrained.state_dict().items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model