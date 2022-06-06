import torch
import torch.nn as nn
from torchvision import models

class Resnet50(nn.Module):
    def __init__(self, preteained = True):
        super(Resnet50, self).__init__()

        model = models.resnet50(pretrained = preteained)

        features1 = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2])
        features2 = list([model.layer3])
        features3 = list([model.layer4])

        self.backbone1 = nn.Sequential(*features1)  # 512
        self.backbone2 = nn.Sequential(*features2)  # 1024
        self.backbone3 = nn.Sequential(*features3)  # 2048
    
    def forward(self, x):
        feat1 = self.backbone1(x)
        feat2 = self.backbone2(feat1)
        feat3 = self.backbone3(feat2)

        return feat1, feat2, feat3

