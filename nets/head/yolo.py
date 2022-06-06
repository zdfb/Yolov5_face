from email.mime import base
import torch
import torch.nn as nn

from nets.layers import SSH
from nets.neck.PAN import PAN
from nets.backbone.CSPdarknet import CSPDarknet
from nets.backbone.resnet50 import Resnet50


class YoloFace(nn. Module):
    def __init__(self, anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]], num_classes = 0, 
                 phi = 'l', backbone = 'cspdarknet', neck = 'PAN', ssh = True,  pretrained = True):
        super(YoloFace, self).__init__()

        depth_dict = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]

        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        if backbone == 'cspdarknet':
            self.backbone = CSPDarknet(base_channels, base_depth, phi, pretrained)
        if backbone == 'resnet50':
            self.backbone = Resnet50(pretrained)
            base_channels = 64 * 2
            base_depth = 3

        if neck == 'PAN':
            self.neck = PAN(base_channels, base_depth)
        
        self.ssh = False
        if ssh == True:
            self.ssh = True
            self.ssh1 = SSH(base_channels * 4, base_channels * 4)
            self.ssh2 = SSH(base_channels * 8, base_channels * 8)
            self.ssh3 = SSH(base_channels * 16, base_channels * 16)
        
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes + 10), 1)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes + 10), 1)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes + 10), 1)
    
    def forward(self, x):
        
        # backbone
        inputs = self.backbone(x)

        # neck
        P3, P4, P5 = self.neck(inputs)

        # ssh
        if self.ssh == True:
            P3 = self.ssh1(P3)
            P4 = self.ssh2(P4)
            P5 = self.ssh3(P5)
        
        # head
        out2 = self.yolo_head_P3(P3)
        out1 = self.yolo_head_P4(P4)
        out0 = self.yolo_head_P5(P5)
        
        return out0, out1, out2
