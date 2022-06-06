import torch
from torch import nn
from nets.backbone.CSPdarknet import SiLU


def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias = False),
        nn.BatchNorm2d(oup, eps = 0.001, momentum = 0.03),
        SiLU(),)


def conv_bn_no_silu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias = False),
        nn.BatchNorm2d(oup, eps = 0.001, momentum = 0.03),)


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
       
        self.conv3X3 = conv_bn_no_silu(in_channel, out_channel//2, stride = 1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1)
        self.conv5X5_2 = conv_bn_no_silu(out_channel//4, out_channel//4, stride = 1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1)
        self.conv7x7_3 = conv_bn_no_silu(out_channel//4, out_channel//4, stride = 1)
        
        self.silu = SiLU()

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim = 1)
        out = self.silu(out)
        return out