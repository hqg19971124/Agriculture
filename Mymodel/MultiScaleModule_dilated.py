# -*-coding:utf-8 -*-
'''
author:侯清刚
datetime:2023年06月15日
'''
import torch
import torch.nn as nn

class MultiScaleModule_dilated(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MultiScaleModule_dilated, self).__init__()

        self.conv1x1 = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.conv3x3_dilated = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=2, dilation=2, groups=input_channels),
            nn.Conv2d(input_channels, output_channels, kernel_size=1)
        )
        self.conv5x5_dilated = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=5, padding=4, dilation=2, groups=input_channels),
            nn.Conv2d(input_channels, output_channels, kernel_size=1)
        )

    def forward(self, x):
        out1x1 = self.conv1x1(x)
        out3x3_dilated = self.conv3x3_dilated(x)
        out5x5_dilated = self.conv5x5_dilated(x)

        out = out1x1 + out3x3_dilated + out5x5_dilated

        return out
