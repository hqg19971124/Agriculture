# -*-coding:utf-8 -*-
'''
author:侯清刚
datetime:2023年06月12日
'''
import torch
import torch.nn as nn


from MultiScaleModule_dilated import *

class ModifiedStudentModel_skip(nn.Module):
    def __init__(self):
        super(ModifiedStudentModel_skip, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, dilation=2)  # Modified to use dilated convolution
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.multi_scale1 = MultiScaleModule_dilated(16,16)  #放在卷积层后
        self.multi_scale2 = MultiScaleModule_dilated(32,32)  #放在卷积层后
        # self.cbam = CBAMBlock(32)
        self.residual = nn.Conv2d(16, 32, kernel_size=1)  # Adjust the channel dimension of the residual connection
        self.classifier = nn.Linear(32 * 55 * 55, 9)

    def forward(self, x):
        residual = self.conv1(x)

        x = self.relu1(residual)
        x = self.maxpool1(x)

        x = self.multi_scale1(x)  # 多尺度放在第y一个卷积层后

        x = self.conv2(x)  # Use the modified dilated convolution
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.multi_scale2(x)  # 多尺度放在第二个卷积层后
        # x = self.cbam(x)

        residual = self.residual(residual)

        residual = nn.functional.interpolate(residual, size=x.size()[2:], mode='bilinear', align_corners=False)

        x = x + residual

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
