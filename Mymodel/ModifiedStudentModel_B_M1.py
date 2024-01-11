# -*-coding:utf-8 -*-
'''
author:侯清刚
datetime:2023年06月12日
'''
import torch
import torch.nn as nn
from CBAM import *
from MultiScaleModule_dilated import *

class ModifiedStudentModel_skip(nn.Module):
    def __init__(self):
        super(ModifiedStudentModel_skip, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.multi_scale = MultiScaleModule_dilated(16,16) #放在卷积层后

        self.residual = nn.Conv2d(16, 32, kernel_size=1)  # Adjust the channel dimension of the residual connection
        self.classifier = nn.Linear(32 * 56 * 56, 9)

    def forward(self, x):
        residual = self.conv1(x)

        x = self.relu1(residual)
        x = self.maxpool1(x)

        x = self.multi_scale(x)  #多尺度放在第一个卷积层后

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)


        # x = self.cbam(x)

        # Adjust the channel dimension of the residual tensor
        residual = self.residual(residual)

        # Upsample the residual tensor to match the size of x
        residual = nn.functional.interpolate(residual, size=x.size()[2:], mode='bilinear', align_corners=False)

        x = x + residual

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x