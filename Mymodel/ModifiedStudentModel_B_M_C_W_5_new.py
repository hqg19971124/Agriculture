# -*-coding:utf-8 -*-
'''
author:侯清刚
datetime:2023年06月13日
'''
# -*-coding:utf-8 -*-
'''
author:侯清刚
datetime:2023年06月13日
'''
from MultiScaleModule_dilated import *
from WindowAttention import *
import torch
import torch.nn as nn
from CBAM import *

class ModifiedStudentModel_skip(nn.Module):
    def __init__(self):
        super(ModifiedStudentModel_skip, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, dilation=2)  # Modified to use dilated convolution
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=2, dilation=2, groups=16),
        #     nn.Conv2d(16, 32, kernel_size=1)
        # )
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cbam = CBAMBlock(32)
        self.win = WindowAttention(32, 32)
        self.residual = nn.Conv2d(16, 32, kernel_size=1)
        self.classifier = nn.Linear(32 * 55 * 55, 9)
        self.multi_scale = MultiScaleModule_dilated(16, 16)

    def forward(self, x):
        residual = self.conv1(x)
        x = self.relu1(residual)
        x = self.maxpool1(x)

        # x=self.multi_scale(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x_cbam = self.cbam(x)
        x_win = self.win(x)
        x=0.7*x_win+0.3*x_cbam

        # x=x+x_out

        residual = self.residual(residual)
        residual = nn.functional.interpolate(residual, size=x.size()[2:], mode='bilinear', align_corners=False)

        x = x + residual
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

# if __name__=='__main__':
#     input = torch.randn(1, 3, 224, 224)
#     model = ModifiedStudentModel_skip()
#     print(model)
#     print(model(input).shape)



