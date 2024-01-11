# -*-coding:utf-8 -*-
'''
author:侯清刚
datetime:2023年06月01日
'''
import torch
import torch.nn as nn
import torch.nn.init as init
from CBAM import *


class FireModule(nn.Module):
    def __init__(self, in_planes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_planes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(512, 64, 256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            # nn.Conv2d(1000, num_classes, kernel_size=1),# 自己添加的一层
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self._initialize_weights()
        # self.CBAM=CBAMBlock(512)

    def forward(self, x):
        x = self.features(x)
        # X=self.CBAM(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

# if __name__=='__main__':
#     model=SqueezeNet()
#     # print(model)
# #
#     original_classifier = model.classifier[:-1]
#     end_classifier=model.classifier[-1]
#     # print(original_classifier[1])
#     # 修改全连接层的结构
#     num_features = model.classifier
#     new_classifier = torch.nn.Sequential(
#         original_classifier,
#         torch.nn.Linear(1000, 9),
#         end_classifier
#
#     )
#     model.classifier = new_classifier
#     print(model)
