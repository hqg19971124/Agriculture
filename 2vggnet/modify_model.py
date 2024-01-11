# -*-coding:utf-8 -*-
'''
author:侯清刚
datetime:2023年05月31日
'''
from model import VGG
import torch.nn as nn
import torchvision
net=torchvision.models.vgg16()
# AlexNet=AlexNet()
# in_channel=AlexNet.classifier[-1].in_features
# # AlexNet.classifier= nn.Linear(in_channel, 9)
# print(type(AlexNet.classifier))

# net = torchvision.models.alexnet()
in_channel=net.classifier[-1].in_features
net.classifier[-1]= nn.Linear(in_channel, 9)
print(net)