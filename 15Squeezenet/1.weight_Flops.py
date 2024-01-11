# -*-coding:utf-8 -*-
'''
author:侯清刚
datetime:2023年06月03日
'''

# 要计算自己训练模型的权重（Weights）和浮点操作（FLOPs），您可以使用PyTorch的torchsummary和torchstat库。

# 确保已安装这些库：
#
# pip install torchsummary
# pip install torchstat

# 下面是使用这两个库计算自己训练模型的权重和FLOPs的示例代码：
from ptflops import get_model_complexity_info
from thop import profile
import torch.nn as nn
import torch
from torchsummary import summary
from torchstat import stat
from model_squeezenet import *
import torchvision
from torchvision import models
# from torch.nn import nn
# pre=1
# # 加载已训练的模型
# if pre:
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("using {} device.".format(device))
#
#     model=models.squeezenet1_0().to(device)
#     model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=9, kernel_size=(1, 1), stride=(1, 1))
#     model.load_state_dict(torch.load('Squeezenet2.pth'))
#
# else:
#     model = AlexNet(num_classes=9)
#
#     model.load_state_dict(torch.load('Squeezenet1.pth'))
#
# # 使用torchsummary库计算模型的权重
# summary(model, input_size=(3, 224, 224))
#
# # 使用torchstat库计算模型的FLOPs
# stat(model, (3, 224, 224))

from thop import profile
import logging
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

logger = get_logger('./Squeezenet参数量.log')
# Model
print('==> Building model..')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device.".format(device))

net=models.squeezenet1_0().to(device)
net.classifier[1] = nn.Conv2d(in_channels=512, out_channels=9, kernel_size=(1, 1), stride=(1, 1))
net.load_state_dict(torch.load('Squeezenet2.pth'))

net.to(device)

pre=0
if pre:
    logger = get_logger('./Squeezenet参数量.log')
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(net, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    logger.info('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

else:
    logger = get_logger('./参数量1.log')
    flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
    logger.info('flops={}\t params={}'.format(flops, params))