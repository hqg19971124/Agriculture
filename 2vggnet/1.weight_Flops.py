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


import torch
from torchsummary import summary
from torchstat import stat
from model import *
import torchvision
from ptflops import get_model_complexity_info
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

# logger = get_logger('./train1参数量.log')
# Model
print('==> Building model..')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device.".format(device))

model_name = "vgg16"
model = vgg(model_name=model_name, num_classes=9, init_weights=True)
model.load_state_dict(torch.load('vgg16Net.pth'))
model.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_parameters = count_parameters(model)
print("Total Parameters: ", total_parameters)

logger = get_logger('./VGG参数量.log')
flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
print('flops: ', flops, 'params: ', params)
logger.info('flops={}\t params={}'.format(flops, params))