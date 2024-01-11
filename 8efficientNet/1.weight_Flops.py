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

from model_efficientNet import efficientnet_b0 as create_model
import torch
from torchsummary import summary
from torchstat import stat
from model_efficientNet import *
import torchvision
from ptflops import get_model_complexity_info
from thop import profile
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


# Model
print('==> Building model..')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device.".format(device))

net = create_model().to(device)

original_classifier = net.classifier
# 修改全连接层的结构
num_features = net.classifier[-1].out_features
new_classifier = torch.nn.Sequential(
    original_classifier,
    torch.nn.Linear(num_features, 9)
)
net.classifier = new_classifier


net.load_state_dict(torch.load('./efficientNet.pth'))

pre=1
if pre:
    logger = get_logger('./EfficientNet参数量.log')
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(net.to(device), (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    logger.info('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

else:
    logger = get_logger('./参数量1.log')
    flops, params = get_model_complexity_info(net.to(device), (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
    logger.info('flops={}\t params={}'.format(flops, params))