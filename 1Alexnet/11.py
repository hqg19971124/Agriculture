# -*-coding:utf-8 -*-
'''
author:侯清刚
datetime:2023年05月31日
'''
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # read class_indict
# json_path = './class_indices.json'
# assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
#
# with open(json_path, "r") as f:
#     class_indict = json.load(f)
#
#
# model = AlexNet(num_classes=9).to(device)
# weights_path = "./AlexNet.pth"
# assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
# print(torch.load(weights_path))    # start_epoch=0
# #断点恢复
RESUME=1
if RESUME:  # 是否恢复
    path_checkpoint = "./AlexNet.pth"  # 断点模型文件路径
    checkpoint = torch.load(path_checkpoint)  # 加载断点
    # start_epoch = checkpoint['epoch']
    print(checkpoint)
#     net=AlexNet()
#     net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
# #
#     start_epoch = checkpoint['epoch']  # 设置开始的epoch
#     optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
#     # lr_schedule.load_state_dict(checkpoint['lr_schedule'])  # 加载lr_scheduler
