# -*-coding:utf-8 -*-
'''
author:侯清刚
datetime:2023年06月15日
'''
import torch
import torch.nn as nn

class WindowAttention(nn.Module):
    def __init__(self, in_channels, out_channels, window_size=7):
        super(WindowAttention, self).__init__()
        self.window_size = window_size
        self.conv_query = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # 对输入进行卷积操作得到查询、键和值
        queries = self.conv_query(x)
        keys = self.conv_key(x)
        values = self.conv_value(x)

        # 将查询、键和值进行reshape操作以得到窗口视图
        queries = queries.view(batch_size, -1, height * width)
        keys = keys.view(batch_size, -1, height * width)
        values = values.view(batch_size, -1, height * width)

        # 计算注意力权重
        attention_weights = torch.matmul(queries.transpose(1, 2), keys)  # 调整矩阵乘法的顺序
        attention_weights = self.softmax(attention_weights / (self.window_size ** 0.5))

        # 对值进行加权求和得到窗口注意力模块的输出
        out = torch.matmul(attention_weights, values.transpose(1, 2))
        out = out.view(batch_size, -1, height, width)

        return out