#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------------------
File Name: Edge
Description :Document description
Author: hzhou
date: 4/11/24
-------------------------------------------------
Change Activity:
4/11/24: Description of change
-------------------------------------------------
"""

import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self,
        n_feat,
        kernel_size=3,
        reduction=16,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(
                self.default_conv(n_feat, n_feat, kernel_size, bias=bias)
            )
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2),
            bias=bias,
        )

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class EdgeDetectionModule(nn.Module):
    def __init__(self, in_dim=64+128, hidden_dim=256, out_dim=128, size=512):
        super(EdgeDetectionModule, self).__init__()
        self.size = size
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.edge_embedding_1 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        self.edge_conv = nn.Conv2d(hidden_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.edge_mask = nn.Sequential(
            nn.AdaptiveAvgPool2d(size),
            nn.Conv2d(out_dim, out_dim, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_dim, out_dim, 1),
            nn.Sigmoid(),
        )
        self.conv_edge_enhance = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.bn = nn.BatchNorm2d(out_dim)

        self.rcab = RCAB(out_dim)
        self.edge_out = nn.Conv2d(out_dim, 1, 3, 1, 1)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        x = torch.cat([x1, x2], dim=1)

        x = self.edge_embedding_1(x)
        # x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)

        edge_feat = self.edge_conv(x)  # out_dim

        edge_feat_mask = self.edge_mask(edge_feat)  # out_dim
        edge_feat = torch.mul(edge_feat, edge_feat_mask) + self.conv_edge_enhance(edge_feat)
        edge_feat = self.bn(edge_feat)
        edge_feat = F.relu(edge_feat, inplace=False)

        edge_feat = self.rcab(edge_feat)
        edge_out = self.edge_out(edge_feat)

        return edge_feat, edge_out


def main():
    pass


if __name__ == "__main__":
    main()
