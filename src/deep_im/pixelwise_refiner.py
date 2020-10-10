import sys
import sys
import os
sys.path.append(os.getcwd() + "/src")
sys.path.append(os.getcwd() + "/lib")
from flownet import FlowNetS, flownets_bn, flownets
import torch.nn as nn
import torch

import argparse
import os
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F


from helper import batched_index_select


class PredictionHead(nn.Module):

    def __init__(self, num_obj, in_features=81920):
        super(PredictionHead, self).__init__()
        self.num_obj = num_obj
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=256, bias=True),
            nn.LeakyReLU(0.1, inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.LeakyReLU(0.1, inplace=True))

        self.fc_trans = nn.Linear(
            in_features=256, out_features=3 * num_obj, bias=True)
        self.fc_rot = nn.Linear(
            in_features=256, out_features=4 * num_obj, bias=True)

    def forward(self, x, obj):
        x = self.fc1(torch.flatten(x, 1))
        x = self.fc2(x)
        t = self.fc_trans(x).view(-1, self.num_obj, 3)
        r = self.fc_rot(x).view(-1, self.num_obj, 4)

        t = batched_index_select(t=t, inds=obj, dim=1).squeeze(1)
        r = batched_index_select(t=r, inds=obj, dim=1).squeeze(1)

        return t, r


class PredictionHeadConv(nn.Module):

    def __init__(self, num_obj, in_features=128):
        super(PredictionHeadConv, self).__init__()
        self.num_obj = num_obj
        self.fc1 = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=64,
                      kernel_size=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True))

        self.fc2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64,
                      kernel_size=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True))

        self.fc_trans = nn.Conv1d(in_channels=64, out_channels=3 * num_obj,
                                  kernel_size=1, bias=True)
        self.fc_rot = nn.Conv1d(in_channels=64, out_channels=4 * num_obj,
                                kernel_size=1, bias=True)

    def forward(self, x, obj):
        bs, feat, h, w = x.shape
        x = x.view(bs, feat, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        t = self.fc_trans(x).view(bs, 3 * self.num_obj, h,
                                  w).view(bs, self.num_obj, 3, h, w)
        r = self.fc_rot(x).view(bs, 4 * self.num_obj, h,
                                w).view(bs, self.num_obj, 4, h, w)
        store_t = t.clone()
        t = batched_index_select(t=t, inds=obj, dim=1).squeeze(1)
        r = batched_index_select(t=r, inds=obj, dim=1).squeeze(1)

        return t, r


def pointwise_conv(in_features, maps, out_features):
    layers = []
    previous = in_features
    for feature_map in maps:
        layers.append(nn.Conv2d(previous, feature_map,
                                kernel_size=1, padding=0, bias=True))
        layers.append(nn.ELU(True))
        previous = feature_map
    layers.append(nn.Conv2d(previous, out_features,
                            kernel_size=1, padding=0, bias=True))
    return nn.Sequential(*layers)


class Conv(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features,
                              kernel_size, padding=padding, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_features)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.act(x)


class DenseBlock(nn.Module):
    def __init__(self, in_features, layers=4, k=8):
        super().__init__()
        self.convolutions = []
        for i in range(layers):
            conv = Conv(in_features + i * k, k)
            self.add_module(f'conv_{i}', conv)
            self.convolutions.append(conv)

    def forward(self, inputs):
        outputs = [inputs]
        for conv in self.convolutions:
            out = conv(torch.cat(outputs, dim=1))
            outputs.append(out)
        return torch.cat(outputs, dim=1)


class Downsample(nn.Sequential):
    def __init__(self, in_features):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_features, in_features,
                                          kernel_size=3, stride=2, padding=1, bias=False))
        self.add_module('bn', nn.BatchNorm2d(in_features))
        self.add_module('act', nn.ReLU(True))


class Upsample(nn.Sequential):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.add_module('conv', nn.ConvTranspose2d(
            in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_features))
        self.add_module('act', nn.ReLU(True))


class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, out_feat=64):
        super().__init__()
        growth_rate = 16
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7,
                      padding=3, stride=2, bias=True),
            nn.ELU(True))
        features1 = 64
        self.conv1 = DenseBlock(features1, layers=4, k=growth_rate)
        features2 = features1 + 4 * growth_rate
        self.downsample1 = Downsample(features2)
        self.conv2 = DenseBlock(features2, layers=5, k=growth_rate)
        features3 = features2 + 5 * growth_rate
        self.downsample2 = Downsample(features3)
        self.conv3 = DenseBlock(features3, layers=7, k=growth_rate)
        features4 = features3 + 7 * growth_rate
        self.downsample3 = Downsample(features4)
        features5 = features4 + 9 * growth_rate
        self.conv4 = DenseBlock(features4, layers=9, k=growth_rate)

        self.upsample1 = Upsample(features5, 256)
        features = 256 + features4
        self.conv5 = DenseBlock(features, layers=7, k=growth_rate)
        features += 7 * growth_rate
        self.upsample2 = Upsample(features, 128)
        features = 128 + features3
        self.conv6 = DenseBlock(features, layers=5, k=growth_rate)
        self.upsample3 = Upsample(features + 5 * growth_rate, 64)
        features = features1 + 64
        self.conv7 = DenseBlock(features, layers=4, k=growth_rate)
        features = features + 4 * growth_rate  # 198
        self.upsample4 = Upsample(features, out_feat)

    def forward(self, data):
        N, C, H, W = data.shape
        features = self.features(data)  # 240 x 320 x 64
        x = self.conv1(features)
        x = self.downsample1(x)  # 120 x 160
        x1 = self.conv2(x)
        x = self.downsample2(x1)  # 60 x 80
        x2 = self.conv3(x)
        x = self.downsample3(x2)  # 30 x 40
        x = self.conv4(x)

        x = self.upsample1(x)  # 60 x 80
        x = self.conv5(torch.cat([x, x2], dim=1))
        x = self.upsample2(x)  # 120 x 160
        x = self.conv6(torch.cat([x, x1], dim=1))
        x = self.upsample3(x)  # 240 x 320
        x = self.conv7(torch.cat([x, features], dim=1))  # features dim 64
        x = self.upsample4(x)
        return x


class PixelwiseRefinerJoint(nn.Module):
    def __init__(self, input_channels=6, num_classes=22, growth_rate=16):
        super().__init__()

        out_features = 128
        self.real_ext = FeatureExtractor(input_channels=3)
        self.render_ext = FeatureExtractor(input_channels=3)

        self.num_classes = num_classes

        # self.translation_head = pointwise_conv(out_features, [128, 64], 3)
        # self.rotation_head = pointwise_conv(out_features, [128, 64], 4)

        self.segmentation_head = pointwise_conv(
            out_features, [128, 64], num_classes)
        self.head = PredictionHeadConv(num_classes, in_features=128)

    def forward(self, data, idx, label=None):
        N, C, H, W = data.shape

        x_real = self.real_ext(data[:, :3, :, :])
        x_render = self.render_ext(data[:, 3:, :, :])

        x = torch.cat([x_real, x_render], dim=1)
        segmentation = self.segmentation_head(x)

        if label is None:
            label = segmentation.argmax(dim=1)
        trans, rotations = self.head(x, idx)

        return trans, rotations, segmentation


class PixelwiseRefiner(nn.Module):
    def __init__(self, input_channels=8, num_classes=22, growth_rate=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7,
                      padding=3, stride=2, bias=True),
            nn.ELU(True))
        features1 = 64
        self.conv1 = DenseBlock(features1, layers=4, k=growth_rate)
        features2 = features1 + 4 * growth_rate
        self.downsample1 = Downsample(features2)
        self.conv2 = DenseBlock(features2, layers=5, k=growth_rate)
        features3 = features2 + 5 * growth_rate
        self.downsample2 = Downsample(features3)
        self.conv3 = DenseBlock(features3, layers=7, k=growth_rate)
        features4 = features3 + 7 * growth_rate
        self.downsample3 = Downsample(features4)
        features5 = features4 + 9 * growth_rate
        self.conv4 = DenseBlock(features4, layers=9, k=growth_rate)

        self.upsample1 = Upsample(features5, 256)
        features = 256 + features4
        self.conv5 = DenseBlock(features, layers=7, k=growth_rate)
        features += 7 * growth_rate
        self.upsample2 = Upsample(features, 128)
        features = 128 + features3
        self.conv6 = DenseBlock(features, layers=5, k=growth_rate)
        self.upsample3 = Upsample(features + 5 * growth_rate, 64)
        features = features1 + 64
        self.conv7 = DenseBlock(features, layers=4, k=growth_rate)
        features = features + 4 * growth_rate  # 198

        self.upsample4 = Upsample(features, 128)
        out_features = 128

        self.num_classes = num_classes
        self.head = PredictionHeadConv(num_classes, in_features=128)

        # self.translation_head = pointwise_conv(out_features, [128, 64], 3)
        # self.rotation_head = pointwise_conv(out_features, [128, 64], 4)

        self.segmentation_head = pointwise_conv(
            out_features, [128, 64], num_classes)

    def forward(self, data, idx, label=None):
        N, C, H, W = data.shape
        features = self.features(data)  # 240 x 320 x 64
        x = self.conv1(features)
        x = self.downsample1(x)  # 120 x 160
        x1 = self.conv2(x)
        x = self.downsample2(x1)  # 60 x 80
        x2 = self.conv3(x)
        x = self.downsample3(x2)  # 30 x 40
        x = self.conv4(x)

        x = self.upsample1(x)  # 60 x 80
        x = self.conv5(torch.cat([x, x2], dim=1))
        x = self.upsample2(x)  # 120 x 160
        x = self.conv6(torch.cat([x, x1], dim=1))
        x = self.upsample3(x)  # 240 x 320
        x = self.conv7(torch.cat([x, features], dim=1))  # features dim 64
        x = self.upsample4(x)

        segmentation = self.segmentation_head(x)

        if label is None:
            label = segmentation.argmax(dim=1)
        trans, rotations = self.head(x, idx)

        # trans = self.translation_head(x)
        # rotations = self.rotation_head(x)

        return trans, rotations, segmentation


# if __name__ == "__main__":
#     c = 6
#     bs = 2
#     model = PixelwiseRefiner(input_channels=c, num_classes=21, growth_rate=16)
#     data = torch.ones((bs, c, 480, 640))
#     obj = torch.ones((bs, 1), dtype=torch.int64)
#     obj[0, 0] = 17
#     out = model(data, obj)
#     print(out)


if __name__ == "__main__":
    c = 6
    bs = 4
    model = PixelwiseRefinerJoint()
    data = torch.ones((bs, c, 480, 640))
    obj = torch.ones((bs, 1), dtype=torch.int64)
    obj[0, 0] = 17
    out = model(data, obj)
    print(out[0].shape, out[1].shape, out[2].shape)
