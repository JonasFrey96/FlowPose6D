import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torch.nn.functional as F

class AtrousConvs(nn.Module):
    def __init__(self, in_features, features, dilations):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, features, kernel_size=3, stride=1, padding=1+dilations[0]-1,
                dilation=dilations[0], bias=False)
        self.conv2 = nn.Conv2d(in_features, features, kernel_size=3, stride=1, padding=1+dilations[1]-1,
                dilation=dilations[1], bias=False)
        self.conv3 = nn.Conv2d(in_features, features, kernel_size=3, stride=1, padding=1+dilations[2]-1,
                dilation=dilations[2], bias=False)
        self.conv4 = nn.Conv2d(in_features, features, kernel_size=3, stride=1, padding=1+dilations[3]-1,
                dilation=dilations[3], bias=False)
        self.bn = nn.BatchNorm2d(4 * features)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.act(self.bn(x))

def pointwise_conv(in_features, out_features):
    return nn.Sequential(AtrousConvs(in_features, 64, dilations=[1, 3, 5, 7]),
            Conv(256, 128, kernel_size=1, padding=0),
            nn.Conv2d(128, out_features, kernel_size=1, padding=0, bias=True))

class SegmentationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.atrous = AtrousConvs(in_features, 64, [1, 3, 5, 7])
        self.conv = Conv(256, 128, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(128, num_classes, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.atrous(x)
        x = self.conv(x)
        return self.conv_out(x)

class KeypointHead(nn.Module):
    def __init__(self, n_classes, in_features, out_features, objectwise_weights):
        super().__init__()
        self.objectwise_weights = objectwise_weights
        self.out_features = out_features
        self.n_out = n_classes
        self.convs = AtrousConvs(in_features, 64, [1, 3, 5, 7])
        self.conv1 = nn.Conv2d(256, 128, 1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        if self.objectwise_weights:
            self.out = nn.Conv2d(128, self.n_out * out_features,
                             1, stride=1, padding=0, bias=True)
        else:
            self.out = nn.Conv2d(128, out_features, kernel_size=1,
                    stride=1, padding=0, bias=True)

    def forward(self, x, label):
        # x: N x C x H x W
        # label: N x 1 x H x W
        x = self.convs(x)
        x = self.bn1(self.conv1(x))
        x = F.relu(x, inplace=True)
        N, C, H, W = x.shape
        if self.objectwise_weights:
            x = self.out(x).reshape(N, self.out_features, self.n_out, H, W)
            out = torch.gather(x, 2, label[:, None, None, :, :].expand(-1, self.out_features, -1, -1, -1))[:, :, 0]
        else:
            out = self.out(x)
        return out

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
    def __init__(self, in_features, layers=4, growth_rate=8):
        super().__init__()
        start = in_features // 2
        self.group_conv = Conv(in_features, start, kernel_size=1, padding=0)
        self.convolutions = []
        for i in range(0, layers):
            conv = Conv(start + i * growth_rate, growth_rate)
            self.add_module(f'conv_{i}', conv)
            self.convolutions.append(conv)

        self.d_out = start + layers * growth_rate

    def forward(self, x, inputs):
        outputs = [self.group_conv(x)]
        for conv in self.convolutions:
            out = conv(torch.cat(outputs, dim=1))
            outputs.append(out)
        out = torch.cat(outputs, dim=1)
        inputs.append(out)
        return out


class Downsample(nn.Sequential):
    def __init__(self, in_features):
        super().__init__()
        self.d_out = in_features
        self.add_module('conv', nn.Conv2d(in_features, self.d_out,
                                          kernel_size=3, stride=2, padding=1, bias=False))
        self.add_module('bn', nn.BatchNorm2d(self.d_out))
        self.add_module('act', nn.ReLU(True))

    def forward(self, x, inputs):
        return super().forward(x)


class Upsample(nn.Module):
    def __init__(self, in_features, size):
        super().__init__()
        self.size = size
        self.d_out = in_features

    def forward(self, x, *args):
        return F.interpolate(x, self.size, mode='bilinear', align_corners=True)

class ConcatBlock(nn.Module):
    def __init__(self, in_features, index):
        super().__init__()
        self.index = index

    def forward(self, x, inputs):
        y = inputs[self.index]
        return torch.cat([x, y], dim=1)

class DropoutBlock(nn.Module):
    def __init__(self, in_features, p):
        super().__init__()
        self.d_out = in_features
        self.p = p

    def forward(self, x, *args):
        return F.dropout(x, p=self.p)


BLOCK_TYPES = {
    'DenseBlock': DenseBlock,
    'Downsample': Downsample,
    'Upsample': Upsample,
    'ConcatBlock': ConcatBlock,
    'Dropout': DropoutBlock
}

class Backbone(nn.Module):
    def __init__(self, features_in, blocks):
        super().__init__()
        self.blocks = []
        features = features_in
        d_features = [features_in]
        for block in blocks:
            block_type = block['type']
            block_class = BLOCK_TYPES[block_type]
            kwargs = block.copy()
            del kwargs['type']
            instance = block_class(features, **kwargs)
            self.blocks.append(instance)

            if block_type == 'DenseBlock':
                features = instance.d_out
                d_features.append(features)
            elif block_type == 'ConcatBlock':
                features = features + d_features[block['index']]
            else:
                features = instance.d_out

        self.blocks = nn.ModuleList(self.blocks)
        self.d_out = features
        self.d_inner = d_features[len(d_features) // 2]

    def forward(self, x):
        inputs = [x]
        for block in self.blocks:
            x = block(x, inputs)
        return x, inputs[len(inputs) // 2]

class KeypointNet(nn.Module):
    def __init__(self, backbone, num_keypoints=8, num_classes=21, normals=False,
            objectwise_weights=True):
        super().__init__()
        self.normals = normals
        self.objectwise_weights = objectwise_weights
        depth_features = 6 if normals else 3
        initial_features = 32
        self.features = nn.Sequential(
            nn.Conv2d(3, initial_features, kernel_size=7,
                      padding=3, stride=2, bias=True),
            nn.ELU(True))
        self.depth_features = nn.Sequential(
                nn.Conv2d(depth_features, initial_features, kernel_size=7,
                    padding=3, stride=2, bias=True),
                nn.ELU(True))

        self.rgb_backbone = Backbone(initial_features, backbone)
        self.depth_backbone = Backbone(initial_features, backbone)
        out_features = self.rgb_backbone.d_out

        self.fuse = Conv(self.rgb_backbone.d_out + self.depth_backbone.d_out,
                out_features,
                kernel_size=1, padding=0)

        self.keypoints_out = num_keypoints * 3
        self.num_classes = num_classes
        self.keypoint_head = KeypointHead(
            num_classes, out_features + 6, self.keypoints_out, objectwise_weights)
        self.center_head = pointwise_conv(out_features + 3, 3)
        self.segmentation_head = SegmentationHead(out_features, num_classes)

    def forward(self, img, points, normals, label=None):
        N, C, H, W = img.shape
        features = self.features(img)  # 240 x 320
        if self.normals:
            points = torch.cat([points, normals], dim=1)

        depth_features = self.depth_features(points)
<<<<<<< HEAD
        x_rgb, rgb_inner = self.rgb_backbone(features)
        x_depth, depth_inner = self.depth_backbone(depth_features)
        x = torch.cat([x_rgb, x_depth], dim=1)
        x = self.fuse(x)

        x_inner = torch.cat([rgb_inner, depth_inner], dim=1)

        output_shape = x.shape[-2:]
        segmentation = self.segmentation_head(x_inner)
        segmentation = F.interpolate(segmentation, output_shape, mode='bilinear')
        seg_mask = segmentation.argmax(dim=1)

        points_small = F.interpolate(points, output_shape, mode='nearest')
=======
        x_rgb  = self.rgb_backbone(features)
        x_depth = self.depth_backbone(depth_features)
        x = torch.cat([x_rgb, x_depth], dim=1)
        x = self.fuse(x)

        segmentation = self.segmentation_head(x)

        if self.objectwise_weights and label is None:
            seg_mask = segmentation.argmax(dim=1)
        elif label is None:
            seg_mask = torch.tensor([])
        else:
            seg_mask = label

        points_small = F.interpolate(points, x.shape[-2:], mode='bilinear', align_corners=True)
>>>>>>> 375d83f38c5051866ed41fa680aba5bffd822a71
        x_points = torch.cat([x, points_small], dim=1)
        centers = self.center_head(x_points)

        x_points_centers = torch.cat([x_points, centers.detach()], dim=1)
        keypoints = self.keypoint_head(x_points_centers, seg_mask.detach())

        return keypoints, centers, segmentation

