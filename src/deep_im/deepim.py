import sys
import sys
import os
sys.path.append(os.getcwd() + "/src/deep_im")

from flownet import FlowNetS, flownets_bn, flownets
import torch.nn as nn
import torch
from helper import batched_index_select


class PredictionHead(nn.Module):

    def __init__(self, num_obj):
        super(PredictionHead, self).__init__()
        self.num_obj = num_obj
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=8 * 10 * 1024, out_features=256, bias=True),
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


class DeepIM(nn.Module):

    def __init__(self, num_obj):
        super(DeepIM, self).__init__()

        self.flow = FlowNetS()
        self.prediction = PredictionHead(num_obj)

    def forward(self, x, obj):
        # if self.training:
        flow2, flow3, flow4, flow5, flow6, feat = self.flow(x)
        feat.flatten()
        t, r = self.prediction(feat, obj)

        return flow2, flow3, flow4, flow5, flow6, t, r

    @ classmethod
    def from_weights(cls, num_obj, state_dict_path):
        "Initialize MyData from a file"
        model = DeepIM(num_obj)
        data = torch.load(state_dict_path)

        model.flow = flownets(data={'state_dict': data})
        return model


if __name__ == "__main__":
    model = DeepIM(num_obj=21).cuda()
    images = torch.ones((10, 6, 480, 640)).cuda()
    num_obj = torch.ones((10, 1), dtype=torch.int64).cuda()
    model(images, num_obj)

    model = DeepIM.from_weights(
        21, '/media/scratch1/jonfrey/models/pretrained_flownet/FlowNetModels/pytorch/flownets_from_caffe.pth.tar').cuda()
    out = model(images, num_obj)
    print(out)
