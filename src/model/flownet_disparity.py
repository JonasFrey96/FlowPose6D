import sys
import sys
import os
sys.path.append(os.getcwd() + "/src/deep_im")
sys.path.append(os.getcwd() + "/src/")
from flownet import FlowNetS, flownets_bn, flownets
import torch.nn as nn
import torch
from helper import batched_index_select

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                           stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1, inplace=True) )
    

class FlownetDisparity(nn.Module):

    def __init__(self, num_classes= 22):
        super(FlownetDisparity, self).__init__()

        self.flow = FlowNetS()
        
        self.up_out = torch.nn.UpsamplingNearest2d(size=(480, 640))
        self.num_classes = num_classes
       
    def forward(self, x, obj, label=None):
        bs, c, h, w = x.shape
        # if self.training:
        flow2, flow3, flow4, flow5, flow6, feat = self.flow(x)
        
        flow = self.up_out(flow2)
        segmentation = torch.zeros ((bs, self.num_classes, h, w) ,device=x.device, dtype=x.dtype) 
        if label is None:
            label = segmentation.argmax(dim=1) 
        return flow, segmentation

    @ classmethod
    def from_weights(cls, num_obj, state_dict_path):
        "Initialize MyData from a file"
        model = FlownetDisparity(num_obj)
        data = torch.load(state_dict_path)
        model.flow = flownets(data={'state_dict': data})
        return model


if __name__ == "__main__":
    model = FlownetDisparity(num_classes= 22).cuda()
    images = torch.ones((10, 6, 480, 640)).cuda()
    num_obj = torch.ones((10, 1), dtype=torch.int64).cuda()
    model(images, num_obj)

    model = FlownetDisparity.from_weights(
        22, '/media/scratch1/jonfrey/models/pretrained_flownet/FlowNetModels/pytorch/flownets_from_caffe.pth.tar').cuda()
    flow, segmentation = model(images, num_obj)
    print(flow.shape)
