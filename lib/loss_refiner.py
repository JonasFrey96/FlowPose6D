from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
# from lib.knn.__init__ import KNearestNeighbor


# def loss_calculation(pred_r, pred_t, target, model_points, idx, points, num_point_mesh, sym_list):
#     pred_r = pred_r.view(1, 1, -1)
#     pred_t = pred_t.view(1, 1, -1)
#     bs, num_p, _ = pred_r.size()
#     num_input_points = len(points[0])

#     pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))

#     base = torch.cat(((1.0 - 2.0 * (pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),
#                       (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 *
#                        pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs, num_p, 1),
#                       (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 *
#                        pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1),
#                       (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 *
#                        pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs, num_p, 1),
#                       (1.0 - 2.0 * (pred_r[:, :, 1]**2 +
#                                     pred_r[:, :, 3]**2)).view(bs, num_p, 1),
#                       (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 *
#                        pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1),
#                       (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 *
#                        pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1),
#                       (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 *
#                        pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1),
#                       (1.0 - 2.0 * (pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)

#     ori_base = base
#     base = base.contiguous().transpose(2, 1).contiguous()
#     model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(
#         1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
#     target = target.view(bs, 1, num_point_mesh, 3).repeat(
#         1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
#     ori_target = target
#     pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
#     ori_t = pred_t

#     pred = torch.add(torch.bmm(model_points, base), pred_t)

#     if idx[0].item() in sym_list:
#         target = target[0].transpose(1, 0).contiguous().view(3, -1)
#         pred = pred.permute(2, 0, 1).contiguous().view(3, -1)
#         inds = KNearestNeighbor.apply(
#             target.unsqueeze(0), pred.unsqueeze(0), 1)
#         target = torch.index_select(target, 1, inds.view(-1) - 1)
#         target = target.view(
#             3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
#         pred = pred.view(
#             3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()

#     dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)

#     t = ori_t[0]
#     points = points.view(1, num_input_points, 3)

#     ori_base = ori_base[0].view(1, 3, 3).contiguous()
#     ori_t = t.repeat(bs * num_input_points,
#                      1).contiguous().view(1, bs * num_input_points, 3)
#     new_points = torch.bmm((points - ori_t), ori_base).contiguous()

#     new_target = ori_target[0].view(1, num_point_mesh, 3).contiguous()
#     ori_t = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)
#     new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

#     # print('------------> ', dis.item(), idx[0].item())
#     return dis, new_points.detach(), new_target.detach()

def knn(ref, query):
    """return indices of ref for each query point. L2 norm

    Args:
        ref ([type]): points * 3
        query ([type]): tar_points * 3

    Returns:
        [knn]: distance = query * 1 , indices = query * 1
    """
    mp2 = ref.unsqueeze(0).repeat(query.shape[0], 1, 1)
    tp2 = query.unsqueeze(1).repeat(1, ref.shape[0], 1)
    dist = torch.norm(mp2 - tp2, dim=2, p=None)
    knn = dist.topk(1, largest=False)
    return knn


def loss_calculation2(pred_r, pred_t, target, model_points, idx, points, num_point_mesh, sym_list):
    pred_r = pred_r.view(1, 1, -1)
    pred_t = pred_t.view(1, 1, -1)
    bs, num_p, _ = pred_r.size()
    num_input_points = len(points[0])

    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))

    base = torch.cat(((1.0 - 2.0 * (pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 *
                       pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs, num_p, 1),
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 *
                       pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1),
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 *
                       pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs, num_p, 1),
                      (1.0 - 2.0 * (pred_r[:, :, 1]**2 +
                                    pred_r[:, :, 3]**2)).view(bs, num_p, 1),
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 *
                       pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1),
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 *
                       pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1),
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 *
                       pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1),
                      (1.0 - 2.0 * (pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)

    ori_base = base
    base = base.contiguous().transpose(2, 1).contiguous()
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(
        1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(
        1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    ori_target = target
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t

    pred = torch.add(torch.bmm(model_points, base), pred_t)

    if idx[0].item() in sym_list:
        knn_obj = knn(
            ref=target[0, :, :], query=pred[0, :, :])
        inds = knn_obj.indices
        target[0, :, :] = target[0, inds[:, 0], :]

    dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)

    t = ori_t[0]
    points = points.view(1, num_input_points, 3)

    ori_base = ori_base[0].view(1, 3, 3).contiguous()
    ori_t = t.repeat(bs * num_input_points,
                     1).contiguous().view(1, bs * num_input_points, 3)
    new_points = torch.bmm((points - ori_t), ori_base).contiguous()

    new_target = ori_target[0].view(1, num_point_mesh, 3).contiguous()
    ori_t = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)
    new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

    # print('------------> ', dis.item(), idx[0].item())
    return dis, new_points.detach(), new_target.detach()


class Loss_refine(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss_refine, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, target, model_points, idx, points):
        return loss_calculation2(pred_r, pred_t, target, model_points, idx, points, self.num_pt_mesh, self.sym_list)
