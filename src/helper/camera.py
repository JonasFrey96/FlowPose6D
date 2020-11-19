import torch
import numpy as np


def backproject_point(p, fx, fy, cx, cy):
    u = int(((p[0] / p[2]) * fx) + cx)
    v = int(((p[1] / p[2]) * fy) + cy)
    return u, v


def backproject_points(p, fx=None, fy=None, cx=None, cy=None, K=None):
    """
    p.shape = (nr_points,xyz)
    """
    if not K is None:
        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]
    # true_divide
    u = torch.round((torch.div(p[:, 0], p[:, 2]) * fx) + cx)
    v = torch.round((torch.div(p[:, 1], p[:, 2]) * fy) + cy)

    if torch.isnan(u).any() or torch.isnan(v).any():
        u = torch.tensor(cx).unsqueeze(0)
        v = torch.tensor(cy).unsqueeze(0)
        print('Predicted z=0 for translation. u=cx, v=cy')
        # raise Exception

    return torch.stack([v, u]).T


def backproject_points_batch(p, fx, fy, cx, cy):
    """
    p.shape = (nr_points,xyz)
    """
    bs, dim, _ = p.shape
    p = p.view(-1, 3)

    u = torch.round(torch.true_divide(p[:, 0], p[:, 2]).view(
        bs, -1) * fx.view(bs, -1).repeat(1, dim) + cx.view(bs, -1).repeat(1, dim))
    v = torch.round(torch.true_divide(p[:, 0], p[:, 1]).view(
        bs, -1) * fy.view(bs, -1).repeat(1, dim) + cy.view(bs, -1).repeat(1, dim))

    return torch.stack([u, v], dim=2)
