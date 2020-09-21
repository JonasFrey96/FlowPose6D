from torch.nn.modules.loss import _Loss
import torch
import torch.nn.functional as F


class FocalLoss(_Loss):
    def __init__(self, gamma=2.0, alpha=0.25, size_average=True, per_batch=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.per_batch = per_batch

    def forward(self, input_x, target):
        """
        semantic: N x C x H x W
        object_ids: N x H x W
        """
        N, C, H, W = input_x.shape
        target = target
        logp = F.log_softmax(input_x, dim=1)
        logp_t = logp.gather(1, target[:, None])
        p_t = torch.exp(logp_t)
        a_t = torch.full(target.shape, self.alpha,
                         dtype=input_x.dtype, device=input_x.device)
        a_t[target == 0] = (1.0 - self.alpha)
        loss = -a_t * torch.pow(1.0 - p_t, self.gamma) * logp_t
        if self.per_batch:
            return loss.mean(dim=3).mean(dim=2).mean(dim=1)
        elif self.size_average:
            return loss.mean()
        else:
            return loss.sum()
