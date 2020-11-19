from torch.nn.modules.loss import _Loss
import torch
import torch.nn.functional as F
from math import ceil

def l2(f,gt,ind):
  div = torch.clamp( torch.sum( ind[:,0,:,:], (1,2)),1)
  return  torch.sum( torch.norm( f[:,:2,:,:] * ind  - gt * ind, p=2, dim=1 ), dim=(1,2)) / div

class FlowLoss(_Loss):
  
  def __init__(self,input_res_h, input_res_w, coefficents=[0.0005,0.001,0.005,0.01,0.02,0.08,1] ):
    super(FlowLoss, self).__init__()
    ups = []
    nns = []
    for i in range(0,len(coefficents)-1 ):
        ups.append( torch.nn.UpsamplingBilinear2d(size=( ceil( input_res_h/(2**i)  ) ,ceil( input_res_w/(2**i)  )) ) )
        nns.append( torch.nn.UpsamplingNearest2d(size=( ceil( input_res_h/(2**i)  ) ,ceil( input_res_w/(2**i)  )) )  )
    self.ups = torch.nn.ModuleList(ups)
    self.nns = torch.nn.ModuleList(nns)
    self.coefficents = coefficents

  def forward(self, pred_flow, pred_mask, gt_flow):
    """
    pred_flow (list of tensors)
    pred_mask torch.tensor torch.bool BS,2,H,W
    gt_flow torch.tensor float BS,2,H,W
    """
    BS,_,H,W = gt_flow.shape
    flow_loss_l2_stack = []
    loss_sum = torch.zeros((BS),device=gt_flow.device, dtype=gt_flow.dtype)

    for j, f in enumerate(pred_flow):
      if j == len(pred_flow)-1:
        # original size
        ind_ = pred_mask
        gt = gt_flow
      else:
        # scaled
        ind_ = self.nns[-(j+1)]( pred_mask.type(torch.float32) ) == 1 
        gt = self.ups[-(j+1)]( gt_flow )
      flow_loss_l2_stack.append( l2(f, gt, ind_) )
      loss_sum += flow_loss_l2_stack[-1]*self.coefficents[j]
    
    return loss_sum, flow_loss_l2_stack
    