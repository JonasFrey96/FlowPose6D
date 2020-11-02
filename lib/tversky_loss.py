import torch
from torch import nn
import torch.nn.functional as F

#PyTorch
ALPHA = 0.5
BETA = 0.5
GAMMA = 1

class FocalTverskyLoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(FocalTverskyLoss, self).__init__()

  def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
    """[summary]

    Args:
        inputs ([type]): BS, H, W
        targets ([torch.tensor torch.long]): BS, H, W
        smooth (int, optional): [description]. Defaults to 1.
        alpha ([type], optional): [description]. Defaults to ALPHA.
        beta ([type], optional): [description]. Defaults to BETA.
        gamma ([type], optional): [description]. Defaults to GAMMA.

    Returns:
        [type]: [description]
    """        
    BS, H, W = inputs.shape
    #comment out if your model contains a sigmoid or equivalent activation layer
    #flatten label and prediction tensors
    inputs = inputs.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    
    #True Positives, False Positives & False Negatives
    TP = torch.sum(inputs == targets, dim=1)    

    FP = ((1-targets) * inputs).sum()
    FN = (targets * (1-inputs)).sum()
    
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    FocalTversky = (1 - Tversky)**gamma
                    
    return FocalTversky