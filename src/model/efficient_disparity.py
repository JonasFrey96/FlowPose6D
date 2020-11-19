import torch

from efficientnet_pytorch import EfficientNet
from torch import nn
from torchvision import transforms

def deconv(in_planes, out_planes, bias=False):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                           stride=2, padding=1, bias=bias),
        nn.LeakyReLU(0.1, inplace=True)
    )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)

def cat(x, y):
  if x == None: 
    return y
  else: 
    return torch.cat( [x,y], dim= 1)
class EfficientDisparity(nn.Module):
  def __init__(self, num_classes = 22, backbone= 'efficientnet-b1', seperate_flow_head= False, pred_flow_pyramid=True, pred_flow_pyramid_add=True, ced_real=1, ced_render=1, ced_render_d=1,ced_real_d=1):
    # tested with b6
    super().__init__()
    self.feature_extractor = EfficientNet.from_pretrained(backbone)
    self.size = self.feature_extractor.get_image_size( backbone ) 
    self.seperate_flow_head = seperate_flow_head
    self.ced_real = ced_real
    self.ced_render = ced_render
    self.ced_real_d = ced_real_d
    self.ced_render_d = ced_render_d
    self.pred_flow_pyramid_add = pred_flow_pyramid_add
    self.pred_flow_pyramid = pred_flow_pyramid
    idxs, feats, res = self.feature_extractor.layer_info( torch.ones( (4,3,self.size, self.size)))
    if ced_render_d > 0 or ced_real_d > 0:
      self.depth_backbone = True
    else: 
      self.depth_backbone = False

    if self.depth_backbone: 
      self.feature_extractor_depth = EfficientNet.from_name(backbone, in_channels=1) 

    r = res[0]
    self.idx_extract = []
    self.feature_sizes = []
    for i in range(len(idxs)):
      if r != res[i]:
        self.idx_extract.append(i-1)
        r = res[i]
        self.feature_sizes.append( feats[i-1] )
    self.idx_extract.append(len(idxs)-1)
    self.feature_sizes.append( feats[len(idxs)-1] )

    self._num_classes = num_classes
    
    dc = []
    pred_flow_pyramid = []
    upsample_flow_layers = []

    self.feature_sizes = [8] + self.feature_sizes
    
    label_feat = [16,8, num_classes]
    label_layers = []
    label_i = -1
    for i in range( 1, len(self.feature_sizes) ):
        if i == 1:
          inc_feat_0 = (int(ced_real>0) + int(ced_render>0) + int(ced_render_d>0) + int(ced_real_d>0)) * self.feature_sizes[-i ] 
        else:
          inc_feat_0 = (int(ced_real>=i) + int(ced_render>=i) + int(ced_render_d>=i) + int(ced_real_d>=i) + 1 ) * self.feature_sizes[-i]
          if self.pred_flow_pyramid_add and self.pred_flow_pyramid:
            inc_feat_0 += 2

        out_feat = self.feature_sizes[- (i+1) ] #leave this number for now on constant
        dc.append( deconv( inc_feat_0 , out_feat ) )
        print( 'Network inp:', inc_feat_0, ' out: ', out_feat )

        if i > len(self.feature_sizes)-len(label_feat):

          if label_i == -1:
            inc_feat_label = inc_feat_0
          else:
            inc_feat_label = label_feat[label_i] 
          label_i += 1
          out_feat_label = label_feat[label_i]
          label_layers.append( deconv( inc_feat_label , out_feat_label, bias=True ) )

        if self.pred_flow_pyramid:
          pred_flow_pyramid.append( predict_flow( inc_feat_0 ) )
          upsample_flow_layers.append( nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)) 

    label_layers.append( deconv(label_feat[-2], label_feat[-1], bias=True) )
    self.label_layers = nn.ModuleList(label_layers)
    self.deconvs = nn.ModuleList(dc)

    pred_flow_pyramid.append( predict_flow( self.feature_sizes[0]) )
    if self.pred_flow_pyramid:
      self.pred_flow_pyramid= nn.ModuleList( pred_flow_pyramid )
      self.upsample_flow_layers = nn.ModuleList(upsample_flow_layers)


    self.up_in = torch.nn.UpsamplingBilinear2d(size=(self.size, self.size))
    self.input_trafos = transforms.Compose([
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    self.norm_depth = transforms.Normalize([0.485,0.485], [0.229,0.229])
    self.up_out = torch.nn.UpsamplingNearest2d(size=(480, 640))
    self.up_out_bl = torch.nn.UpsamplingBilinear2d(size=(480, 640))
    self.up_nn_in= torch.nn.UpsamplingNearest2d(size=(self.size, self.size))


  def forward(self, data, idx=False, label=None):
    """Forward pass

    Args:
        data ([torch.tensor]): BS,C,H,W (C=6) if self.depth_backbone: C = 8 else: C = 6 
        idx ([torch.tensor]): BS,1 starting for first object with 0 endind with num_classes-1
        label ([type], optional): [description]. Defaults to None.

    Returns:
      flow ([torch.tensor]): BS,2,H,W
      segmentation ([torch.tensor]): BS,num_classes,H,W
    """    

    # is it smart to have the residual skip connections only for the real image of course the information should be given for the real image but therfore the network needs to learn how to fully encode the rendered image correctly
    # data BS, C, H, W
    BS,C,H,W = data.shape
    real = self.up_in(data[:,:3] )
    render =  self.up_in(data[:,3:6] )
    if self.depth_backbone:
      data[:,6:] = data[:,6:]/10000
      
    for i in range(BS):
      real[i] = self.input_trafos( real[i] ) 
      render[i] = self.input_trafos( render[i] )

    if self.depth_backbone: 
      real_d =  self.up_nn_in(data[:,6][:,None,:,:] ) 
      render_d =  self.up_nn_in(data[:,7][:,None,:,:] )
      feat_real_d = self.feature_extractor_depth.extract_features_layerwise( real_d , idx_extract = self.idx_extract[-(self.ced_real_d):])
      feat_render_d = self.feature_extractor_depth.extract_features_layerwise( render_d , idx_extract = self.idx_extract[-(self.ced_render_d):])
    feat_real  = self.feature_extractor.extract_features_layerwise( real , idx_extract = self.idx_extract)
    feat_render = self.feature_extractor.extract_features_layerwise( render, idx_extract = self.idx_extract)
    
    pred_flow_pyramid_feat = []

    x = None
    
    for j in range( 1,len( self.deconvs)+1 ):
      # calculate input: 

      # accumulate input to each layer      
      if j-1 < self.ced_real:
        x = cat( x, feat_real[-j] )
      if j-1 < self.ced_render: 
        x = cat( x, feat_render[-j])
      if j-1 < self.ced_real_d:
        x = cat( x, feat_real_d[-j])
      if j-1 < self.ced_render_d:
        x = cat( x, feat_render_d[-j])
      if j > 1 and self.pred_flow_pyramid_add:
        dim = x.shape[3]
        # upsample flow
        f_up = self.upsample_flow_layers[j-2]( pred_flow_pyramid_feat[-1]) [:,:,:dim,:dim]
        x = cat( x, f_up )
      
      # predict flow at each level
      if self.pred_flow_pyramid:
        pred_flow_pyramid_feat.append( self.pred_flow_pyramid[ j-1 ](x) )
        try:
          dim = feat_real[-(j+1)].shape[3]
          pred_flow_pyramid_feat[-1] = pred_flow_pyramid_feat[-1][:,:,:dim,:dim] 
        except:
          pass
      

      if j == len(self.deconvs) - len(self.label_layers)+2 :
        # clone features for mask prediction.
        # here the conv are with bias !!!
        segmentation = x.clone()

      # apply upcovn layer
      x = self.deconvs[j-1](x)
      try:
        dim = feat_real[-(j+1)].shape[3]
        x = x[:,:,:dim,:dim]
      except:
        pass
    
    # predict label
    for l in self.label_layers:
      segmentation = l(segmentation)
    segmentation = self.up_out(segmentation)
    # predict flow
    pred_flow_pyramid_feat.append( self.pred_flow_pyramid[-1](x) )
    pred_flow_pyramid_feat.append( self.up_out_bl( pred_flow_pyramid_feat[-1] ) )
    
    if label is None:
      label = segmentation.argmax(dim=1)

    return pred_flow_pyramid_feat, segmentation

if  __name__ == "__main__":
  model = EfficientDisparity(num_classes = 22, backbone= 'efficientnet-b2', seperate_flow_head= False, pred_flow_pyramid=True, pred_flow_pyramid_add=True, ced_real=3, ced_render=3, ced_render_d=2,ced_real_d=2)
  BS = 2
  H = 480
  W = 640
  C = 8
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  data = torch.ones( (BS,C,H,W), device=device )
  model = model.to(device)
  idx = torch.linspace(0,BS-1,BS)[:,None]
  out = model(data, idx = idx)
  
  # for i in range(0,7):
  #   model = EfficientDisparity(num_classes = 22, backbone= f'efficientnet-b{i}', connections_encoder_decoder = 2, depth_backbone = True)
  