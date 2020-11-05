import torch

from efficientnet_pytorch import EfficientNet
from torch import nn
from torchvision import transforms

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                           stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1, inplace=True)
    )

class EfficientDisparity(nn.Module):
  def __init__(self, num_classes = 22, backbone= 'efficientnet-b1', connections_encoder_decoder = 2, depth_backbone = False):
    # tested with b6
    super().__init__()
    self.connections_encoder_decoder = connections_encoder_decoder
    self.feature_extractor = EfficientNet.from_pretrained(backbone)
    self.size = self.feature_extractor.get_image_size( backbone ) 
    idxs, feats, res = self.feature_extractor.layer_info( torch.ones( (4,3,self.size, self.size)))
    
    self.depth_backbone = depth_backbone
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
    for i in range( len(self.feature_sizes)-1 ):
      if i == 0:
        if self.depth_backbone: 
          mult = 4
        else:
          mult = 2
        dc.append( deconv(self.feature_sizes[- (i+1) ] * mult, self.feature_sizes[-(i+2)] ) )
      else:
        dc.append( deconv(self.feature_sizes[- (i+1) ] , self.feature_sizes[-(i+2)] ) )
    self.deconvs = nn.ModuleList(dc)

    self.pred_head_flow = deconv(self.feature_sizes[0], 2)
    self.pred_head_label = deconv(self.feature_sizes[0], self._num_classes)

    self.up_in = torch.nn.UpsamplingBilinear2d(size=(self.size, self.size))
    self.input_trafos = transforms.Compose([
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    self.up_out = torch.nn.UpsamplingBilinear2d(size=(480, 640))

    self.up_nn_in= torch.nn.UpsamplingNearest2d(size=(self.size, self.size))

  def forward(self, data, idx=False, label=None):
    """Forward pass

    Args:
        data ([torch.tensor]): BS,C,H,W (C=6) if self.depth_backbone: C = 8 else: C = 6 
        idx ([torch.tensor]): BS,1
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
      real_d =  self.up_nn_in(data[:,6][:,None,:,:] ) 
      render_d =  self.up_nn_in(data[:,7][:,None,:,:] )
      feat_real_d = self.feature_extractor_depth.extract_features_layerwise( real_d , idx_extract = self.idx_extract[-1:])
      feat_render_d = self.feature_extractor_depth.extract_features_layerwise( render_d , idx_extract = self.idx_extract[-1:])

    for i in range(BS):
      real[i] = self.input_trafos( real[i] ) 
      render[i] = self.input_trafos( render[i] ) 


    feat_real  = self.feature_extractor.extract_features_layerwise( real , idx_extract = self.idx_extract)
    feat_render = self.feature_extractor.extract_features_layerwise( render, idx_extract = self.idx_extract)
    
    # Residual network structure with skip connections
    if self.depth_backbone: 
      inp_up1 = torch.cat([feat_real[-1],feat_render[-1], feat_real_d[-1], feat_render_d[-1]], dim=1)
    else:
      inp_up1 = torch.cat([feat_real[-1],feat_render[-1] ], dim=1)
    
    out_deconv = self.deconvs[0](inp_up1)

    # dim is used to make sure when upsampling to have the correct shape 17-> 34 but 33 is needed. Simply delete last row/col
    for j, d in enumerate( self.deconvs[1:]):
      dim = feat_real[-2-j].shape[3]
      
      if j < self.connections_encoder_decoder: 
        inp = feat_real[-2-j] + out_deconv[:,:,:dim,:dim] 
      else:
        inp = out_deconv[:,:,:dim,:dim]
      # DEVONV with skip conncetions d( feat_real[-2-j] + out_deconv[:,:,:dim,:dim] )
      out_deconv = d( inp )
    
    # no residual for last layer. Here maybe convert to gray scale and add residual. Not sure if this would be a good idea of if this has been proofen to work before
    dim = real.shape[3]
    flow = self.pred_head_flow( out_deconv )[:,:,:dim,:dim]
    segmentation = self.pred_head_label( out_deconv)[:,:,:dim,:dim]

    flow = self.up_out(flow)
    segmentation = self.up_out(segmentation)

    if label is None:
      label = segmentation.argmax(dim=1)

    return flow, segmentation

if  __name__ == "__main__":
  from torchsummary import summary
  model = EfficientDisparity(num_classes = 22, backbone= 'efficientnet-b1', connections_encoder_decoder = 2, depth_backbone = True)
  BS = 2
  H = 480
  W = 640
  C = 8

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  data = torch.ones( (BS,C,H,W), device=device )
  model = model.to(device)
  out = model(data, idx =None)
  summary( model, (C,H,W) )