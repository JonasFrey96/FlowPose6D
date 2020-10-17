import torch 

def flow_to_pose(u_map, v_map, flow_mask, real_depth, render_depth, real_K, render_K ):
  """
  u_map: BS 480 640 
  v_map: BS 480 640 
  flow_mask: BS 480 640
  real_K: BS 3x3
  render_K: BS 3x3 
  """
  real_K_inv = torch.inverse(real_K)
  render_K_inv =  torch.inverse(render_K)
  
  for i in range(u_map.shape[0]):
    indices = torch.nonzero( flow_mask[i] )
    torch.ones( (indices.shape[0],3) )
    real_K_inv @ indices 
  # reproject depth map to 3D points

if __name__ == "__main__":
  mean = 50
  std = 20
  bs = 2
  h = 480
  w = 640
  u_map = torch.normal ( mean, std,(bs,h,w) )    
  v_map = torch.normal ( mean, std,(bs,h,w) )    
  flow_mask = torch.normal ( 0,1, (bs,h,w) ) > 0.2   
  real_depth = torch.normal ( 1.00, 0.1,(bs,h,w) ) 
  render_depth = torch.normal ( 1.00, 0.1,(bs,h,w) ) 

  cx_1 = 312.9869
  cy_1 = 241.3109
  fx_1 = 1066.778
  fy_1 = 1067.487
  real_K = torch.tensor([[fx_1,0,cx_1],[0,fy_1,cy_1],[0,0,1]])
  real_K = real_K[None,:,:].repeat(bs,1,1)  
  render_K = real_K 

  flow_to_pose( u_map,v_map,flow_mask, real_depth, render_depth, real_K, render_K)

