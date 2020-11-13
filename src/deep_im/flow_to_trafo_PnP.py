import torch
import numpy as np
import copy
from helper import anal_tensor
from scipy.spatial.transform import Rotation as R
import cv2
def filter_pcd( pcd, tol = 0.05):
    """
    input:
        pcd : Nx3 torch.float32
    returns:
        mask : N torch.bool 
    """
    return pcd[:,2] > tol

def rvec_tvec_to_H(r_vec,t_vec):
    """
        input:
            r_vec: 3 torch.float32
            t_vec: 3 torch.float32
        returns:
            h: np.array( [4,4] )
    """
    rot = R.from_rotvec(r_vec)
    h = np.eye(4)
    h[:3,:3] = rot.as_matrix()
    h[:3,3] = t_vec.T
    return h

def get_H(pcd):
    pcd_ret = torch.ones( (pcd.shape[0],pcd.shape[1]+1),device=pcd.device, dtype=pcd.dtype )
    pcd_ret[:,:3] = pcd
    return pcd_ret

def flow_to_trafo_PnP(*args, **kwargs):
    """
    input:
      real_br: torch.tensor torch.Size([2])
      real_tl: torch.tensor torch.Size([2])
      ren_br: torch.tensor torch.Size([2])
      ren_tl: torch.tensor torch.Size([2])
      flow_mask: torch.Size([480, 640])
      u_map: torch.Size([480, 640])
      v_map: torch.Size([480, 640])
      K_ren: torch.Size([3, 3])
      render_d: torch.Size([480, 640])
      h_render: torch.Size([4, 4])
      h_real_est: torch.Size([4, 4])
    output:
      suc: bool
      h:  torch.Size([4, 4])
    """
    real_br = kwargs['real_br']
    real_tl = kwargs['real_tl']
    ren_br = kwargs['ren_br']
    ren_tl = kwargs['ren_tl']
    flow_mask = kwargs['flow_mask']
    u_map = kwargs['u_map']
    v_map = kwargs['v_map']
    K_ren = kwargs['K_ren']
    K_real = kwargs['K_real']
    render_d = kwargs['render_d']
    h_render = kwargs['h_render']
    h_real_est = kwargs['h_real_est']
    
    typ = u_map.dtype

    # Grid for upsampled real
    grid_real_h = torch.linspace(int(real_tl[0]) ,int(real_br[0]) , 480, device=u_map.device)[:,None].repeat(1,640)
    grid_real_w = torch.linspace(int(real_tl[1]) ,int(real_br[1]) , 640, device=u_map.device)[None,:].repeat(480,1)
    # Project depth map to the pointcloud real
    cam_scale = 10000
    real_pixels = torch.stack( [grid_real_w[flow_mask], grid_real_h[flow_mask], torch.ones(grid_real_h.shape, device = u_map.device,  dtype= u_map.dtype)[flow_mask]], dim=1 ).type(typ)

    
    grid_ren_h = torch.linspace(int(ren_tl[0]) ,int(ren_br[0]), 480, device=u_map.device)[:,None].repeat(1,640)
    grid_ren_w = torch.linspace(int(ren_tl[1]) ,int(ren_br[1]) , 640, device=u_map.device)[None,:].repeat(480,1)
    crop_d_pixels = torch.stack( [grid_ren_w.flatten(), grid_ren_h.flatten(), torch.ones(grid_ren_w.shape, device = u_map.device,  dtype= torch.float64).flatten()], dim=1 ).type(typ)
    K_inv = torch.inverse(K_ren.type(torch.float64)).type(typ)
    P_crop_d = K_inv @ crop_d_pixels.T.type(typ)
    P_crop_d = P_crop_d.type(torch.float64) * render_d.flatten() / cam_scale
    P_crop_d = P_crop_d.T


    render_d_ind_h = torch.linspace(0 ,479 , 480, device=u_map.device)[:,None].repeat(1,640)
    render_d_ind_w= torch.linspace(0 ,639 , 640, device=u_map.device)[None,:].repeat(480,1)
    render_d_ind_h = torch.clamp((render_d_ind_h - u_map).type(torch.float64) ,0,479).type( torch.long )[flow_mask]
    render_d_ind_w = torch.clamp((render_d_ind_w - v_map).type(torch.float32),0,639).type( torch.long )[flow_mask] 
    index = render_d_ind_h*640 + render_d_ind_w # hacky indexing along two dimensions

    P_crop_d  = P_crop_d[index] 


    m = filter_pcd( P_crop_d)

    if torch.sum(m) < 50:
        return False, torch.eye(4, dtype= u_map.dtype, device=u_map.device )
    P_crop_d  = P_crop_d[ m ]
    real_pixels = real_pixels[m]
    P_ren = P_crop_d

    # random shuffel
    pts_trafo = min(P_ren.shape[0], 50000)
    idx = torch.randperm( P_ren.shape[0] )[0:pts_trafo]
    P_ren = P_ren[idx]
    real_pixels = real_pixels[idx]

    nr = 10000

    # Move the rendered points to the origin 
    P_ren_in_origin =  (get_H( P_ren ).type(typ) @ torch.inverse( h_render.type(torch.float64) ).type(typ).T) [:,:3]

    # PNP estimation
    objectPoints = P_ren_in_origin.clone().cpu().type(torch.float32).numpy()    
    imagePoints = real_pixels[:,:2].cpu().type(torch.float32).numpy()
    dist = np.array( [[0.0,0.0,0.0,0.0]] )

    if objectPoints.shape[0] < 8:        
        print(f'Failed due to missing corsspondences ({ objectPoints.shape[0]})')
        return False, torch.eye(4, dtype= u_map.dtype, device=u_map.device )
    # set current guess as the inital estimate

    rvec = R.from_matrix(h_real_est[:3,:3].cpu().numpy()).as_rotvec().astype(np.float32)
    tvec = h_real_est[:3,3].cpu().numpy().astype(np.float32)
    # calculate PnP between the pixels coordinates in the real image and the corrosponding points in the origin frame
    r_vec2, t_vec2 = cv2.solvePnPRefineLM(copy.deepcopy(objectPoints), \
        copy.deepcopy(imagePoints), 
        K_real.cpu().type(torch.float32).numpy(), 
        dist, 
        copy.deepcopy(rvec),
        copy.deepcopy( tvec))
    r_vec2 = r_vec2[:,None]
    # retval, r_vec2, t_vec2, inliers = cv2.solvePnPRansac(copy.deepcopy(objectPoints), \
    #     copy.deepcopy(imagePoints), 
    #     K_real.cpu().numpy(), 
    #     dist, 
    #     iterationsCount = 1000, reprojectionError= 0.3)    
    
    h = rvec_tvec_to_H(r_vec2[:,0],t_vec2)

    return True, torch.tensor(h, device=u_map.device ).type(u_map.dtype) 
