import torch
import numpy as np
import copy
from helper import anal_tensor

def solve_transform(keypoints, gt_keypoints):
    """
    keypoints: N x K x 3
    gt_keypoints: K x 3
    return: N x 4 x 4 transformation matrix
    """
    try:
        keypoints = keypoints.clone()
        gt_keypoints = gt_keypoints.clone()
        N, K, _ = keypoints.shape
        center = keypoints.mean(dim=1)
        gt_center = gt_keypoints.mean(dim=0)
        keypoints -= center[:, None, :]
        gt_keypoints -= gt_center[None]
        matrix = keypoints.transpose(2, 1) @ gt_keypoints[None]
        U, S, V = torch.svd(matrix)
        Vt = V.transpose(2, 1)
        Ut = U.transpose(2, 1)

        d = (V @ Ut).det()
        I = torch.eye(3, 3, dtype=gt_center.dtype, device= keypoints.device)[None].repeat(N, 1, 1)
        I[:, 2, 2] = d.clone()

        R = U @ I @ Vt
        T = torch.zeros(N, 4, 4, dtype=gt_center.dtype, device= keypoints.device)
        T[:, 0:3, 0:3] = R
        T[:, 0:3, 3] = center[None] - (R @ gt_center[None :, None])[:, :, 0]
        T[:, 3, 3] = 1.0

        return T
    except RuntimeError as error:
        import ipdb; ipdb.set_trace()
        print("Something went wrong")

def filter_pcd_given_depthmap(pcd, depth, scal= 10000):
    """
    pcd = Nx3 troch.float32
    depth = N torch.float32

    return N torch.bool
    """
    m1 = (depth/scal) > 0.2
    return m1

    m1 = depth != 0
    val_d = depth[ m1 ]
    mean = torch.mean(val_d)
    new_d = depth - mean
    tol = 0.45
    m2 = torch.abs( new_d/scal ) < tol 
    return m1 * m2
    
def filter_pcd( pcd, tol = 0.3):
    """
    input:
        pcd : Nx3 torch.float32
    returns:
        mask : NX3 torch.bool 
    """
    m = torch.mean(pcd, dim = 0)
    comp = m[None,:].repeat(pcd.shape[0],1) + tol
    mean_free = pcd-m[None,:].repeat(comp.shape[0],1)
    mask = torch.norm( mean_free,  dim= 1) > tol

    return mask[:,None].repeat(1,3) == False

def filter_pcd_cor(pcd1, pcd2, max_mean_deviation=0.2):
    
    dif = torch.norm( pcd1-pcd2 , dim= 1)
    mean = torch.mean(dif, dim = 0)
    mean_free = torch.abs(dif-mean)
    
    return mean_free < max_mean_deviation
    
def flow_to_trafo(real_br, 
                  real_tl, 
                  ren_br, 
                  ren_tl, 
                  flow_mask, 
                  u_map, 
                  v_map, 
                  K_real, 
                  K_ren, 
                  real_d, 
                  render_d, 
                  h_real, 
                  h_render):
    """
    input:
      real_br: torch.tensor torch.Size([2])
      real_tl: torch.tensor torch.Size([2])
      ren_br: torch.tensor torch.Size([2])
      ren_tl: torch.tensor torch.Size([2])
      flow_mask: torch.Size([480, 640])
      u_map: torch.Size([480, 640])
      v_map: torch.Size([480, 640])
      K_real: torch.Size([3, 3])
      K_ren: torch.Size([3, 3])
      real_d: torch.Size([480, 640]) 
      render_d: torch.Size([480, 640])
      h_real: torch.Size([4, 4])
      h_render: torch.Size([4, 4])
    output:
      P_real_in_center: torch.Size([N, 3])
      P_ren_in_center: torch.Size([N, 3]) 
      P_real_trafo: torch.Size([N, 3])
      T_res: torch.Size([4, 4])
      
      The output rotation T_res is defined in the Camera coordinate frame. 
      Therfore premultiply the T_Res with h_render to get the new h_real_new !!!
    """
    anal_tensor( real_br, 'real_br', print_on_error = True)
    anal_tensor( real_tl, 'real_tl', print_on_error = True)

    # Grid for upsampled real
    a = float(real_br[0]-real_tl[0])/480*1.0000001
    b = float(real_br[1]-real_tl[1])/640*1.0000001
    grid_real_h = torch.arange(int(real_tl[0]) ,int(real_br[0]) , a, device=u_map.device)[:,None].repeat(1,640)
    grid_real_w = torch.arange(int(real_tl[1]) ,int(real_br[1]) , b, device=u_map.device)[None,:].repeat(480,1)


    # Grid for upsampled ren
    a = float(ren_br[0]-ren_tl[0])/480*1.0000001
    b = float(ren_br[1]-ren_tl[1])/640*1.0000001
    c = 0
    
    grid_ren_h = torch.arange(int(ren_tl[0]) ,int(ren_br[0]) , a, device=u_map.device)[:,None].repeat(1,640)
    grid_ren_w = torch.arange(int(ren_tl[1]) ,int(ren_br[1]) , b, device=u_map.device)[None,:].repeat(480,1)
    # Calculate valid depth map for rendered image
    render_d_ind_h = torch.arange(0 ,480 , 1, device=u_map.device)[:,None].repeat(1,640)
    render_d_ind_w= torch.arange(0 ,640 , 1, device=u_map.device)[None,:].repeat(480,1)

    render_d_ind_h = torch.clamp(torch.round((render_d_ind_h - u_map).type(torch.float32)) ,0,479).type( torch.long )[flow_mask]
    render_d_ind_w = torch.clamp(torch.round((render_d_ind_w - v_map).type(torch.float32)),0,639).type( torch.long )[flow_mask] 
    index = render_d_ind_h*640 + render_d_ind_w # hacky indexing along two dimensions
    ren_d_masked  = render_d.flatten()[index]
    
    # Project depth map to the pointcloud real
    cam_scale = 10000

    anal_tensor( flow_mask, 'flow_mask')
    anal_tensor( grid_real_w, 'grid_real_w')
    anal_tensor( grid_real_h, 'grid_real_h')
    anal_tensor( u_map, 'u_map')
    
    real_pixels = torch.stack( [grid_real_w[flow_mask], grid_real_h[flow_mask], torch.ones(grid_real_h.shape, device = u_map.device,  dtype= u_map.dtype)[flow_mask]], dim=1 ).type(u_map.dtype)
    K_inv = torch.inverse(K_real.type(torch.float32)).type(u_map.dtype)
    P_real = K_inv @ real_pixels.T
    P_real = P_real * real_d[flow_mask] / cam_scale
    P_real = P_real.T
    
    # Project depth map to the pointcloud render
    K_ren_inv = torch.inverse(K_ren.type(torch.float32)).type(u_map.dtype)
    ren_pixels = torch.stack( [grid_ren_w[flow_mask] - v_map[flow_mask], 
                            grid_ren_h[flow_mask] - u_map[flow_mask],
                            torch.ones(grid_ren_h.shape, device = u_map.device,  dtype= u_map.dtype )[flow_mask]], 
                            dim=1 ).type(u_map.dtype)
    P_ren = K_ren_inv @ ren_pixels.T
    P_ren = P_ren * ren_d_masked / cam_scale
    P_ren = P_ren.T

    # Filter the pointclouds given the depthmap
    m_ren_depth = filter_pcd_given_depthmap(P_ren, render_d[flow_mask])
    m_real_depth = filter_pcd_given_depthmap(P_real, real_d[flow_mask])
    m_total =  m_ren_depth * m_real_depth
    
    min_points = 20
    if torch.sum(m_total) < min_points:
        print('Violated filter pcd_given_depthmap min points constrain in flow_to_trafo')
        return P_real, P_ren, P_real, torch.eye(4, dtype= u_map.dtype, device=u_map.device)

    P_ren = P_ren[m_total] 
    P_real = P_real[m_total]
    # anal_tensor(  P_ren, 'P_ren m_total masked')

    # Do not transfrom to center coordinate system
    P_real_in_center = P_real                      
    P_ren_in_center = P_ren 
    
    m_real = filter_pcd( P_real_in_center )
    m_ren = filter_pcd( P_ren_in_center )
    m_tot = m_real * m_ren
    if torch.sum(m_tot) < min_points:
        print('Violated filter pcd min points constrain in flow_to_trafo')
        return P_real, P_ren, P_real, torch.eye(4, dtype= u_map.dtype, device=u_map.device)

    P_real_in_center = P_real_in_center[m_tot[:,0]]
    P_ren_in_center = P_ren_in_center[m_tot[:,0]]
    # anal_tensor(  P_real_in_center, 'P_real_in_center m_tot masked')

    # Max mean deviation
    m_new = filter_pcd_cor(P_real_in_center, P_ren_in_center)
    
    if torch.sum(m_new) < min_points:
        print('Violated filter pcd_cor min points constrain in flow_to_trafo')
        return P_real, P_ren, P_real, torch.eye(4, dtype= u_map.dtype, device=u_map.device)

    P_real_in_center = P_real_in_center[m_new]
    P_ren_in_center = P_ren_in_center[m_new]

    
    # if ( anal_tensor(  P_real_in_center, 'P_real_in_center') or
    #     anal_tensor(  P_ren_in_center, 'P_ren_in_center') or
    #     anal_tensor(  u_map, 'U_map') ):

    #     anal_tensor(  K_ren_inv , 'K_ren_inv ')
    #     anal_tensor(  ren_pixels , 'ren_pixels ')
    #     anal_tensor(  real_pixels , 'real_pixels ')
    #     raise Exception('ERROR in flow to trafo')
    #
    # Get transformation
    pts_trafo = min( P_real_in_center.shape[0], 1000 )
    idx = torch.randperm( P_real_in_center.shape[0] )[0:pts_trafo]
    if idx.shape[0] < 10:
        print('EEEEEEEEEEEEEEEEEEEERRRRRRRRRRRRRor')
        return P_real, P_ren, P_real, torch.eye(4, dtype= u_map.dtype, device=u_map.device)

    P_real_in_center = P_real_in_center[idx]
    P_ren_in_center = P_ren_in_center[idx]

    T_res = solve_transform( P_real_in_center[None].type(torch.float64 ) , P_ren_in_center.type(torch.float64 ) ).type(u_map.dtype )
    
    # Transform the real points according to calculated transformation
    P_hr = torch.ones( (P_real_in_center.shape[0],4 ) , device=u_map.device, dtype= u_map.dtype)
    P_hr[:,:3] = P_real_in_center
    P_real_trafo = (torch.inverse( T_res[0].type(torch.float32) ).type(u_map.dtype ) @ copy.deepcopy(P_hr).T).T [:,:3]

    return P_real_in_center, P_ren_in_center, P_real_trafo, T_res[0]
    