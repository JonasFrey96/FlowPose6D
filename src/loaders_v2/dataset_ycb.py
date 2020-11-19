from helper import rotation_angle, re_quat
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.nn as nn
import torch
import time
import random
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
from os import path
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import torch.utils.data as data
from PIL import Image
import string
import math
import coloredlogs
import logging
import os
import sys
import pickle
import glob
import torchvision
from pathlib import Path


from helper import rotation_angle, re_quat
from visu import plot_pcd, plot_two_pcd
from helper import generate_unique_idx
from loaders_v2 import Backend, ConfigLoader
from helper import flatten_dict, get_bbox_480_640
from deep_im import ViewpointManager
from helper import get_bb_from_depth, get_bb_real_target, backproject_points
from rotations import *

# for flow
import cv2
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
from scipy.interpolate import griddata
class YCB(Backend):
    def __init__(self, cfg_d, cfg_env):
        super(YCB, self).__init__(cfg_d, cfg_env)
        self._cfg_d = cfg_d
        self._cfg_env = cfg_env
        self._p_ycb = cfg_env['p_ycb']
        self._pcd_cad_dict, self._name_to_idx, self._name_to_idx_full = self.get_pcd_cad_models()
        self._batch_list = self.get_batch_list()
        self.h = 480
        self.w = 640
        self._length = len(self._batch_list)
        self._norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._num_pt = cfg_d.get('num_points', 1000)
        self._trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

        if cfg_d['output_cfg'].get('color_jitter_real', {}).get('active', False):
            self._color_jitter_real = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(**cfg_d['output_cfg'].get('color_jitter_real', {}).get('cfg', False)),
                transforms.ToTensor()
            ]
            )
        if cfg_d['output_cfg'].get('color_jitter_render', {}).get('active', False):
            self._color_jitter_render = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(**cfg_d['output_cfg'].get('color_jitter_render', {}).get('cfg', False)),
                transforms.ToTensor()
            ]
            )
        if cfg_d['output_cfg'].get('norm_real', False):
            self._norm_real = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if cfg_d['output_cfg'].get('norm_render', False):
            self._norm_render = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self._minimum_num_pt = 50
        self._xmap = np.array([[j for i in range(self.w)] for j in range(self.h)])
        self._ymap = np.array([[i for i in range(self.w)] for j in range(self.h)])

        if self._cfg_d['output_cfg'].get('vm_in_dataloader', False):
            self._use_vm = True
            store = cfg_env['p_ycb'] + '/viewpoints_renderings'
            self._vm = ViewpointManager(
                store=copy.deepcopy(store),
                name_to_idx=copy.deepcopy(self._name_to_idx),
                nr_of_images_per_object=5000,
                device='cpu',
                load_images=False)
            self.up = torch.nn.UpsamplingBilinear2d(size=(self.h, self.w))
            self.up_nearest = torch.nn.UpsamplingNearest2d(size=(self.h, self.w))
            self.K_ren = self.get_camera('data_syn/0019', K=True)
        

        if self._cfg_d['noise_cfg'].get('use_input_jitter', False):
            n = self._cfg_d['noise_cfg']
            self.input_jitter = torchvision.transforms.ColorJitter(
                n['jitter_brightness'],
                n['jitter_contrast'],
                n['jitter_saturation'],
                n['jitter_hue'])
        self.input_grey = torchvision.transforms.RandomGrayscale(
            p=self._cfg_d['noise_cfg'].get('p_grey', 0))
        self._load_background()
        self.load_flow()
        self.err = False

    def getElement(self, desig, obj_idx, h_real_est=None):
        """
        desig : sequence/idx
        two problems we face. What is if an object is not visible at all -> meta['obj'] = None
        obj_idx is elemnt 1-21 !!!
        """
        img = Image.open(
            '{0}/{1}-color.png'.format(self._p_ycb, desig))
        depth = np.array(Image.open(
            '{0}/{1}-depth.png'.format(self._p_ycb, desig)))
        label = np.array(Image.open(
            '{0}/{1}-label.png'.format(self._p_ycb, desig)))
        meta = scio.loadmat(
            '{0}/{1}-meta.mat'.format(self._p_ycb, desig))
        K_cam= self.get_camera(desig, K=True, idx=False)

        img_copy = copy.copy( np.array( img) )
        if self._cfg_d['noise_cfg'].get('use_input_jitter', False):
            img = self.input_jitter(img)
        if self._cfg_d['noise_cfg'].get('p_grey', 0) > 0:
            img = self.input_grey(img)

        obj = meta['cls_indexes'].flatten().astype(np.int32)

        obj_idx_in_list = int(np.argwhere(obj == obj_idx))
        
        h_gt = np.eye(4)
        h_gt[:3,:4] =  meta['poses'][:, :, obj_idx_in_list]   
        unique_desig = (desig, obj_idx)
        
        if np.sum( label == obj_idx) <= self._minimum_num_pt:
            if self.err:
                print("Violating in min number points in get Element")
            return False

        dellist = [j for j in range(0, len(self._pcd_cad_dict[obj_idx]))]
        dellist = random.sample(dellist, len(
            self._pcd_cad_dict[obj_idx]) - self._num_pt_mesh_large)
        model_points = np.delete(self._pcd_cad_dict[obj_idx], dellist, axis=0).astype(np.float32)

        # Desig, CAD, Idx
        tup = ( unique_desig, 
                torch.from_numpy(model_points),
                torch.LongTensor([int(obj_idx) - 1]))

        if self._use_vm:
            cam_flag = self.get_camera(desig,K=False,idx=True)

            new_tup = self.get_rendered_data( np.array(img)[:,:,:3], depth, label, model_points, int(obj_idx), K_cam, cam_flag, h_gt, h_real_est)
            if new_tup is False:
                if self.err:
                    print("Violation in get render data")
                return False
            else:
                tup += new_tup

  

            if self.visu:
                # Depth map # Label # Image
                tup += (torch.from_numpy(img_copy.astype(np.float32))[:,:,:3],
                        torch.from_numpy(depth),\
                        torch.from_numpy(label) )

        else:
            raise AssertionError

        return tup

    def get_rendered_data(self, img, depth_real, label, model_points, obj_idx, K_real, cam_flag, h_gt, h_real_est=None):
        """Get Rendered Data

        Args:
            img ([np.array numpy.uint8]): H,W,3
            depth_real ([np.array numpy.int32]): H,W
            label ([np.array numpy.uint8]): H,W
            model_points ([np.array numpy.float32]): 2300,3
            obj_idx: (Int)
            K_real ([np.array numpy.float32]): 3,3
            cam_flag (Bool)
            h_gt ([np.array numpy.float32]): 4,4
            h_real_est ([np.array numpy.float32]): 4,4
        Returns:
            real_img ([torch.tensor torch.float32]): H,W,3
            render_img ([torch.tensor torch.float32]): H,W,3
            real_d ([torch.tensor torch.float32]): H,W
            render_d ([torch.tensor torch.float32]): H,W
            gt_label_cropped ([torch.tensor torch.long]): H,W
            u_cropped_scaled ([torch.tensor torch.float32]): H,W
            v_cropped_scaled([torch.tensor torch.float32]): H,W
            valid_flow_mask_cropped([torch.tensor torch.bool]): H,W
            bb ([tuple]) containing torch.tensor( real_tl, dtype=torch.int32) , torch.tensor( real_br, dtype=torch.int32) , torch.tensor( ren_tl, dtype=torch.int32) , torch.tensor( ren_br, dtype=torch.int32 )         
            h_render ([torch.tensor torch.float32]): 4,4
            h_init ([torch.tensor torch.float32]): 4,4
        """ 
        h = self.h
        w = self.w

        st = time.time()
        if not  ( h_real_est is None ): 
            h_init = h_real_est
        else:
            nt = self._cfg_d['output_cfg'].get('noise_translation', 0.03) 
            nr = self._cfg_d['output_cfg'].get('noise_rotation', 10) 
            h_init = add_noise( h_gt, nt, nr)

        # transform points
        pred_points = (model_points @ h_init[:3,:3].T) + h_init[:3,3]

        render_img = torch.zeros((3, h, w))
        render_d = torch.empty((1, h, w))

        init_rot_wxyz = torch.tensor( re_quat( R.from_matrix(h_init[:3,:3]).as_quat(), 'xyzw') )
        idx = torch.LongTensor([int(obj_idx) - 1])
        img_ren, depth_ren, h_render = self._vm.get_closest_image_batch(
            i=idx[None], rot=init_rot_wxyz[None,:], conv='wxyz')


        # rendered data BOUNDING BOX Computation
        bb_lsd = get_bb_from_depth(depth_ren)
        b_ren = bb_lsd[0]
        tl, br = b_ren.limit_bb()
        if br[0] - tl[0] < 30 or br[1] - tl[1] < 30 or b_ren.violation():
            if self.err:
                print("Violate BB in get render data for rendered bb")
            return False
        center_ren = backproject_points(
            h_render[0, :3, 3].view(1, 3), K=self.K_ren)
        center_ren = center_ren.squeeze()
        b_ren.move(-center_ren[1], -center_ren[0])
        b_ren.expand(1.1)
        b_ren.expand_to_correct_ratio(w, h)
        b_ren.move(center_ren[1], center_ren[0])
        ren_h = b_ren.height()
        ren_w = b_ren.width()
        ren_tl = b_ren.tl
        render_img = b_ren.crop(img_ren[0], scale=True, mode="bilinear") # Input H,W,C
        render_d = b_ren.crop(depth_ren[0][:,:,None], scale=True, mode="nearest") # Input H,W,C

        # real data BOUNDING BOX Computation
        u_cropped = torch.zeros((h, w), dtype=torch.long)
        v_cropped = torch.zeros((h, w), dtype=torch.long)
        gt_label_cropped = torch.zeros((h, w), dtype=torch.long)
        bb_lsd = get_bb_real_target(torch.from_numpy( pred_points[None,:,:] ), K_real[None])
        b_real = bb_lsd[0]
        tl, br = b_real.limit_bb()
        if br[0] - tl[0] < 30 or br[1] - tl[1] < 30 or b_real.violation():
            if self.err:
                print("Violate BB in get render data for real bb")
            return False
        center_real = backproject_points(
            torch.from_numpy( h_init[:3,3][None] ), K=K_real)
        center_real = center_real.squeeze()
        b_real.move(-center_real[0], -center_real[1])
        b_real.expand(1.1)
        b_real.expand_to_correct_ratio(w, h)
        b_real.move(center_real[0], center_real[1])
        real_h = b_real.height()
        real_w = b_real.width()
        real_tl = b_real.tl
        real_img = b_real.crop(torch.from_numpy(img).type(torch.float32) , scale=True, mode="bilinear")
        real_d = b_real.crop(torch.from_numpy(depth_real[:, :,None]).type(
            torch.float32), scale=True, mode="nearest")
        gt_label_cropped = b_real.crop(torch.from_numpy(label[:, :, None]).type(
            torch.float32), scale=True, mode="nearest").type(torch.int32)

        #get flow
        flow = self.get_flow(h_render[0].numpy(), h_gt, obj_idx, label, cam_flag, b_real, b_ren )
        if type( flow ) is bool: 
            if self.err:
                print("Flow calc failed")
            return False
        
        u_cropped = b_real.crop( torch.from_numpy( flow[0][:,:,None] ).type(
            torch.float32), scale=True, mode="bilinear").numpy()
        v_cropped =  b_real.crop(  torch.from_numpy( flow[1][:,:,None]).type(
            torch.float32), scale=True, mode="bilinear").numpy()
        valid_flow_mask_cropped =  b_real.crop(  torch.from_numpy( flow[2][:,:,None]).type(
            torch.float32), scale=True, mode="nearest").type(torch.bool).numpy()
       
        # scale the u and v so this is not in the uncropped space !
        v_cropped_scaled = np.zeros( v_cropped.shape, dtype=np.float32 )
        u_cropped_scaled = np.zeros( u_cropped.shape, dtype=np.float32)

        nr1 = np.full((h,w), float(w/real_w) , dtype=np.float32)
        nr2 = np.full((h,w), float(real_tl[1])  , dtype=np.float32)
        nr3 = np.full((h,w), float(ren_tl[1]) , dtype=np.float32 )
        nr4 = np.full((h,w), float(w/ren_w) , dtype=np.float32 )
        # v_cropped_scaled = (self.grid_y -((np.multiply((( np.divide( self.grid_y , nr1)+nr2) +(v_cropped[:,:,0]).astype(np.long)) - nr3 , nr4)).astype(np.long))).astype(np.long)
        v_cropped_scaled = (self.grid_y.astype(np.float32) -((np.multiply((( np.divide( self.grid_y.astype(np.float32) , nr1)+nr2) +(v_cropped[:,:,0])) - nr3 , nr4))))
        
        nr1 = np.full((h,w), float( h/real_h) , dtype=np.float32)
        nr2 = np.full((h,w), float( real_tl[0]) , dtype=np.float32)
        nr3 = np.full((h,w), float(ren_tl[0]) , dtype=np.float32)
        nr4 = np.full((h,w), float(h/ren_h) , dtype=np.float32)
        u_cropped_scaled = self.grid_x.astype(np.float32) -(np.round((((self.grid_x.astype(np.float32) /nr1)+nr2) +np.round(u_cropped[:,:,0]))-nr3)*(nr4))

        if self._cfg_d['output_cfg'].get('color_jitter_render', {}).get('active', False):
            render_img = self._color_jitter_render( render_img.permute(2,0,1) ).permute(1,2,0).type(torch.float32)
        if self._cfg_d['output_cfg'].get('color_jitter_real', {}).get('active', False):
            real_img = self._color_jitter_real( real_img.permute(2,0,1)).permute(1,2,0).type(torch.float32)
        if self._cfg_d['output_cfg'].get('norm_render', False):
            render_img = self._norm_render(render_img)
        if self._cfg_d['output_cfg'].get('norm_real', False):
            real_img = self._norm_real(real_img)
               
        tup = (real_img, render_img, \
                real_d[:,:,0], render_d[:,:,0], 
                gt_label_cropped.type(torch.long)[:,:,0],
                torch.from_numpy( u_cropped_scaled[:,:] ).type(torch.float32), 
                torch.from_numpy( v_cropped_scaled[:,:]).type(torch.float32), 
                torch.from_numpy(valid_flow_mask_cropped[:,:,0]), 
                flow[-4:],
                h_render[0].type(torch.float32),
                torch.from_numpy( h_init ).type(torch.float32),
                torch.from_numpy(h_gt).type(torch.float32),
                torch.from_numpy(K_real.astype(np.float32)))   
        if self.visu: 
            tup += (img_ren[0], depth_ren[0])

        return tup


    def get_flow(self, h_render, h_real, idx, label_img, cam, b_real, b_ren):
        st___ = time.time()
        f_1 = label_img == int( idx)
        
        min_vis_size = self._cfg_d.get('flow_cfg', {}).get('min_vis_size',200)
        if np.sum(f_1) < min_vis_size:
            # to little of the object is visible 
            return False
        st = time.time()
        m_real = copy.deepcopy(self.mesh[idx])
        m_render = copy.deepcopy(self.mesh[idx])

        m_real = self.transform_mesh(m_real, h_real)
        m_render = self.transform_mesh(m_render, h_render)

        rmi_real = RayMeshIntersector(m_real)
        rmi_render = RayMeshIntersector(m_render)
        # locations index_ray index_tri

        # crop the rays to the bounding box of the object to compute less rays
        # subsample to even compute less rays ! 
        sub = self._cfg_d.get('flow_cfg', {}).get('sub',2)

        tl, br = b_real.limit_bb()
        h_idx_real = np.reshape( self.grid_x [int(tl[0]): int(br[0]), int(tl[1]): int(br[1])][::sub,::sub], (-1) ) 
        w_idx_real = np.reshape( self.grid_y [int(tl[0]): int(br[0]), int(tl[1]): int(br[1])][::sub,::sub], (-1) ) 

        rays_origin_real = self.rays_origin_real[cam]  [int(tl[0]): int(br[0]), int(tl[1]): int(br[1])][::sub,::sub]
        rays_dir_real =self.rays_dir[cam] [int(tl[0]) : int(br[0]), int(tl[1]): int(br[1])][::sub,::sub]
        
        tl, br = b_ren.limit_bb()
        rays_origin_render = self.rays_origin_real[0] [int(tl[0]): int(br[0]), int(tl[1]): int(br[1])][::sub,::sub]
        rays_dir_render = self.rays_dir[0] [int(tl[0]): int(br[0]), int(tl[1]): int(br[1])][::sub,::sub]
        h_idx_render = np.reshape( self.grid_x [int(tl[0]): int(br[0]), int(tl[1]): int(br[1])][::sub,::sub], (-1) ) 
        w_idx_render = np.reshape( self.grid_y [int(tl[0]): int(br[0]), int(tl[1]): int(br[1])][::sub,::sub], (-1) ) 

        st_ = time.time()
        # ray traceing
        render_res_mesh_id = rmi_render.intersects_first(ray_origins= np.reshape( rays_origin_render, (-1,3) ), 
            ray_directions=np.reshape(rays_dir_render ,(-1,3)))
        real_res_mesh_id = rmi_real.intersects_first(ray_origins=np.reshape( rays_origin_real, (-1,3) ) , 
            ray_directions=np.reshape(rays_dir_real, (-1,3)))

        # a = np.reshape( rays_origin_render, (-1,3) )
        # b = np.reshape( a, rays_origin_render.shape )

        # tl, br = b_real.limit_bb()
        # c1 = np.zeros ( self.grid_x.shape ) - 1 
        # c1 [int(tl[0]): int(br[0]), int(tl[1]): int(br[1])][::sub,::sub]  = np.reshape ( real_res_mesh_id  , (rays_origin_real.shape[0], rays_origin_real.shape[1]) )

        # tl, br = b_ren.limit_bb()
        # c2 = np.zeros ( self.grid_x.shape ) - 1 
        # c2 [int(tl[0]): int(br[0]), int(tl[1]): int(br[1])][::sub,::sub]  = np.reshape ( render_res_mesh_id  , (rays_origin_render.shape[0], rays_origin_render.shape[1]) )


        f_real = real_res_mesh_id != -1 
        f_render = render_res_mesh_id != -1
        
        render_res_mesh_id = render_res_mesh_id[f_render]
        h_idx_render = h_idx_render[f_render]
        w_idx_render = w_idx_render[f_render]
        
        real_res_mesh_id = real_res_mesh_id[f_real]
        h_idx_real = h_idx_real[f_real]
        w_idx_real = w_idx_real[f_real]
  
        real_res_mesh_id.shape[0] 

        disparity_pixels = np.zeros((self.h,self.w,2))-999
        matches = 0
        i = 0
        idx_pre = np.random.permutation( np.arange(0,real_res_mesh_id.shape[0]) ).astype(np.long)
        while matches < self.max_matches and i < self.max_iterations and i < real_res_mesh_id.shape[0]:
            r_id = idx_pre[i]
            mesh_id = int(  real_res_mesh_id [r_id] )
            s = np.where(  render_res_mesh_id == mesh_id ) 
            if s[0].shape[0] > 0:
                j = s[0][0]
                _h = h_idx_real[r_id]
                _w = w_idx_real[r_id]

                if _h == -1 or _w == -1 or h_idx_render[j] == -1  or w_idx_render[j] == -1:  
                    pass
                    # print('encountered invalid pixel')
                else:
                    disparity_pixels[_h,_w,0] = h_idx_render[j] - _h
                    disparity_pixels[_h,_w,1] = w_idx_render[j] - _w
                    matches += 1
            i += 1
        
        # print(f'Rays origin real: {rays_origin_real.shape},  Rays dir: {rays_dir_real.shape}')
        # try:
            # print(f'IDX REAL max{np.max ( h_idx_real[ h_idx_real !=  -1 ] )}')
            # print(f'IDX REAL min{np.min ( h_idx_real[ h_idx_real !=  -1 ] )}')
        # except:
            # pass
        f_2 = disparity_pixels[:,:,0] != -999
        f_3 = f_2  * f_1
        points = np.where(f_3!=False)
        points = np.stack( [np.array(points[0]), np.array( points[1]) ], axis=1)
        
        min_matches = self._cfg_d.get('flow_cfg', {}).get('min_matches',50)
        if matches < 50 or np.sum(f_3) < min_matches: 
            # print(f'not enough matches{matches}, F3 {np.sum(f_3)}, REAL {h_idx_real.shape}')
            # print(render_res, rays_dir_render2.shape, rays_origin_render2.shape )
            return False
        
        u_map = griddata(points, disparity_pixels[f_3][:,0], (self.grid_x, self.grid_y), method='nearest')
        v_map = griddata(points, disparity_pixels[f_3][:,1], (self.grid_x, self.grid_y), method='nearest')

        dil_kernel_size = self._cfg_d.get('flow_cfg', {}).get('dil_kernel_size',2)
        inp = np.uint8( f_3*255 ) 
        kernel = np.ones((dil_kernel_size,dil_kernel_size),np.uint8)
        valid_flow_mask = ( cv2.dilate(inp, kernel, iterations = 1) != 0 )
        valid_flow_mask = valid_flow_mask * f_1

        real_tl = np.zeros( (2) )
        real_tl[0] = int(b_real.tl[0])
        real_tl[1] = int(b_real.tl[1])
        real_br = np.zeros( (2) )
        real_br[0] = int(b_real.br[0])
        real_br[1] = int(b_real.br[1])
        ren_tl = np.zeros( (2) )
        ren_tl[0] = int(b_ren.tl[0])
        ren_tl[1] = int(b_ren.tl[1])
        ren_br = np.zeros( (2) )
        ren_br[0] = int( b_ren.br[0] )
        ren_br[1] = int( b_ren.br[1] )
        
        return u_map, v_map, valid_flow_mask, torch.tensor( real_tl, dtype=torch.int32) , torch.tensor( real_br, dtype=torch.int32) , torch.tensor( ren_tl, dtype=torch.int32) , torch.tensor( ren_br, dtype=torch.int32 ) 

    def get_desig(self, path):
        desig = []
        with open(path) as f:
            for line in f:
                if line[-1:] == '\n':
                    desig.append(line[:-1])
                else:
                    desig.append(line)
        return desig

    def convert_desig_to_batch_list(self, desig, lookup_desig_to_obj):
        """ only works without sequence setting """
        if self._cfg_d['batch_list_cfg']['seq_length'] == 1:
            seq_list = []
            for d in desig:
                for o in lookup_desig_to_obj[d]:

                    obj_full_path = d[:-7]
                    obj_name = o
                    index_list = []
                    index_list.append(d.split('/')[-1])
                    seq_info = [obj_name, obj_full_path, index_list]
                    seq_list.append(seq_info)
        else:
            seq_added = 0
            # this method assumes that the desig list is sorted correctly
            # only adds synthetic data if present in desig list if fixed lendth = false

            seq_list = []
            # used frames keep max length to 10000 d+str(o) is the content
            used_frames = []
            mem_size = 100 * self._cfg_d['batch_list_cfg']['seq_length']
            total = len(desig)
            start = time.time()
            for j, d in enumerate(desig[::self._cfg_d['batch_list_cfg']['sub_sample']]):
                if j % 100 == 0:
                    print(f'progress: {j}/{total} time: {time.time()-start}')
                # limit memory for faster in search
                if len(used_frames) > mem_size:
                    used_frames = used_frames[-mem_size:]

                # tries to generate s sequence out of each object in the frame
                # memorize which frames we already added to a sequence
                for o in lookup_desig_to_obj[d]:
                    
                    if not d + '_obj_' + str(o) in used_frames:
                        # try to run down the full sequence

                        if d.find('syn') != -1:
                            # synthetic data detected
                            if not self._cfg_d['batch_list_cfg']['fixed_length']:
                                # add the frame to seq_list
                                # object_name, full_path, index_list
                                seq_info = [o, d, [d.split('/')[-1]]]
                                seq_list.append(seq_info)
                                used_frames.append(d + '_obj_' + str(o))


                                # cant add synthetic data because not in sequences

                        else:
                            
                            # no syn data
                            seq_idx = []
                            store = False
                            used_frames_tmp = []
                            used_frames_tmp.append(d + '_obj_' + str(o))

                            seq = int(d.split('/')[1])

                            seq_idx.append(int(desig[j*self._cfg_d['batch_list_cfg']['sub_sample']].split('/')[-1]))
                            k = j*self._cfg_d['batch_list_cfg']['sub_sample']
                            while len(seq_idx) < self._cfg_d['batch_list_cfg']['seq_length']:
                                k += self._cfg_d['batch_list_cfg']['sub_sample']
                                # check if same seq or object is not present anymore
                                if k < total:
                                    if seq != int(desig[k].split('/')[1]) or not (o in lookup_desig_to_obj[desig[k]]):
                                        if self._cfg_d['batch_list_cfg']['fixed_length']:
                                            store = False
                                            break
                                        else:
                                            store = True
                                            break
                                    else:
                                        seq_idx.append(
                                            int(desig[k].split('/')[-1]))

                                        used_frames_tmp.append(
                                            desig[k] + '_obj_' + str(o))
                                    
                                                

                                        # used_frames_tmp.append(
                                        #     desig[k] + '_obj_' + str(o))
                                else:
                                    if self._cfg_d['batch_list_cfg']['fixed_length']:
                                        store = False
                                        break
                                    else:
                                        store = True
                                        break

                            if len(seq_idx) == self._cfg_d['batch_list_cfg']['seq_length']:
                                store = True

                            if store:

                                seq_info = [o, d[:-7], seq_idx]
                                seq_list.append(seq_info)
                                used_frames += used_frames_tmp
                                store = False
        return seq_list

    def _load_background(self):
        p = self._cfg_env['p_background']
        self.background = [str(p) for p in Path(p).rglob('*.jpg')]

    def _get_background_image(self):
        seed = random.choice(self.background)
        img = Image.open(seed).convert("RGB")
        w, h = img.size
        w_g, h_g = 640, 480
        if w / h < w_g / h_g:
            h = int(w * h_g / w_g)
        else:
            w = int(h * w_g / h_g)
        crop = transforms.CenterCrop((h, w))
        img = crop(img)
        img = img.resize((w_g, h_g))
        return np.array(self._trancolor(img))

    def get_batch_list(self):
        """create batch list based on cfg"""
        lookup_arr = np.load(
            self._cfg_env['p_ycb_lookup_desig_to_obj'], allow_pickle=True)
        arr = np.array(['data_syn/000000', [20, 6, 2, 16, 8, 4]])[None, :]

        lookup_arr = np.concatenate([arr, lookup_arr])
        lookup_dict = {}
        for i in range(lookup_arr.shape[0]):
            lookup_dict[lookup_arr[i, 0]] = lookup_arr[i, 1]

        if self._cfg_d['batch_list_cfg']['mode'] == 'dense_fusion_test':
            desig_ls = self.get_desig(self._cfg_env['p_ycb_dense_test'])
            self._cfg_d['batch_list_cfg']['fixed_length'] = True
            self._cfg_d['batch_list_cfg']['seq_length'] = 1

        elif self._cfg_d['batch_list_cfg']['mode'] == 'dense_fusion_train':
            desig_ls = self.get_desig(self._cfg_env['p_ycb_dense_train'])
            self._cfg_d['batch_list_cfg']['fixed_length'] = True
            self._cfg_d['batch_list_cfg']['seq_length'] = 1

        elif self._cfg_d['batch_list_cfg']['mode'] == 'train':
            desig_ls = self.get_desig(self._cfg_env['p_ycb_seq_train'])

        elif self._cfg_d['batch_list_cfg']['mode'] == 'train_inc_syn':
            desig_ls = self.get_desig(self._cfg_env['p_ycb_seq_train_inc_syn'])

        elif self._cfg_d['batch_list_cfg']['mode'] == 'test':
            desig_ls = self.get_desig(self._cfg_env['p_ycb_seq_test'])
        else:
            raise AssertionError

        # this is needed to add noise during runtime
        self._syn = self.get_desig(self._cfg_env['p_ycb_syn'])
        self._real = self.get_desig(self._cfg_env['p_ycb_seq_train'])
        name = str(self._cfg_d['batch_list_cfg'])
        name = name.replace("""'""", '')
        name = name.replace(" ", '')
        name = name.replace(",", '_')
        name = name.replace("{", '')
        name = name.replace("}", '')
        name = name.replace(":", '')
        name = self._cfg_env['p_ycb_config'] + '/' + name + '.pkl'
        try:

            with open(name, 'rb') as f:
                batch_ls = pickle.load(f)
        except:
            batch_ls = self.convert_desig_to_batch_list(desig_ls, lookup_dict)

            pickle.dump(batch_ls, open(name, "wb"))
        return batch_ls

    def get_camera(self, desig, K=False, idx=False):
        """
        make this here simpler for cameras
        """
        
        if desig[:8] != 'data_syn' and int(desig[5:9]) >= 60:
            cx_2 = 323.7872
            cy_2 = 279.6921
            fx_2 = 1077.836
            fy_2 = 1078.189
            if K :
                return np.array([[fx_2,0,cx_2],[0,fy_2,cy_2],[0,0,1]])
            elif idx:
                return 1
            else:
                return np.array([cx_2, cy_2, fx_2, fy_2])
                
        else:
            cx_1 = 312.9869
            cy_1 = 241.3109
            fx_1 = 1066.778
            fy_1 = 1067.487
            if K:
                return np.array([[fx_1,0,cx_1],[0,fy_1,cy_1],[0,0,1]])
            elif idx:
                return 0 
            else:
                return np.array([cx_1, cy_1, fx_1, fy_1])
            
    def load_flow(self):
        self.load_rays_dir() 
        self.load_meshes()

        self.max_matches = self._cfg_d.get('flow_cfg', {}).get('max_matches',1500)
        self.max_iterations =  self._cfg_d.get('flow_cfg', {}).get('max_iterations',10000)
        self.grid_x, self.grid_y = np.mgrid[0:self.h, 0:self.w]

    def transform_mesh(self, mesh, H):
        """ directly operates on mesh and does not create a copy!"""
        t = np.ones((mesh.vertices.shape[0],4)) 
        t[:,:3] = mesh.vertices
        H[:3,:3] = H[:3,:3]
        mesh.vertices = (t @ H.T)[:,:3]
        return mesh
        
    def load_rays_dir(self): 
        K1 = self.get_camera('data_syn/000001', K=True)
        K2 = self.get_camera('data/0068/000001',  K=True)
        
        self.nr_to_image_plane = np.zeros((self.h*self.w,2), dtype=np.float)
        self.rays_origin_real = []
        self.rays_origin_render = []
        self.rays_dir = []
        
        for K in [K1,K2]:
            u_cor = np.arange(0,self.h,1)
            v_cor = np.arange(0,self.w,1)
            K_inv = np.linalg.inv(K)
            rays_dir = np.zeros((self.w,self.h,3))
            nr = 0
            rays_origin_render = np.zeros((self.w,self.h,3))
            rays_origin_real = np.zeros((self.w,self.h,3))
            for u in v_cor:
                for v in u_cor:
                    n = K_inv @ np.array([u,v, 1])
                    #n = np.array([n[1],n[0],n[2]])
                    rays_dir[u,v,:] = n * 0.6 - n * 0.25                     
                    rays_origin_render[u,v,:] = n * 0.1
                    rays_origin_real[u,v,:] =  n * 0.25
                    self.nr_to_image_plane[nr, 0] = u
                    self.nr_to_image_plane[nr, 1] = v
                    nr += 1
            rays_origin_render 
            self.rays_origin_real.append( np.swapaxes(rays_origin_real,0,1) )
            self.rays_origin_render.append( np.swapaxes(rays_origin_render,0,1) )
            self.rays_dir.append( np.swapaxes( rays_dir,0,1) )

    def load_meshes(self):
        st = time.time()
        p = self._p_ycb + '/models'
        cad_models = [str(p) for p in Path(p).rglob('*scaled.obj')] #textured
        self.mesh = {}
        for pa in cad_models:
            try:
                idx = self._name_to_idx[pa.split('/')[-2]]
                self.mesh[ idx ] = trimesh.load(pa)
            except:
                pass
        # print(f'Finished loading meshes {time.time()-st}')

    def get_pcd_cad_models(self):
        p = self._cfg_env['p_ycb_obj']
        class_file = open(p)
        cad_paths = []
        obj_idx = 1

        name_to_idx = {}
        name_to_idx_full = {}
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            name_to_idx_full[class_input[:-1]] = obj_idx
            if self._obj_list_fil is not None:
                if obj_idx in self._obj_list_fil:
                    cad_paths.append(
                        self._cfg_env['p_ycb'] + '/models/' + class_input[:-1])
                    name_to_idx[class_input[:-1]] = obj_idx
            else:
                cad_paths.append(
                    self._cfg_env['p_ycb'] + '/models/' + class_input[:-1])
                name_to_idx[class_input[:-1]] = obj_idx

            obj_idx += 1

        if len(cad_paths) == 0:
            raise AssertionError

        cad_dict = {}

        for path in cad_paths:
            input_file = open(
                '{0}/points.xyz'.format(path))

            cld = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                cld.append([float(input_line[0]), float(
                    input_line[1]), float(input_line[2])])
            cad_dict[name_to_idx[path.split('/')[-1]]] = np.array(cld)
            input_file.close()

        return cad_dict, name_to_idx, name_to_idx_full

    @ property
    def visu(self):
        return self._cfg_d['output_cfg']['visu']['status']

    @ visu.setter
    def visu(self, vis):
        self._cfg_d['output_cfg']['visu']['status'] = vis

    @ property
    def refine(self):
        return self._cfg_d['output_cfg']['refine']

    @ refine.setter
    def refine(self, refine):
        self._cfg_d['output_cfg']['refine'] = refine


def rel_h (h1,h2):
    'Input numpy arrays'
    from pytorch3d.transforms import so3_relative_angle
    return so3_relative_angle(torch.tensor( h1 ) [:3,:3][None], torch.tensor( h2 ) [:3,:3][None])
    
def add_noise(h, nt = 0.01, nr= 30):
    h_noise =np.eye(4)
    while  True:
        h_noise[:3,:3] = R.from_euler('zyx', np.random.uniform( -nr, nr, (1, 3) ) , degrees=True).as_matrix()[0]
        h_noise[:3,:3] = h_noise[:3,:3] @ h[:3,:3]
        if abs( float( rel_h(h[:3,:3], h_noise[:3,:3])/(2* float( np.math.pi) )* 360) ) < nr:
            break
    h_noise[:3,3] = np.random.normal(loc=h[:3,3], scale=nt)
    return h_noise
