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


from estimation.state import State_R3xQuat, State_SE3, points
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
            self._color_jitter_real = transforms.ColorJitter(
                **cfg_d['output_cfg'].get('color_jitter_real', {}).get('cfg', False))
        if cfg_d['output_cfg'].get('color_jitter_render', {}).get('active', False):
            self._color_jitter_render = transforms.ColorJitter(
                **cfg_d['output_cfg'].get('color_jitter_render', {}).get('cfg', False))
        if cfg_d['output_cfg'].get('norm_real', False):
            self._norm_real = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if cfg_d['output_cfg'].get('norm_render', False):
            self._norm_render = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self._front_num = 2
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

    def getElement(self, desig, obj_idx):
        """
        desig : sequence/idx
        two problems we face. What is if an object is not visible at all -> meta['obj'] = None
        obj_idx is elemnt 1-21 !!!
        """
        st = time.time()
        try:
            img = Image.open(
                '{0}/{1}-color.png'.format(self._p_ycb, desig))
            depth = np.array(Image.open(
                '{0}/{1}-depth.png'.format(self._p_ycb, desig)))
            label = np.array(Image.open(
                '{0}/{1}-label.png'.format(self._p_ycb, desig)))
            meta = scio.loadmat(
                '{0}/{1}-meta.mat'.format(self._p_ycb, desig))

        except:
            logging.error(
                'cant find files for {0}/{1}'.format(self._p_ycb, desig))
            return False
        cam = self.get_camera(desig)

        if self._cfg_d['noise_cfg'].get('use_input_jitter', False):
            img = self.input_jitter(img)

        if self._cfg_d['noise_cfg'].get('p_grey', 0) > 0:
            img = self.input_grey(img)

        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))
        mask_ind = label == 0

        add_front = False

        # TODO add here correct way to load noise
        if self._cfg_d['noise_cfg']['status'] and False:
            for k in range(5):

                seed = random.choice(self._syn)

                front = np.array(self._trancolor(Image.open(
                    '{0}/{1}-color.png'.format(self._p_ycb, desig)).convert("RGB")))

                front = np.transpose(front, (2, 0, 1))
                f_label = np.array(Image.open(
                    '{0}/{1}-label.png'.format(self._p_ycb, seed)))

                front_label = np.unique(f_label).tolist()[1:]
                if len(front_label) < self._front_num:
                    continue
                front_label = random.sample(front_label, self._front_num)
                for f_i in front_label:
                    mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
                    if f_i == front_label[0]:
                        mask_front = mk
                    else:
                        mask_front = mask_front * mk

                t_label = label * mask_front
                if len(t_label.nonzero()[0]) > 1000:
                    label = t_label
                    add_front = True
                    break

        obj = meta['cls_indexes'].flatten().astype(np.int32)

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(label, obj_idx))
        mask = mask_label * mask_depth

        obj_idx_in_list = int(np.argwhere(obj == obj_idx))
        target_r = meta['poses'][:, :, obj_idx_in_list][:, 0:3]
        target_t = np.array(
            [meta['poses'][:, :, obj_idx_in_list][:, 3:4].flatten()])

        #gt_trans = copy.deepcopy(target_t[0, :])
        #gt_rot = re_quat(R.from_matrix(target_r).as_quat(), 'xyzw')
        if self._cfg_d['noise_cfg']['status']:
            add_t = np.array(
                [random.uniform(-self._cfg_d['noise_cfg']['noise_trans'], self._cfg_d['noise_cfg']['noise_trans']) for i in range(3)])
        else:
            add_t = np.zeros(3)

        gt_rot_wxyz = re_quat(
            R.from_matrix(target_r).as_quat(), 'xyzw')
        gt_trans = np.squeeze(target_t + add_t, 0)
        unique_desig = (desig, obj_idx)

        if len(mask.nonzero()[0]) <= self._minimum_num_pt:
            return False

        # take the noise color image
        if self._cfg_d['noise_cfg']['status']:
            img = self._trancolor(img)

        rmin, rmax, cmin, cmax = get_bbox_480_640(mask_label)
        # return the pixel coordinate for the bottom left and
        # top right corner
        # cropping the image

        if desig[:8] == 'data_syn':
            back = self._get_background_image()
            img = np.array(img)[:, :, :3]
            img[mask_ind] = back[:, :, :3][mask_ind]
            img_masked = np.transpose(
                img[rmin:rmax, cmin:cmax, :], (2, 0, 1))  # 3, h_, w_

            if self._cfg_d['output_cfg']['visu']['return_img']:
                img_copy = img

        else:
            img_masked = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[
                :, rmin:rmax, cmin:cmax]

            if self._cfg_d['output_cfg']['visu']['return_img']:
                img_copy = np.array(img.convert("RGB"))

        if self._cfg_d['noise_cfg']['status'] and add_front:
            img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + \
                front[:, rmin:rmax, cmin:cmax] * \
                ~(mask_front[rmin:rmax, cmin:cmax])

        if desig[:8] == 'data_syn':
            img_masked = img_masked + \
                np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

        # check how many pixels/points are within the masked area
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        # choose is a flattend array containg all pixles/points that are part of the object
        if len(choose) > self._num_pt:
            # randomly sample some points choose since object is to big
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self._num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            # take some padding around the tiny box
            choose = np.pad(choose, (0, self._num_pt - len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten(
        )[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self._xmap[rmin:rmax, cmin:cmax].flatten(
        )[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self._ymap[rmin:rmax, cmin:cmax].flatten(
        )[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam_scale = meta['factor_depth'][0][0]
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam[0]) * pt2 / cam[2]
        pt1 = (xmap_masked - cam[1]) * pt2 / cam[3]
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        cloud = np.add(cloud, add_t)

        dellist = [j for j in range(0, len(self._pcd_cad_dict[obj_idx]))]
        if self._cfg_d['output_cfg']['refine']:
            dellist = random.sample(dellist, len(
                self._pcd_cad_dict[obj_idx]) - self._num_pt_mesh_large)
        else:
            dellist = random.sample(dellist, len(
                self._pcd_cad_dict[obj_idx]) - self._num_pt_mesh_small)
        model_points = np.delete(self._pcd_cad_dict[obj_idx], dellist, axis=0)

        # adds noise to target to regress on
        target = np.dot(model_points, target_r.T)
        target = np.add(target, target_t + add_t)

        if self._cfg_d['noise_cfg'].get('normalize_output_image_crop', True):
            torch_img = self._norm(torch.from_numpy(
                img_masked.astype(np.float32)))
        else:
            torch_img = torch.from_numpy(
                img_masked.astype(np.float32))

        if self._cfg_d['output_cfg'].get('return_same_size_tensors', False):
            # maybe not zero the image completly
            # find complete workaround to deal with choose the target and the model point cloud do we need the corrospondence between points

            padded_img = torch.zeros((3, 480, 640), dtype=torch.float32)
            sha = torch_img.shape
            padded_img[:sha[0], :sha[1], :sha[2]
                       ] = torch_img

            tup = (torch.from_numpy(cloud.astype(np.float32)),
                   torch.LongTensor(choose.astype(np.int32)),
                   padded_img,
                   torch.from_numpy(target.astype(np.float32)),
                   torch.from_numpy(model_points.astype(np.float32)),
                   torch.LongTensor([int(obj_idx) - 1]))
        else:
            tup = (torch.from_numpy(cloud.astype(np.float32)),
                   torch.LongTensor(choose.astype(np.int32)),
                   torch_img,
                   torch.from_numpy(target.astype(np.float32)),
                   torch.from_numpy(model_points.astype(np.float32)),
                   torch.LongTensor([int(obj_idx) - 1]))

        if self._cfg_d['output_cfg']['add_depth_image']:
            if self._cfg_d['output_cfg'].get('return_same_size_tensors', False):
                tup += tuple([torch.from_numpy(depth)])
            else:
                tup += tuple([torch.from_numpy(np.transpose(
                    depth[rmin:rmax, cmin:cmax], (1, 0)))])
        else:
            tup += tuple([0])

        if self._cfg_d['output_cfg'].get('add_mask_image', False):

            tup += tuple([torch.from_numpy(label)])
        else:
            tup += tuple([0])

        if self._cfg_d['output_cfg']['visu']['status']:
            # append visu information
            if self._cfg_d['output_cfg']['visu']['return_img']:
                info = (torch.from_numpy(img_copy.astype(np.float32)),
                        torch.from_numpy(cam.astype(np.float32)))
            else:
                info = (0, torch.from_numpy(cam.astype(np.float32)))

            tup += (info)
        else:
            tup += (0, 0)

        gt_rot_wxyz = re_quat(
            R.from_matrix(target_r).as_quat(), 'xyzw')
        gt_trans = np.squeeze(target_t + add_t, 0)
        unique_desig = (desig, obj_idx)

        tup = tup + (gt_rot_wxyz, gt_trans, unique_desig)

        if self._use_vm:
            # print('time to get flow:', time.time()-st)
            cam_flag= self.get_camera(desig, K=False, idx=True)

            h_real = np.eye(4)
            h_real[:3,:3] = target_r
            h_real[:3,3] = gt_trans


            new_tup = self.get_rendered_data(tup, h_real, int(obj_idx) ,label, cam_flag)
            
            if new_tup is False:
                return False
            n_tup = (0, 0, 0, 0, tup[4], tup[5] , 0, 0,*tup[8:13])
            # n_tup += new_tu
            # tup[0] = 0
            # tup[1] = 0
            # tup[2] = 0
            # tup[3] = 0
            # tup[6] = 0
            # tup[7] = 0

            n_tup = tup + new_tup
            return n_tup
        return tup

    def get_rendered_data(self, batch, h_real, obj_idx, label,cam_flag):
        h = 480
        w = 640
        nt = self._cfg_d['output_cfg'].get('noise_translation', 0.03) 
        nr = self._cfg_d['output_cfg'].get('noise_rotation', 10) 

        st = time.time()
        points, choose, img, target, model_points, idx = batch[0:6]
        depth_img, label, img_orig, cam = batch[6:10]
        gt_rot_wxyz, gt_trans, unique_desig = batch[10:13]

        # set inital translation
        init_trans = torch.normal(mean=torch.tensor(gt_trans), std=nt)

        # set inital rotaiton
        r1 = R.from_quat( re_quat( copy.copy(gt_rot_wxyz) , 'wxyz') ).as_matrix()
        animate = False
        if animate:
            try:
                self.animation_step += 1
            except:
                self.animation_step = 0
            r2 = R.from_euler('zyx', np.array([[0,self.animation_step*5,0]]), degrees=True).as_matrix()
        else:
            r2 = R.from_euler('zyx', np.random.normal(
                0, nr, (1, 3)), degrees=True).as_matrix()
        init_rot_mat = r1 @ r2
        init_rot_wxyz = torch.tensor( re_quat( R.from_matrix(init_rot_mat).as_quat()[0], 'xyzw') )
        
        # transform points
        pred_points = torch.add((model_points @ init_rot_mat[0].T), init_trans)
        
        
        render_img = torch.zeros((3, h, w))
        render_d = torch.empty((1, h, w))
        img_ren, depth, h_render = self._vm.get_closest_image_batch(
            i=idx[None], rot=init_rot_wxyz[None,:], conv='wxyz')

        bb_lsd = get_bb_from_depth(depth)
        b = bb_lsd[0]
        tl, br = b.limit_bb()
        if br[0] - tl[0] < 30 or br[1] - tl[1] < 30 or b.violation():
            print('BB to render')
            return False

        K1 = self.get_camera('data_syn/000001', K=True)
        center_ren = backproject_points(
            h_render[0, :3, 3].view(1, 3), fx=K1[0,0], fy=K1[1,1], cx=K1[0,2], cy=K1[1,2])
        center_ren = center_ren.squeeze()
        b.move(-center_ren[1], -center_ren[0])
        b.expand(1.1)
        b.expand_to_correct_ratio(w, h)
        b.move(center_ren[1], center_ren[0])
        ren_h = b.height()
        ren_w = b.width()
        ren_tl = b.tl
        b_ren = copy.deepcopy(b)
        crop_ren = b.crop(img_ren[0]).unsqueeze(0)
        crop_ren = torch.transpose(crop_ren, 1, 3)
        crop_ren = torch.transpose(crop_ren, 2, 3)
        render_img = self.up(crop_ren)[0, :, :]

        _d = depth.transpose(0, 2)
        _d = _d.transpose(0, 1)
        crop_d = b.crop(_d.type(
            torch.float32))

        crop_d = torch.transpose(crop_d, 0, 2)
        crop_d = torch.transpose(crop_d, 1, 2)
        render_d = self.up(crop_d[None])[0, :, :]

        # real data
        u_cropped = torch.zeros((h, w), dtype=torch.long)
        v_cropped = torch.zeros((h, w), dtype=torch.long)
        gt_label_cropped = torch.zeros((h, w), dtype=torch.long)
        
        bb_lsd = get_bb_real_target(pred_points[None,:,:], cam[None,:])
        b = bb_lsd[0]
        tl, br = b.limit_bb()
        if br[0] - tl[0] < 30 or br[1] - tl[1] < 30 or b.violation():
            # TODO invalid sample
            print('BB to real')
            return False
        center_real = backproject_points(
            init_trans[None], fx=cam[2], fy=cam[3], cx=cam[0], cy=cam[1])
        center_real = center_real.squeeze()
        b.move(-center_real[0], -center_real[1])
        b.expand(1.1)
        b.expand_to_correct_ratio(w, h)
        b.move(center_real[0], center_real[1])
        
        real_h = b.height()
        real_w = b.width()
        real_tl = b.tl
        b_real = copy.deepcopy(b)
        
        crop_real = b.crop(img_orig).unsqueeze(0)
        crop_real = torch.transpose(crop_real, 1, 3)
        crop_real = torch.transpose(crop_real, 2, 3)
        real_img = self.up(crop_real)[0]

        crop_d = b.crop(depth_img[:, :, None].type(
            torch.float32))[None]
        crop_d = torch.transpose(crop_d, 1, 3)
        crop_d = torch.transpose(crop_d, 2, 3)
        real_d = self.up(crop_d)[0]

        
        tmp = torch.transpose(torch.transpose(
            b.crop(label.unsqueeze(2)), 0, 2), 1, 2)
        gt_label_cropped = torch.round(self.up(tmp.type(
            torch.float32).unsqueeze(0))).clamp(0, 21)[0][0]
        
        def l_to_cropped(l):
            tmp = b.crop(torch.from_numpy(  l[:,:,None] ))
            tmp = self.up(tmp[:,:,0] [None,None,:,:] )
            return tmp[0,0]

        #get flow
        flow = self.get_flow(h_render[0].numpy(), h_real, obj_idx ,label.numpy(), cam_flag, b_real, b_ren )

        if type( flow ) is bool: 
            # print('Flow failed')
            return False
        u_cropped = l_to_cropped(  flow[0] )
        v_cropped = l_to_cropped(  flow[1] )
        valid_flow_mask_cropped =  l_to_cropped( np.float32(flow[2]) )

        # scale the u and v so this is not in the uncropped space !
        v_cropped_scaled = np.zeros( v_cropped.shape )
        u_cropped_scaled = np.zeros( u_cropped.shape )
        h = self.h
        w = self.w
        nr1 = np.full((h,w), float(w/real_w) )
        nr2 = np.full((h,w), float(real_tl[1]) )
        nr3 = np.full((h,w), float(ren_tl[1]) )
        nr4 = np.full((h,w), float(w/ren_w) )
        v_cropped_scaled = (self.grid_y -((np.multiply((( np.divide( self.grid_y , nr1)+nr2) +(v_cropped.numpy()).astype(np.long)) - nr3 , nr4)).astype(np.long))).astype(np.long)
        nr1 = np.full((h,w), float( h/real_h))
        nr2 = np.full((h,w), float( real_tl[0]))
        nr3 = np.full((h,w), float(ren_tl[0]))
        nr4 = np.full((h,w), float(h/ren_h))
        u_cropped_scaled = np.round(self.grid_x -(np.round((((self.grid_x /nr1)+nr2) +np.round(u_cropped.numpy()[:,:]))-nr3)*(nr4)))

        if self._cfg_d['output_cfg'].get('color_jitter_render', {}).get('active', False):
            render_img = self._color_jitter_render(render_img)
        if self._cfg_d['output_cfg'].get('color_jitter_real', {}).get('active', False):
            real_img = self._color_jitter_real(real_img)
        if self._cfg_d['output_cfg'].get('norm_render', False):
            render_img = self._norm_render(render_img)
        if self._cfg_d['output_cfg'].get('norm_real', False):
            real_img = self._norm_real(real_img)
        
        pred_points = 0
        return (real_img, render_img, real_d, render_d, gt_label_cropped.type(torch.long), init_rot_wxyz, init_trans, pred_points, h_render[0], torch.from_numpy(np.float32( h_real )), img_ren[0], torch.from_numpy( u_cropped_scaled ), torch.from_numpy( v_cropped_scaled), valid_flow_mask_cropped.type(torch.bool), flow[-4:])
    
    def get_flow(self, h_render, h_real, idx, label_img, cam, b_real, b_ren):
        st___ = time.time()
        f_1 = label_img == int( idx)
        if np.sum(f_1) < 200:
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
        # b_real.tl = torch.tensor([0,0])
        # b_real.br = torch.tensor([480,640])
        
        # b_ren.tl = torch.tensor([0,0])
        # b_ren.br = torch.tensor([480,640])
        sub = 2
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
        f_3 = f_2  # *f_1
        points = np.where(f_3!=False)
        points = np.stack( [np.array(points[0]), np.array( points[1]) ], axis=1)
        if matches < 50 or np.sum(f_3) < 10: 
            # print(f'not enough matches{matches}, F3 {np.sum(f_3)}, REAL {h_idx_real.shape}')
            # print(render_res, rays_dir_render2.shape, rays_origin_render2.shape )
            return False
        
        u_map = griddata(points, disparity_pixels[f_3][:,0], (self.grid_x, self.grid_y), method='nearest')
        v_map = griddata(points, disparity_pixels[f_3][:,1], (self.grid_x, self.grid_y), method='nearest')

        inp = np.uint8( f_3*255 ) 
        kernel = np.ones((6,6),np.uint8)
        valid_flow_mask = ( cv2.dilate(inp, kernel, iterations = 1) != 0 )
        valid_flow_mask = valid_flow_mask * f_1

        return u_map, v_map, valid_flow_mask, b_real.tl, b_real.br, b_ren.tl, b_ren.br 

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
            mem_size = 10 * self._cfg_d['batch_list_cfg']['seq_length']
            total = len(desig)
            start = time.time()
            for j, d in enumerate(desig):
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

                            seq_idx.append(int(desig[j].split('/')[-1]))
                            k = j
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

        self.max_matches = 40000
        self.max_iterations = 40000
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
        print(f'Finished loading meshes {time.time()-st}')

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
