import warnings
warnings.simplefilter("ignore", UserWarning)

import copy
import datetime
import sys
import os
import time
import shutil
import argparse
import logging
import signal
import pickle


# misc
import numpy as np
import pandas as pd
import random
import sklearn
from scipy.spatial.transform import Rotation as R
from math import pi
import coloredlogs
import datetime

sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))
sys.path.append(os.path.join(os.getcwd() + '/lib'))
# src modules
from helper import pad
from loaders_v2 import ConfigLoader
from eval import *
import torch
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

coloredlogs.install()

# network dense fusion
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.loss_focal import FocalLoss
from lib.network import PoseNet, PoseRefineNet

# from lib.motion_network import MotionNetwork
# from lib.motion_loss import motion_loss
# dataset

from loaders_v2 import GenericDataset
from visu import Visualizer
from helper import re_quat, flatten_dict
from helper import get_bb_from_depth, get_bb_real_target
from deep_im import DeepIM, ViewpointManager
from helper import BoundingBox
from helper import get_delta_t_in_euclidean, compute_auc
from helper import backproject_points_batch, backproject_points, backproject_point
from deep_im import LossAddS
from rotations import *
from pixelwise_refiner import PixelwiseRefiner

import torch.autograd.profiler as profiler


def get_ref_ite(exp):
    # Hyper parameters that should  be moved to config
    refine_iterations = exp.get(
        'training', {}).get('refine_iterations', 1)
    rand = exp.get(
        'training', {}).get('refine_iterations_range', 0)
    # uniform distributions of refine iterations +- refine_iterations_range
    # default: refine_iterations = refine_iterations
    if rand > 0:
        refine_iterations = random.randrange(
            refine_iterations - rand)
    return refine_iterations


from scipy.spatial.transform import Rotation as R


def get_inital(mode, gt_rot_wxyz, gt_trans, pred_r_current, pred_t_current, cfg={}, d='cpu'):
    if mode == 'DenseFusionInit':
        pred_rot_wxyz = pred_r_current
        pred_trans = pred_t_current
    elif mode == 'TransNoise':
        n = 0
        m = cfg.get('translation_noise_inital', 0.01)
    elif mode == 'RotTransNoise':
        n = cfg.get('rot_noise_deg_inital', 10)
        m = cfg.get('translation_noise_inital', 0.01)
    elif mode == 'RotNoise':
        n = cfg.get('rot_noise_deg_inital', 10)
        m = 0
    else:
        raise AssertionError

    if not mode == 'DenseFusionInit':

        r = R.from_euler('zyx', np.random.normal(
            0, n, (gt_trans.shape[0], 3)), degrees=True)
        a = RearangeQuat(gt_trans.shape[0])
        tn = a(torch.tensor(r.as_quat(), device=d), 'xyzw')

        pred_rot_wxyz = compose_quat(
            gt_rot_wxyz, tn)
        pred_trans = torch.normal(mean=gt_trans, std=m)
    return pred_rot_wxyz, pred_trans


def ret_cropped_image(img):
    test = img.nonzero(as_tuple=False)
    a = torch.max(test[:, 0]) + 1
    b = torch.max(test[:, 1]) + 1
    c = torch.max(test[:, 2]) + 1
    return img[:a, :b, :c]

# TODO Move if finalized to other files


def padded_cat(list_of_images, device):
    """returns torch.tensor of concatenated images with dim = max size of image padded with zeros

    Args:
        list_of_images ([type]): List of Images Channels x Heigh x Width

    Returns:
        padded_cat [type]: Tensor of concatination result len(list_of_images) x Channels x max(Height) x max(Width)
        valid_indexe: len(list_of_images) x 2
    """
    c = list_of_images[0].shape[0]
    h = [x.shape[1] for x in list_of_images]
    w = [x.shape[2] for x in list_of_images]
    max_h = max(h)
    max_w = max(w)
    padded_cat = torch.zeros(
        (len(list_of_images), c, max_h, max_w), device=device)
    for i, img in enumerate(list_of_images):
        padded_cat[i, :, :h[i], :w[i]] = img

    valid_indexes = torch.tensor([h, w], device=device)
    return padded_cat, valid_indexes

# TODO Move if finalized to other files


def tight_image_batch(img_batch, device):
    ls = []
    for i in range(img_batch.shape[0]):
        ls.append(ret_cropped_image(img_batch[i]))

    tight_padded_img_batch, valid_indexes = padded_cat(
        ls,
        device=device)
    return tight_padded_img_batch


def check_exp(exp):
    if exp['model']['inital_pose']['mode'] == 'DenseFusionInit' and not exp['model']['df_load']:
        raise AssertionError

    if exp['d_test'].get('overfitting_nr_idx', -1) != -1 or exp['d_train'].get('overfitting_nr_idx', -1) != -1:
        print('Overfitting on ONE batch is activated')
        print('Overfitting on ONE batch is activated')
        print('Overfitting on ONE batch is activated')
        print('Overfitting on ONE batch is activated')
        print('Overfitting on ONE batch is activated')
        time.sleep(5)


class TrackNet6D(LightningModule):
    def __init__(self, exp, env):
        super().__init__()
        self._mode = 'init'

        # check exp for errors
        check_exp(exp)
        self._k = 0
        self.vm = None
        self.visu_forward = False
        # logging h-params
        exp_config_flatten = flatten_dict(copy.deepcopy(exp))
        for k in exp_config_flatten.keys():
            if exp_config_flatten[k] is None:
                exp_config_flatten[k] = 'is None'

        self.hparams = exp_config_flatten
        self.hparams['lr'] = exp['lr']
        self.pin_mem = True
        self.test_size = 0.1
        self.env, self.exp = env, exp

        for i in range(0, int(torch.cuda.device_count())):
            print(f'GPU {i} Type {torch.cuda.get_device_name(i)}')
        num_obj = 21

        # number of input points to the network
        num_points_small = exp['d_train']['num_pt_mesh_small']
        num_points_large = exp['d_train']['num_pt_mesh_large']

        self.pixelwise_refiner = PixelwiseRefiner(
            input_channels=6, num_classes=22, growth_rate=16)

        # df stands for DenseFusion
        if exp.get('model', {}).get('df_refine', False):
            self.df_pose_estimator = PoseNet(
                num_points=exp['d_test']['num_points'], num_obj=num_obj)

            if exp.get('model', {}).get('df_refine', False):
                self.df_refiner = PoseRefineNet(
                    num_points=exp['d_test']['num_points'], num_obj=num_obj)

            if exp.get('model', {}).get('df_load', False):
                self.df_pose_estimator.load_state_dict(
                    torch.load(exp['model']['df_pose_estimator']))
                if exp.get('model', {}).get('df_refine', False):
                    self.df_refiner.load_state_dict(
                        torch.load(exp['model']['df_refiner']))

            self.df_criterion = Loss(
                num_points_mesh=num_points_large,
                sym_list=exp['d_test']['obj_list_sym'])
            self.df_criterion_refine = Loss_refine(
                num_points_mesh=num_points_large,
                sym_list=exp['d_test']['obj_list_sym'])

        self.criterion_adds = LossAddS(sym_list=exp['d_train']['obj_list_sym'])
        self.criterion_focal = FocalLoss()
        self.best_validation = 999
        self.best_validation_patience = 5
        self.early_stopping_value = 0.1

        self.visualizer = None
        self._dict_track = {}
        self.up = torch.nn.UpsamplingBilinear2d(size=(480, 640))
        self.number_images_log_test = self.exp.get(
            'visu', {}).get('number_images_log_test', 1)
        self.counter_images_logged = 0
        self.init_train_vali_split = False

        mp = exp['model_path']
        fh = logging.FileHandler(f'{mp}/Live_Logger_Lightning.log')

        # log = open(f'{mp}/Live_Logger_Lightning.log', "a")
        # sys.stdout = log

        fh.setLevel(logging.DEBUG)
        self.start = time.time()
        logging.getLogger("lightning").addHandler(fh)
        # optional, set the logging level
        if self.exp.get('visu', {}).get('log_to_file', False):
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)
            logging.getLogger("lightning").addHandler(console)
            log = open(f'{mp}/Live_Logger_Lightning.log', "a")
            sys.stdout = log
            print('Logging to File')

    def forward(self, batch):
        st = time.time()
        if self.visualizer is None:
            self.visualizer = Visualizer(self.exp['model_path'] +
                                         '/visu/', self.logger.experiment)

        # unpack batch
        #points, choose, img, target, model_points, idx = batch[0:6]
        #depth_img, label, real_img_original, cam = batch[6:10]
        model_points = batch[4]
        idx = batch[5]
        real_img_original = batch[8]
        cam = batch[9]
        gt_rot_wxyz, gt_trans, unique_desig = batch[10:13]
        
        log_scalars = {}
        bs = model_points.shape[0]

        # check if skip
        if batch[13] is False:
            loss = torch.zeros(
                bs.shape, requires_grad=True, dtype=torch.float32, device=self.device)
            return loss, log_scalars

        real_img, render_img, real_d, render_d, gt_label_cropped = batch[13:18]
        pred_rot_wxyz, pred_trans, pred_points, h_render, h_real, render_img_original = batch[18:24]
        u_map, v_map, flow_mask = batch[24:]
        data = torch.cat([real_img, render_img], dim=1)

        # TODO idx is currently unused !!!!
        delta_v, rotations, p_label = self.pixelwise_refiner(
            data, idx)

        focal_loss = self.criterion_focal(
            p_label, gt_label_cropped)

        ind = (flow_mask == True )[:,None,:,:].repeat(1,2,1,1)
        uv_gt = torch.stack( [u_map, v_map], dim=3 ).permute(0,3,1,2)
        flow_loss = torch.sum( torch.norm( delta_v[:,:2,:,:] * ind  - uv_gt * ind, dim=1 ), dim=(1,2)) / torch.sum( ind  )
        if self.visu_forward:
            self._k += 1
            self.counter_images_logged += 1
            mask = (flow_mask == True)
            self.visualizer.plot_translations(
                f'predicted_votes_{self._mode}_nr_{self.counter_images_logged}',
                self.current_epoch,
                real_img[0].permute(1, 2, 0).cpu(),
                delta_v[0, :2, :, :].permute(1, 2, 0).cpu(),
                mask=mask[0].cpu(),
                store=True)
            
            self.visualizer.plot_translations(
                f'gt_votes_{self._mode}_nr_{self.counter_images_logged}',
                self.current_epoch,
                real_img[0].permute(1, 2, 0).cpu(),
                uv_gt.permute(0,2,3,1)[0].cpu(),
                mask=mask[0].cpu(),
                store=True)

            seg_max = p_label.argmax(dim=1)
            self.visualizer.plot_segmentation(tag=f'gt_segmentation_{self._mode}_nr_{self.counter_images_logged}',
                                                epoch=self.current_epoch,
                                                label=gt_label_cropped[0].cpu(
                                                ).numpy(),
                                                store=True)
            
            self.visualizer.plot_segmentation(tag=f'predicted_segmentation_{self._mode}_nr_{self.counter_images_logged}',
                                                epoch=self.current_epoch,
                                                label=seg_max[0].cpu(
                                                ).numpy(),
                                                store=True)
    
            self.visualizer.plot_corrospondence(tag=f'gt_flow_{self._mode}_nr_{self.counter_images_logged}',
                                                epoch=self.current_epoch,
                                                u_map=u_map[0], 
                                                v_map=v_map[0], 
                                                flow_mask=flow_mask[0], 
                                                real_img=real_img[0], 
                                                render_img=render_img[0],
                                                store=True)
            self.visualizer.plot_corrospondence(tag=f'predicted_flow_{self._mode}_nr_{self.counter_images_logged}',
                                                epoch=self.current_epoch,
                                                u_map= delta_v[0,0,:,:], 
                                                v_map= delta_v[0,1,:,:], 
                                                flow_mask=flow_mask[0], 
                                                real_img=real_img[0], 
                                                render_img=render_img[0],
                                                store=True)


        w_s = self.exp.get('loss', {}).get('weight_semantic_segmentation', 0.5)
        w_f = self.exp.get('loss', {}).get('weight_flow', 0.5)
        loss = w_s * focal_loss + w_f * flow_loss  #dis + w_t * translation_loss

        log_scalars[f'loss_segmentation'] = float(
            torch.mean(focal_loss, dim=0).detach())
        log_scalars[f'loss_flow'] = float(torch.mean(flow_loss, dim=0).detach())
        return loss, log_scalars

    def training_step(self, batch, batch_idx):
        self._mode = 'train'
        st = time.time()
        unique_desig = batch[0][12]
        total_loss = 0
        total_dis = 0
        nr = self.exp.get('visu', {}).get('number_images_log_train', 1)
        if self.counter_images_logged < nr:
            self.visu_forward = True
        else:
            self.visu_forward = False

        # forward
        dis, log_scalars = self(batch[0])

        loss = torch.mean(dis)
        
        # # for epoch average logging
        try:
            self._dict_track['train_loss  [+inf - 0]'].append(float(loss))
        except:
            self._dict_track['train_loss  [+inf - 0]'] = [float(loss)]

        # tensorboard logging
        tensorboard_logs = {'train_loss': float(loss)}
        tensorboard_logs = {**tensorboard_logs, **log_scalars}
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': {'L_Seg': log_scalars['loss_segmentation'], 'L_Flow': log_scalars['loss_flow']}}

    def validation_step(self, batch, batch_idx):
        self._mode = 'val'
        st = time.time()
        unique_desig = batch[0][12]
        total_loss = 0
        total_dis = 0

        nr = self.exp.get('visu', {}).get('number_images_log_val', 1)
        if self.counter_images_logged < nr:
            self.visu_forward = True
        else:
            self.visu_forward = False

        # forward
        dis, log_scalars = self(batch[0])
        bs = dis.shape[0]
        
        # aggregate statistics per object (ADD-S sym and ADD non sym)
        loss = torch.mean(dis)
        
        try:
            self._dict_track['val_loss  [+inf - 0]'].append(float(loss))
        except:
            self._dict_track['val_loss  [+inf - 0]'] = [float(loss)]
        
        for i in range(0, bs):
            # object loss for each object
            obj = int(unique_desig[1][i])
            obj = list(
                self.trainer.val_dataloaders[0].dataset._backend._name_to_idx_full.keys())[obj - 1]
            if f'val_{obj}_adds_dis  [+inf - 0]' in self._dict_track.keys():
                self._dict_track[f'val_{obj}_adds_dis  [+inf - 0]'].append(
                    float(dis[i]))
            else:
                self._dict_track[f'val_{obj}_adds_dis  [+inf - 0]'] = [
                    float(dis[i])]
        
        tensorboard_logs = {'val_loss': float(loss)}
        tensorboard_logs = {**tensorboard_logs, **log_scalars}

        val_loss = loss
        val_dis = loss
        return {'val_loss': val_loss, 'val_dis': val_dis, 'log': tensorboard_logs}
    # def test_step(self, batch, batch_idx):
    #     self._mode = 'test'
    #     total_loss = 0
    #     total_dis = 0

    #     st = time.time()

    #     nr = self.exp.get('visu', {}).get('number_images_log_test', 1)
    #     if self.counter_images_logged < nr:
    #         self.visu_forward = True
    #     else:
    #         self.visu_forward = False

    #     print(
    #         f'Visu Forward {self.visu_forward}, already logged {self.counter_images_logged}')

    #     # forward
    #     dis, pred_r, pred_t, log_scalars = self(batch[0])
    #     # aggregate statistics per object (ADD-S sym and ADD non sym)
    #     bs = batch[0][0].shape[0]
    #     unique_desig = batch[0][12]
    #     thr = self.exp.get('eval', {}).get('threshold_add', 0.02)
    #     # check if smaller ADD / ADD-s < 2cm
    #     within_add = torch.ge(torch.tensor(
    #         [thr] * bs, device=self.device), dis)

    #     loss = torch.mean(torch.sum(dis))

    #     self.visu_step(nr, batch, pred_r, pred_t, batch_idx)

    #     try:
    #         for _i in range(0, bs):
    #             self._dict_track['test_adds_dis  [+inf - 0]'].append(
    #                 float(dis[_i]))

    #         self._dict_track['test_loss  [+inf - 0]'].append(float(loss))
    #     except:
    #         self._dict_track['test_adds_dis  [+inf - 0]'] = [float(dis[0])]
    #         for _i in range(1, bs):
    #             self._dict_track['test_adds_dis  [+inf - 0]'].append(
    #                 float(dis[_i]))

    #         self._dict_track['test_loss  [+inf - 0]'] = [float(loss)]

    #     for i in range(0, bs):
    #         # object loss for each object
    #         obj = int(unique_desig[1][i])
    #         obj = list(
    #             self.trainer.test_dataloaders[0].dataset._backend._name_to_idx.keys())[obj - 1]
    #         if f'test_{obj}_adds_dis  [+inf - 0]' in self._dict_track.keys():
    #             self._dict_track[f'test_{obj}_adds_dis  [+inf - 0]'].append(
    #                 float(dis[i]))
    #         else:
    #             self._dict_track[f'test_{obj}_adds_dis  [+inf - 0]'] = [
    #                 float(dis[i])]
    #     for key in log_scalars.keys():
    #         try:
    #             self._dict_track[f'test_{key}'].append(
    #                 float(log_scalars[key]))
    #         except:
    #             self._dict_track[f'test_{key}'] = [
    #                 float(log_scalars[key])]

    #     tensorboard_logs = {'test_loss': float(loss)}
    #     tensorboard_logs = {**tensorboard_logs, **log_scalars}

    #     test_loss = loss
    #     test_dis = loss
    #     return {'test_loss': test_loss, 'test_dis': test_dis, 'log': tensorboard_logs}

    def visu_step(self, nr, batch, pred_r, pred_t, batch_idx):
        pass
        # if self.visu_forward:
        #     self.counter_images_logged += 1
        #     points, choose, img, target, model_points, idx = batch[0][0:6]
        #     depth_img, label_img, img_orig, cam = batch[0][6:10]
        #     gt_rot_wxyz, gt_trans, unique_desig = batch[0][10:13]
        #     self.visu_pose(batch_idx, pred_r[0], pred_t[0],
        #                    target[0], model_points[0], cam[0], img_orig[0], unique_desig, idx[0])

    def validation_epoch_end(self, outputs):
        avg_dict = {}
        self._k = 0
        self.counter_images_logged = 0  # reset image log counter

        # only keys that are logged in tensorboard are removed from log_scalars !
        for old_key in list(self._dict_track.keys()):
            if old_key.find('val') == -1:
                continue

            newk = 'avg_' + old_key
            avg_dict['avg_' +
                     old_key] = float(np.mean(np.array(self._dict_track[old_key])))

            p = old_key.find('adds_dis')
            if p != -1:
                auc = compute_auc(self._dict_track[old_key])
                avg_dict[old_key[:p] + 'auc [0 - 100]'] = auc

            self._dict_track.pop(old_key, None)

        df1 = dict_to_df(avg_dict)
        df2 = dict_to_df(get_df_dict(pre='val'))
        img = compare_df(df1, df2, key='auc [0 - 100]')
        tag = 'val_table_res_vs_df'
        img.save(self.exp['model_path'] +
                 f'/visu/{self.current_epoch}_{tag}.png')
        self.logger.experiment.add_image(tag, np.array(img).astype(
            np.uint8), global_step=self.current_epoch, dataformats='HWC')

        avg_val_dis_float = float(avg_dict['avg_val_loss  [+inf - 0]'])
        return {'avg_val_dis_float': avg_val_dis_float,
                'avg_val_dis': avg_dict['avg_val_loss  [+inf - 0]'],
                'log': avg_dict}

    def train_epoch_end(self, outputs):
        self._k = 0
        self.counter_images_logged = 0  # reset image log counter
        avg_dict = {}
        for old_key in list(self._dict_track.keys()):
            if old_key.find('train') == -1:
                continue
            avg_dict['avg_' +
                     old_key] = float(np.mean(np.array(self._dict_track[old_key])))
            self._dict_track.pop(old_key, None)
        string = 'Time for one epoch: ' + str(time.time() - self.start)
        print(string)
        self.start = time.time()
        return {**avg_dict, 'log': avg_dict}

    def test_epoch_end(self, outputs):
        self._k = 0
        self.counter_images_logged = 0  # reset image log counter
        avg_dict = {}
        # only keys that are logged in tensorboard are removed from log_scalars !
        for old_key in list(self._dict_track.keys()):
            if old_key.find('test') == -1:
                continue

            newk = 'avg_' + old_key
            avg_dict['avg_' +
                     old_key] = float(np.mean(np.array(self._dict_track[old_key])))

            p = old_key.find('adds_dis')
            if p != -1:
                auc = compute_auc(self._dict_track[old_key])
                avg_dict[old_key[:p] + 'auc [0 - 100]'] = auc

            self._dict_track.pop(old_key, None)

        avg_test_dis_float = float(avg_dict['avg_test_loss  [+inf - 0]'])

        df1 = dict_to_df(avg_dict)
        df2 = dict_to_df(get_df_dict(pre='test'))
        img = compare_df(df1, df2, key='auc [0 - 100]')
        tag = 'test_table_res_vs_df'
        img.save(self.exp['model_path'] +
                 f'/visu/{self.current_epoch}_{tag}.png')
        self.logger.experiment.add_image(tag, np.array(img).astype(
            np.uint8), global_step=self.current_epoch, dataformats='HWC')

        return {'avg_test_dis_float': avg_test_dis_float,
                'avg_test_dis': avg_dict['avg_test_loss  [+inf - 0]'],
                'log': avg_dict}

    def visu_pose(self, batch_idx, pred_r, pred_t, target, model_points, cam, img_orig, unique_desig, idx, store=True):
        pass
        # if self.visualizer is None:
        #     self.visualizer = Visualizer(self.exp['model_path'] +
        #                                  '/visu/', self.logger.experiment)
        # points = copy.deepcopy(target.detach().cpu().numpy())
        # img = img_orig.detach().cpu().numpy()
        # if self.exp['visu'].get('visu_gt', False):
        #     self.visualizer.plot_estimated_pose(tag='gt_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
        #                                         epoch=self.current_epoch,
        #                                         img=img,
        #                                         points=points,
        #                                         cam_cx=float(cam[0]),
        #                                         cam_cy=float(cam[1]),
        #                                         cam_fx=float(cam[2]),
        #                                         cam_fy=float(cam[3]),
        #                                         store=store)
        #     self.visualizer.plot_contour(tag='gt_contour_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
        #                                  epoch=self.current_epoch,
        #                                  img=img,
        #                                  points=points,
        #                                  cam_cx=float(cam[0]),
        #                                  cam_cy=float(cam[1]),
        #                                  cam_fx=float(cam[2]),
        #                                  cam_fy=float(cam[3]),
        #                                  store=store)

        # t = pred_t.detach().cpu().numpy()
        # r = pred_r.detach().cpu().numpy()

        # rot = R.from_quat(re_quat(r, 'wxyz'))

        # self.visualizer.plot_estimated_pose(tag='pred_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
        #                                     epoch=self.current_epoch,
        #                                     img=img,
        #                                     points=copy.deepcopy(
        #     model_points[:, :].detach(
        #     ).cpu().numpy()),
        #     trans=t.reshape((1, 3)),
        #     rot_mat=rot.as_matrix(),
        #     cam_cx=float(cam[0]),
        #     cam_cy=float(cam[1]),
        #     cam_fx=float(cam[2]),
        #     cam_fy=float(cam[3]),
        #     store=store)

        # self.visualizer.plot_contour(tag='pred_contour_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
        #                              epoch=self.current_epoch,
        #                              img=img,
        #                              points=copy.deepcopy(
        #     model_points[:, :].detach(
        #     ).cpu().numpy()),
        #     trans=t.reshape((1, 3)),
        #     rot_mat=rot.as_matrix(),
        #     cam_cx=float(cam[0]),
        #     cam_cy=float(cam[1]),
        #     cam_fx=float(cam[2]),
        #     cam_fy=float(cam[3]),
        #     store=store)

       
        # # get the bounding box !
        # w = 640
        # h = 480

        # real_img = torch.zeros((1, 3, h, w), device=self.device)
        # # update the target to get new bb

        # base_inital = quat_to_rot(
        #     pred_r.unsqueeze(0), 'wxyz', device=self.device).squeeze(0)
        # base_new = base_inital.view(-1, 3, 3).permute(0, 2, 1)
        # pred_points = torch.add(
        #     torch.bmm(model_points.unsqueeze(0), base_inital.unsqueeze(0)), pred_t)
        # # torch.Size([16, 2000, 3]), torch.Size([16, 4]) , torch.Size([16, 3])
        # bb_ls = get_bb_real_target(
        #     pred_points, cam.unsqueeze(0))

        # for j, b in enumerate(bb_ls):
        #     if not b.check_min_size():
        #         pass
        #     c = cam.unsqueeze(0)
        #     center_real = backproject_points(
        #         pred_t.view(1, 3), fx=c[j, 2], fy=c[j, 3], cx=c[j, 0], cy=c[j, 1])
        #     center_real = center_real.squeeze()
        #     b.move(-center_real[0], -center_real[1])
        #     b.expand(1.1)
        #     b.expand_to_correct_ratio(w, h)
        #     b.move(center_real[0], center_real[1])
        #     crop_real = b.crop(img_orig).unsqueeze(0)
        #     up = torch.nn.UpsamplingBilinear2d(size=(h, w))
        #     crop_real = torch.transpose(crop_real, 1, 3)
        #     crop_real = torch.transpose(crop_real, 2, 3)
        #     real_img[j] = up(crop_real)
        # inp = real_img[0].unsqueeze(0)
        # inp = torch.transpose(inp, 1, 3)
        # inp = torch.transpose(inp, 1, 2)
        # data = torch.cat([inp, render_img], dim=3)
        # data = torch.transpose(data, 1, 3)
        # data = torch.transpose(data, 2, 3)
        # self.visualizer.visu_network_input(tag='render_real_comp_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
        #                                    epoch=self.current_epoch,
        #                                    data=data,
        #                                    max_images=1, store=store)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{'params': self.pixelwise_refiner.parameters()}], lr=self.hparams['lr'])
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.exp['lr_cfg']['on_plateau_cfg']),
            'monitor': 'train_loss',  # Default: val_loss
            'interval': self.exp['lr_cfg']['interval'],
            'frequency': self.exp['lr_cfg']['frequency']
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset_train = GenericDataset(
            cfg_d=self.exp['d_train'],
            cfg_env=self.env)

        # initalize train and validation indices
        if not self.init_train_vali_split:
            self.init_train_vali_split = True
            self.indices_valid, self.indices_train = sklearn.model_selection.train_test_split(
                range(0, len(dataset_train)), test_size=self.test_size)

        dataset_subset = torch.utils.data.Subset(
            dataset_train, self.indices_train)

        dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                       **self.exp['loader'])

        return dataloader_train

    # def test_dataloader(self):
    #     dataset_test = GenericDataset(
    #         cfg_d=self.exp['d_test'],
    #         cfg_env=self.env)
    #     store = self.env['p_ycb'] + '/viewpoints_renderings'
    #     if self.vm is None:
    #         self.vm = ViewpointManager(
    #             store=store,
    #             name_to_idx=dataset_test._backend._name_to_idx,
    #             nr_of_images_per_object=self.exp.get(
    #                 'vm', {}).get('nr_of_images_per_object', 1000),
    #             device=self.device,
    #             load_images=self.exp.get('vm', {}).get('load_images', False))

    #     dataloader_test = torch.utils.data.DataLoader(dataset_test,
    #                                                   **self.exp['loader'])
    #     return dataloader_test

    def val_dataloader(self):
        dataset_val = GenericDataset(
            cfg_d=self.exp['d_train'],
            cfg_env=self.env)

        # initalize train and validation indices
        if not self.init_train_vali_split:
            self.init_train_vali_split = True
            self.indices_valid, self.indices_train = sklearn.model_selection.train_test_split(
                range(0, len(dataset_val)), test_size=self.test_size)

        dataset_subset = torch.utils.data.Subset(
            dataset_val, self.indices_valid)
        dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                     **self.exp['loader'])
        return dataloader_val


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)


def move_dataset_to_ssd(env, exp):
    try:
        # Update the env for the model when copying dataset to ssd
        if env.get('leonhard', {}).get('copy', False):
            files = ['data', 'data_syn', 'models', 'viewpoints_renderings']
            p_ls = os.popen('echo $TMPDIR').read().replace('\n', '')

            p_ycb_new = p_ls + '/YCB_Video_Dataset'
            p_ycb = env['p_ycb']
            print(p_ls)
            try:
                os.mkdir(p_ycb_new)
                os.mkdir('$TMPDIR/YCB_Video_Dataset')
            except:
                pass
            for f in files:

                p_file_tar = f'{p_ycb}/{f}.tar'
                logging.info(f'Copying {f} to {p_ycb_new}/{f}')

                if os.path.exists(f'{p_ycb_new}/{f}'):
                    logging.info(
                        "data already exists! Interactive session?")
                else:
                    start_time = time.time()
                    if f == 'data':
                        bashCommand = "tar -xvf" + p_file_tar + \
                            " -C $TMPDIR | awk 'BEGIN {ORS=\" \"} {if(NR%1000==0)print NR}\' "
                    else:
                        bashCommand = "tar -xvf" + p_file_tar + \
                            " -C $TMPDIR/YCB_Video_Dataset | awk 'BEGIN {ORS=\" \"} {if(NR%1000==0)print NR}\' "
                    os.system(bashCommand)
                    logging.info(
                        f'Transferred {f} folder within {str(time.time() - start_time)}s to local SSD')

            env['p_ycb'] = p_ycb_new
    except:
        env['p_ycb'] = p_ycb_new
        logging.info('Copying data failed')
    return exp, env


def move_background(env, exp):
    try:
        # Update the env for the model when copying dataset to ssd
        if env.get('leonhard', {}).get('copy', False):

            p_file_tar = env['p_background'] + '/indoorCVPR_09.tar'
            p_ls = os.popen('echo $TMPDIR').read().replace('\n', '')
            p_n = p_ls + '/Images'
            try:
                os.mkdir(p_n)
            except:
                pass

            if os.path.exists(f'{p_n}/office'):
                logging.info(
                    "data already exists! Interactive session?")
            else:
                start_time = time.time()
                bashCommand = "tar -xvf" + p_file_tar + \
                    " -C $TMPDIR | awk 'BEGIN {ORS=\" \"} {if(NR%1000==0)print NR}\' "
                os.system(bashCommand)

            env['p_background'] = p_n
    except:
        logging.info('Copying data failed')
    return exp, env


if __name__ == "__main__":
    # for reproducability
    seed_everything(42)

    def signal_handler(signal, frame):
        print('exiting on CRTL-C')
        sys.exit(0)

    # this is needed for leonhard to use interactive session and dont freeze on
    # control-C !!!!
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=file_path, default='yaml/exp/exp_ws_deepim_debug_natrix.yml',  # required=True,
                        help='The main experiment yaml file.')
    parser.add_argument('--env', type=file_path, default='yaml/env/env_natrix_jonas.yml',
                        help='The environment yaml file.')
    args = parser.parse_args()
    exp_cfg_path = args.exp
    env_cfg_path = args.env

    exp = ConfigLoader().from_file(exp_cfg_path).get_FullLoader()
    env = ConfigLoader().from_file(env_cfg_path).get_FullLoader()

    if exp['model_path'].split('/')[-2] == 'debug':
        p = '/'.join(exp['model_path'].split('/')[:-1])
        try:
            shutil.rmtree(p)
        except:
            pass
        timestamp = '_'
    else:
        timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    p = exp['model_path'].split('/')
    p.append(str(timestamp) + '_' + p.pop())
    new_path = '/'.join(p)
    exp['model_path'] = new_path
    model_path = exp['model_path']

    # copy config files to model path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print((pad("Generating network run folder")))
    else:
        print((pad("Network run folder already exits")))

    if exp.get('visu', {}).get('log_to_file', False):
        log = open(f'{model_path}/Live_Logger_Lightning.log', "a")
        sys.stdout = log
        print('Logging to File')

    exp_cfg_fn = os.path.split(exp_cfg_path)[-1]
    env_cfg_fn = os.path.split(env_cfg_path)[-1]

    print(pad(f'Copy {env_cfg_path} to {model_path}/{exp_cfg_fn}'))
    shutil.copy(exp_cfg_path, f'{model_path}/{exp_cfg_fn}')
    shutil.copy(env_cfg_path, f'{model_path}/{env_cfg_fn}')

    exp, env = move_dataset_to_ssd(env, exp)
    exp, env = move_background(env, exp)
    dic = {'exp': exp, 'env': env}
    model = TrackNet6D(**dic)

    early_stop_callback = EarlyStopping(
        monitor='avg_val_dis_float',
        patience=exp.get('early_stopping_cfg', {}).get('patience', 100),
        strict=True,
        verbose=True,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=exp['model_path'] + '/{epoch}-{avg_val_dis_float:.4f}',
        verbose=True,
        monitor="avg_val_dis",
        mode="min",
        prefix="",
        save_last=True,
        save_top_k=10,
    )
    if exp.get('checkpoint_restore', False):
        checkpoint = torch.load(
            exp['checkpoint_load'], map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

    with torch.autograd.set_detect_anomaly(True):
        trainer = Trainer(**exp['trainer'],
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        default_root_dir=exp['model_path'])


        if exp.get('model_mode', 'fit') == 'fit':
            # lr_finder = trainer.lr_find(
            #     model, min_lr=0.0000001, max_lr=0.001, num_training=50, early_stop_threshold=100)
            # lr_finder.results
            # lr_finder.suggestion()
            # print('LR FInder suggestion', lr_finder.suggestion())
            # print(lr_finder.results)
            trainer.fit(model)
            print()
        elif exp.get('model_mode', 'fit') == 'test':
            trainer.test(model)
            if exp.get('conv_test2df', False):
                command = 'python scripts/evaluation/experiment2df.py %s --write-csv --write-pkl' % (
                    model_path + '/lightning_logs/version_0')
                os.system(command)
        elif exp.get('model_mode', 'fit') == 'profile':

            trainer.test(model)
            with profiler.profile(record_shapes=True) as prof:
                with profiler.record_function("model_inference"):
                    subset_indices = [0]  # select your indices here as a list
                    subset = torch.utils.data.Subset(
                        model.train_dataloader().dataset, subset_indices)
                    testloader_subset = torch.utils.data.DataLoader(
                        subset, batch_size=1, num_workers=0, shuffle=False)
                    for inputs in testloader_subset:
                        for j, t in enumerate(inputs[0]):
                            try:
                                inputs[0][j] = inputs[0][j].cuda()
                            except:
                                pass
                        model(inputs[0])

            print(prof.key_averages().table(
                sort_by="cpu_time_total", row_limit=10))
            print(prof.key_averages(group_by_input_shape=True).table(
                sort_by="cpu_time_total", row_limit=10))
            with profiler.profile(profile_memory=True, record_shapes=True) as prof:
                # select your indices here as a list
                subset_indices = [0]
                subset = torch.utils.data.Subset(
                    model.train_dataloader().dataset, subset_indices)
                testloader_subset = torch.utils.data.DataLoader(
                    subset, batch_size=1, num_workers=0, shuffle=False)
                for inputs in testloader_subset:
                    for j, t in enumerate(inputs[0]):
                        try:
                            inputs[0][j] = inputs[0][j].cuda()
                        except:
                            pass
                    model(inputs[0])
                    print('START')
            print(prof.key_averages().table(
                sort_by="self_cpu_memory_usage", row_limit=10))

        else:
            print("Wrong model_mode defined in exp config")
            raise Exception
