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


from loaders_v2 import GenericDataset
from visu import Visualizer
from helper import re_quat, flatten_dict
from helper import get_bb_from_depth, get_bb_real_target
from deep_im import DeepIM, ViewpointManager
from helper import BoundingBox, anal_tensor
from helper import get_delta_t_in_euclidean, compute_auc
from helper import backproject_points_batch, backproject_points, backproject_point
from deep_im import LossAddS
from rotations import *
from pixelwise_refiner import PixelwiseRefiner
from model import EfficientDisparity
import torch.autograd.profiler as profiler

from deep_im import flow_to_trafo
from scipy.spatial.transform import Rotation as R

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

        

        # self.pixelwise_refiner = PixelwiseRefiner(
        #     input_channels=6, num_classes=22, growth_rate=16)
        self.pixelwise_refiner = EfficientDisparity(num_classes=22)

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
        bls = list( batch )
        for j, e in enumerate( bls ) : 
            if type(e) is torch.Tensor:
                anal_tensor( e, f'{j}', print_on_error = True)
        # unpack batch
        #points, choose, img, target, model_points, idx = batch[0:6]
        #depth_img, label, real_img_original, cam = batch[6:10]
        model_points = batch[4]
        idx = batch[5]  # Be carefull here the first objects starts with 0. Normally 0 is the NO object class in all other datastructures
        real_img_original = batch[8]
        cam = batch[9]
        gt_rot_wxyz, gt_trans, unique_desig = batch[10:13] # unique_desig[1] contains the idx starting at 1 for the first object 
        
        log_scalars = {}
        bs = model_points.shape[0]

        # check if skip
        if batch[13] is False:
            loss = torch.zeros(
                bs.shape, requires_grad=True, dtype=torch.float32, device=self.device)
            return loss, log_scalars

        real_img, render_img, real_d, render_d, gt_label_cropped = batch[13:18]
        pred_rot_wxyz, pred_trans, pred_points, h_render, h_real, render_img_original = batch[18:24]
        u_map, v_map, flow_mask, bb = batch[24:]
        data = torch.cat([real_img, render_img], dim=1)

        # TODO idx is currently unused !!!!
        
        flow, p_label = self.pixelwise_refiner(
            data, idx)

        focal_loss = self.criterion_focal(
            p_label, gt_label_cropped)

        ind = (flow_mask == True )[:,None,:,:].repeat(1,2,1,1)
        uv_gt = torch.stack( [u_map, v_map], dim=3 ).permute(0,3,1,2)
        flow_loss = torch.sum( torch.norm( flow[:,:2,:,:] * ind  - uv_gt * ind, dim=1 ), dim=(1,2)) / torch.sum( ind[:,0,:,:], (1,2))
        if torch.any( torch.sum( ind[:,0,:,:], (1,2)) < 30 ): 
            print( 'Invalid Flow Mask' )
        if self.visu_forward or self.exp.get('visu', {}).get('always_calculate', False): 
            real_tl, real_br, ren_tl, ren_br = bb 
            K_ren = torch.tensor( self.trainer.val_dataloaders[0].dataset._backend.get_camera('data_syn/0019', K=True), device=self.device )

            b = 0
            K_real = torch.tensor( [[cam[b,2],0,cam[b,0]],[b,cam[b,3],cam[b,1]],[0,0,1]], device=self.device )
            
            h_real_est = torch.eye(4,device=self.device)
            h_real_est[:3,:3] = quat_to_rot(pred_rot_wxyz[b][None,:], conv='wxyz', device=self.device)
            h_real_est[:3,3] = torch.tensor( pred_trans[b] ,device=self.device )
            
            typ = u_map.dtype
        
            
            anal_tensor( real_br, 'real_br', print_on_error = True)
            anal_tensor( real_tl, 'real_tl', print_on_error = True)
            P_real_in_center, P_ren_in_center, P_real_trafo, T_res = flow_to_trafo(real_br[b], 
                copy.deepcopy(real_tl[b]), 
                copy.deepcopy(ren_br[b]), 
                copy.deepcopy(ren_tl[b]), 
                copy.deepcopy(flow_mask[b]), 
                copy.deepcopy(u_map[b].type( typ )), 
                copy.deepcopy(v_map[b].type( typ )), 
                copy.deepcopy(K_real.type( typ )), 
                copy.deepcopy(K_ren.type( typ )), 
                copy.deepcopy(real_d[b][0].type( typ )), 
                copy.deepcopy(render_d[b][0].type( typ )), 
                copy.deepcopy(h_real_est.type( typ )), 
                copy.deepcopy(h_render[b].type( typ )))

            

            if anal_tensor( T_res, 'T_res GT Flow', print_on_error = True):
                raise Exception('T_res GT Flow contains inf or nan')

            
            h_real_new_est =  T_res @ h_render[0].type(typ) # set rotation
            typ = flow[b, 0, :, :].dtype
             
            P_real_in_center, P_ren_in_center, P_real_trafo, T_res = flow_to_trafo(real_br[b], 
                real_tl[b],
                ren_br[b], 
                ren_tl[b], 
                flow_mask[b], 
                flow[b, 0, :, :].type( typ ), 
                flow[b, 1, :, :].type( typ ), 
                K_real.type( typ ), 
                K_ren.type( typ ), 
                real_d[b][0].type( typ ), 
                render_d[b][0].type( typ ), 
                h_real_est.type( typ ), 
                h_render[b].type( typ ))
            P_real_in_center, P_ren_in_center, P_real_trafo, T_res = flow_to_trafo(real_br[b], 
                copy.deepcopy(real_tl[b]), 
                copy.deepcopy(ren_br[b]), 
                copy.deepcopy(ren_tl[b]), 
                copy.deepcopy(flow_mask[b]), 
                torch.zeros( flow[b, 0, :, :].shape, device= self.device), 
                torch.zeros( flow[b, 1, :, :].shape, device= self.device), 
                copy.deepcopy(K_real.type( typ )), 
                copy.deepcopy(K_ren.type( typ )), 
                copy.deepcopy(real_d[b][0].type( typ )), 
                copy.deepcopy(render_d[b][0].type( typ )), 
                copy.deepcopy(h_real_est.type( typ )), 
                copy.deepcopy(h_render[b].type( typ )))

            h_real_new_est_pred_flow =  T_res @ h_render[0] # set rotation
        
        if self.visu_forward:
            self._k += 1
            self.counter_images_logged += 1
            mask = (flow_mask == True)

            
            self.visualizer.plot_translations(
                tag = f'gt_votes_{self._mode}_nr_{self.counter_images_logged}',
                epoch = self.current_epoch,
                img = real_img[0].permute(1, 2, 0).cpu(),
                flow = uv_gt.permute(0,2,3,1)[0].cpu(),
                mask=mask[0].cpu(),
                store=True,
                method= 'left')
            self.visualizer.plot_translations(
                tag = f'predicted_votes_{self._mode}_nr_{self.counter_images_logged}',
                epoch = self.current_epoch,
                img = real_img[0].permute(1, 2, 0).cpu(),
                flow = flow[0, :2, :, :].permute(1, 2, 0).cpu(),
                mask=mask[0].cpu(),
                store=True,
                method= 'right')

                
            seg_max = p_label.argmax(dim=1)
            self.visualizer.plot_segmentation(tag=f'_',
                                                epoch=self.current_epoch,
                                                label=gt_label_cropped[0].cpu(
                                                ).numpy(),
                                                store=True,
                                                method='left')
            
            self.visualizer.plot_segmentation(tag=f'Segmentation_(left gt , right predicted)_{self._mode}_nr_{self.counter_images_logged}',
                                                epoch=self.current_epoch,
                                                label=seg_max[0].cpu(
                                                ).numpy(),
                                                store=True,
                                                method='right')
    
            self.visualizer.plot_corrospondence(tag=f'_',
                                                epoch=self.current_epoch,
                                                u_map=u_map[0], 
                                                v_map=v_map[0], 
                                                flow_mask=flow_mask[0], 
                                                real_img=real_img[0], 
                                                render_img=render_img[0],
                                                store=True,
                                                method='left')
            self.visualizer.plot_corrospondence(tag=f'Flow_(left gt , right predicted)_{self._mode}_nr_{self.counter_images_logged}',
                                                epoch=self.current_epoch,
                                                u_map= flow[0,0,:,:], 
                                                v_map= flow[0,1,:,:], 
                                                flow_mask=flow_mask[0], 
                                                real_img=real_img[0], 
                                                render_img=render_img[0],
                                                store=True,
                                                method='right')
            
            self.visualizer.plot_estimated_pose(    tag = f"_",
                                        epoch = self.current_epoch,
                                        img= real_img_original[b].cpu().numpy(),
                                        points = copy.deepcopy(model_points[b].cpu().numpy()),
                                        store = True,
                                        K = K_real.cpu().numpy(),
                                        H = h_real_new_est.cpu().numpy(),
                                        method='left')
            self.visualizer.plot_estimated_pose(    tag = f"Pose_estimate_(left gt_flow_trafo, right pred_flow_trafo)_{self._mode}_nr_{self.counter_images_logged}",
                                        epoch = self.current_epoch,
                                        img= real_img_original[b].cpu().numpy(),
                                        points = copy.deepcopy(model_points[b].cpu().numpy()),
                                        store = True,
                                        K = K_real.cpu().numpy(),
                                        H = h_real_new_est_pred_flow.detach().cpu().numpy(),
                                        method='right')

 
            self.visualizer.plot_estimated_pose(    tag = f"_",
                                        epoch = self.current_epoch,
                                        img= real_img_original[b].cpu().numpy(),
                                        points =copy.deepcopy(model_points[b].cpu().numpy()),
                                        store = True,
                                        K = K_real.cpu().numpy(),
                                        H = h_real[b].cpu().numpy(), 
                                      method='left' )
            self.visualizer.plot_estimated_pose( tag = f"Pose_estimate_(left gt_pose, right pred_flow_trafo)_{self._mode}_nr_{self.counter_images_logged}",
                                        epoch = self.current_epoch,
                                        img= real_img_original[b].cpu().numpy(),
                                        points =copy.deepcopy(model_points[b].cpu().numpy()),
                                        store = True,
                                        K = K_real.cpu().numpy(),
                                        H = h_real_new_est_pred_flow.detach().cpu().numpy(),
                                        method='right')
 
 
 
            self.visualizer.plot_estimated_pose(    tag = f"_",
                                        epoch = self.current_epoch,
                                        img= real_img_original[b].cpu().numpy(),
                                        points =copy.deepcopy(model_points[b].cpu().numpy()),
                                        store = True,
                                        K = K_real.cpu().numpy(),
                                        H = h_real_est.cpu().numpy(), 
                                        method='left' )
            self.visualizer.plot_estimated_pose(    tag = f"Pose_estimate_(left input_pose, right pred_flow_trafo)_{self._mode}_nr_{self.counter_images_logged}",
                                        epoch = self.current_epoch,
                                        img= real_img_original[b].cpu().numpy(),
                                        points =copy.deepcopy(model_points[b].cpu().numpy()),
                                        store = True,
                                        K = K_real.cpu().numpy(),
                                        H = h_real_new_est_pred_flow.detach().cpu().numpy(),
                                        method='right')

            p = model_points.shape[1]

            target = torch.bmm( model_points, torch.transpose(h_real[:,:3,:3], 1,2 ) ) + h_real[:,:3,3][:,None,:].repeat(1,p,1)
            # Compute ADD-S
            adds_res_gt_flow = self.criterion_adds(target[b][None], model_points[b][None], idx[b][None], H = h_real_new_est[None].type( target.dtype) )
            adds_res_pred_flow = self.criterion_adds(target[b][None], model_points[b][None], idx[b][None], H = h_real_new_est_pred_flow[None].type( target.dtype))
            adds_init = self.criterion_adds(target[b][None], model_points[b][None], idx[b][None], H = h_real_est[None].type( target.dtype))

            # adds_gt = self.criterion_adds(target[b][None], model_points[b][None], idx[b][None], H = h_real[0][None])
            # log scalars            
            log_scalars[f'adds_init'] = float(adds_init.detach())
            log_scalars[f'adds_res_gt_flow'] = float(adds_res_gt_flow.detach())
            log_scalars[f'adds_res_pred_flow'] = float(adds_res_pred_flow.detach())
            
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
            self._dict_track['val_disparity  [+inf - 0]'].append(float(loss))
        except:
            self._dict_track['val_disparity  [+inf - 0]'] = [float(loss)]
        
        for i in range(0, bs):
            # object loss for each object
            obj = int(unique_desig[1][i])
            obj = list(
                self.trainer.val_dataloaders[0].dataset._backend._name_to_idx_full.keys())[obj - 1]
            if f'val_{obj}_avg_disparity_L2_dis  [+inf - 0]' in self._dict_track.keys():
                self._dict_track[f'val_{obj}_avg_disparity_L2_dis  [+inf - 0]'].append(
                    float(dis[i]))
            else:
                self._dict_track[f'val_{obj}_avg_disparity_L2_dis  [+inf - 0]'] = [
                    float(dis[i])]
        
        tensorboard_logs = {'val_disparity': float(loss)}
        tensorboard_logs = {**tensorboard_logs, **log_scalars}

        val_loss = loss
        val_dis = loss
        return {'val_loss': val_loss, 'val_dis': val_dis, 'log': tensorboard_logs}
   
    def visu_step(self, nr, batch, pred_r, pred_t, batch_idx):
        pass

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

        try:
            df1 = dict_to_df(avg_dict)
            df2 = dict_to_df(get_df_dict(pre='val'))
            img = compare_df(df1, df2, key='auc [0 - 100]')
            tag = 'val_table_res_vs_df'
            img.save(self.exp['model_path'] +
                    f'/visu/{self.current_epoch}_{tag}.png')
            self.logger.experiment.add_image(tag, np.array(img).astype(
                np.uint8), global_step=self.current_epoch, dataformats='HWC')
        except:
            pass
        avg_val_dis_float = float(avg_dict['avg_val_disparity  [+inf - 0]'])
        return {'avg_val_dis_float': avg_val_dis_float,
                'avg_val_dis': avg_dict['avg_val_disparity  [+inf - 0]'],
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
    # with torch.autograd.set_detect_anomaly(True):
    # early_stop_callback=early_stop_callback,
    trainer = Trainer(**exp['trainer'],
        checkpoint_callback=checkpoint_callback,
        default_root_dir=exp['model_path'])

    if exp.get('model_mode', 'fit') == 'fit':
        # lr_finder = trainer.lr_find(
        #     model, min_lr=0.0000001, max_lr=0.001, num_training=50, early_stop_threshold=100)
        # lr_finder.results
        # lr_finder.suggestion()
        # print('LR FInder suggestion', lr_finder.suggestion())
        # print(lr_finder.results)
        trainer.fit(model)
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
