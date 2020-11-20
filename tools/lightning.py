import warnings
warnings.simplefilter("ignore", UserWarning)

import copy
import sys
import os
import time
import shutil
import argparse
import logging
import signal
import pickle
import math

# misc
import numpy as np
import pandas as pd
import random
import sklearn
# from scipy.spatial.transform import Rotation as R
from math import pi
from math import ceil


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

from loss import FocalLoss, FlowLoss, AddSLoss

from loaders_v2 import GenericDataset
from visu import Visualizer
from helper import re_quat, flatten_dict
from helper import anal_tensor
from helper import compute_auc
from rotations import *
from model import EfficientDisparity
from deep_im import flow_to_trafo_PnP

def check_exp(exp):
    if exp['d_test'].get('overfitting_nr_idx', -1) != -1 or exp['d_train'].get('overfitting_nr_idx', -1) != -1:
        print('Overfitting on ONE batch is activated')
        print('Overfitting on ONE batch is activated')
        print('Overfitting on ONE batch is activated')
        print('Overfitting on ONE batch is activated')
        print('Overfitting on ONE batch is activated')
        time.sleep(5)
def filter_dict( dic, remove_key):
    new_dict = {}
    for k in list( dic.keys()):
        if k.find(remove_key) != -1 and k.find('auc') != -1 :
            nk = k 
            nk = nk.replace(remove_key+' ', '')
            nk = nk.replace(remove_key, '')
            
            new_dict[ nk ] = dic[k]
    return new_dict
class Logger2(object):
    def __init__(self, p):
        self.terminal = sys.stdout
        self.p = p

    def write(self, message):
        with open (self.p, "a") as self.log:            
            self.log.write(message)
        self.terminal.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass
 

class TrackNet6D(LightningModule):
    def __init__(self, exp, env):
        super().__init__()
        self._mode = 'test'

        # check exp for errors
        check_exp(exp)
        self._k = 0
        self.visu_forward = False

        self.track = exp['model_mode'] == 'test' and \
            exp['d_test']['batch_list_cfg']['seq_length'] != 1 and \
            exp['d_test']['batch_list_cfg']['mode'].find('dense') == -1

        # logging h-params
        exp_config_flatten = flatten_dict(copy.deepcopy(exp))
        for k in exp_config_flatten.keys():
            if exp_config_flatten[k] is None:
                exp_config_flatten[k] = 'is None'

        self.hparams = exp_config_flatten
        self.hparams['lr'] = exp['lr']
        self.env, self.exp = env, exp

        for i in range(0, int(torch.cuda.device_count())):
            print(f'GPU {i} Type {torch.cuda.get_device_name(i)}')

        self.pixelwise_refiner = EfficientDisparity( **exp['efficient_disp_cfg'] )

        self.criterion_adds = AddSLoss(sym_list=exp['d_train']['obj_list_sym'])
        coe = exp['loss'].get('coefficents',[0.0005,0.001,0.005,0.01,0.02,0.08,1])
        self.criterion_focal = FocalLoss() 
        s = self.pixelwise_refiner.size
        self.criterion_flow = FlowLoss(s, s, coe)
       
        self.best_validation = 999
        self.visualizer = None
        self._dict_track = {}
        self.counter_images_logged = 0
        self.test_size = 0.1
        self.init_train_vali_split = False

        if self.exp.get('visu', {}).get('log_to_file', False):
            mp = exp['model_path']
            sys.stdout = Logger2(f'{mp}/Live_Logger_Lightning.log')
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)
            logging.getLogger().addHandler(console)
            logging.getLogger("lightning").addHandler(console)
            sys.stderr = sys.stdout

        self.adds_mets = ['init','gt_flow__gt_label', 'pred_flow__gt_label','pred_flow__flow_mask','pred_flow__pred_label']
        lis = []
        for i in range(0, 6):
            lis.append( f'L2_{i}')
        self.df = pd.DataFrame(columns= self.adds_mets + lis )

        self.start = time.time()

    def reload_batch(self, frame, h):
        """
        Args:
            batch ([list]): dataloader one frame
            h ([np.array]): BSx4x4
        """
        for b, desig, obj in zip(range(len(frame[0][0])),frame[0][0], frame[0][1].cpu().tolist() ): 
            # get the new frame blocking the training process of the dataloader
            if self._mode == 'train': 
                batch_new = self.trainer.train_dataloader.dataset._backend.getElement(desig, obj, h[b])
            elif  self._mode == 'test' :
                batch_new = self.trainer.test_dataloaders[0].dataset._backend.getElement(desig, obj, h[b])
            elif self._mode == 'val':
                batch_new = self.trainer.val_dataloaders[0].dataset._backend.getElement(desig, obj, h[b])
            else:
                raise Exception('Unknown Mode during refinement')

            # TODO Check if loading batch failed 
            if True:
                for i in range(1, len(batch_new)):
                    if i == 11:
                        for j in range(0,4):
                            frame[i][j][b] = batch_new[i][j]
                    else:
                        frame[i][b] = batch_new[i]
            else:
                print('Failed to load flow with the new pose estimate')

    def forward(self, batch):
        suc = True
        st = time.time()
        
        # check if we are in tracking mode (only for testing implemented)
        if self.track and not self.first:
            # reload the frame
            batch = list(batch)
            self.reload_batch( batch, self.h_track.unsqueeze(0) )
            
            
        log_scalars = {}
        bs = batch[1].shape[0]
        # this loop is used for iterative refinement

        for iteration in range(self.exp.get('refine_cfg', {}).get('iterations',1 )):            
            if iteration > 0:      
                # Load data blocking when refineing ('useally only done in testing')                   
                batch = list(batch)
                self.reload_batch( batch, h_pred_flow__gt_label[None,:,:].cpu().numpy() )


            unique_desig, model_points,idx= batch[0:3] 
            real_img, render_img, real_d, render_d, gt_label_cropped, u_map, v_map,flow_mask = batch[3:11] 
            bb, h_render, h_init, h_gt, K_real= batch[11:16]                       

            data = torch.cat([real_img, render_img], dim=3) # BS,H,W,C
            data = data.permute(0,3,1,2) # BS,C,H,W

            if self.exp['efficient_disp_cfg'].get('ced_real_d',0) > 0 :
                data_with_depth = data = torch.cat([data, real_d[:,None,:,:], render_d[:,None,:,:]], dim=1) 
                flow, p_label = self.pixelwise_refiner(
                    data_with_depth, idx)
            else:
                flow, p_label = self.pixelwise_refiner(
                    data, idx)
            
            focal_loss = self.criterion_focal(
                p_label, gt_label_cropped)

            ind = (flow_mask == True )[:,None,:,:].repeat(1,2,1,1)
            uv_gt = torch.stack( [u_map, v_map], dim=3 ).permute(0,3,1,2)

            loss_l2_sum,flow_loss_l2_stack = self.criterion_flow(flow, ind, uv_gt)
            flow = flow[-1]
            
            if self.visu_forward or self.exp.get('visu', {}).get('always_calculate', False) or (self._mode == 'val' and self.exp.get('visu', {}).get('full_val', False) ) or self._mode == 'test': 
                real_tl, real_br, ren_tl, ren_br = bb

                b = 0
                
                # Calculate gt_flow__gt_label 
                gt_label_obj = (gt_label_cropped ==  unique_desig[1][:,None,None].repeat(1,480,640)   ) # BS,H,W
                flow_mask_in = flow_mask == True# BS,H,W
                # Calculate pred_flow__flow_mask
                typ =  torch.float32

                suc1,  h_gt_flow__gt_label = flow_to_trafo_PnP( 
                    real_br = real_br[b], 
                    real_tl = real_tl[b], 
                    ren_br = ren_br[b], 
                    ren_tl = ren_tl[b], 
                    flow_mask = gt_label_obj[b], 
                    u_map = u_map[b].type( typ ), 
                    v_map = v_map[b].type( typ ), 
                    K_ren = self.K_ren.type( typ ), 
                    K_real = K_real[b].type( typ ), 
                    render_d = render_d[b].type( typ ), 
                    h_render = h_render[b].type( typ ),
                    h_real_est = h_init[b].type( typ ))
            
                
                suc2,  h_pred_flow__flow_mask = flow_to_trafo_PnP( 
                    real_br = real_br[b], 
                    real_tl = real_tl[b], 
                    ren_br = ren_br[b], 
                    ren_tl = ren_tl[b], 
                    flow_mask = flow_mask_in[b], 
                    u_map = flow[b, 0, :, :].type( typ ), 
                    v_map = flow[b, 1, :, :].type( typ ), 
                    K_ren = self.K_ren.type( typ ), 
                    K_real = K_real[b].type( typ ), 
                    render_d = render_d[b].type( typ ), 
                    h_render = h_render[b].type( typ ),
                    h_real_est = h_init[b].type( typ ))

                # Calculate pred_flow__pred_label
                pred_seg_mask = ( p_label.argmax(dim=1) ==  unique_desig[1][:,None,None].repeat(1,480,640)   ) # BS,H,W
                suc3, h_pred_flow__pred_label = flow_to_trafo_PnP( 
                    real_br = real_br[b], 
                    real_tl = real_tl[b], 
                    ren_br = ren_br[b], 
                    ren_tl = ren_tl[b], 
                    flow_mask = pred_seg_mask[b], 
                    u_map = flow[b, 0, :, :].type( typ ), 
                    v_map = flow[b, 1, :, :].type( typ ), 
                    K_ren = self.K_ren.type( typ ), 
                    K_real = K_real[b].type( typ ), 
                    render_d = render_d[b].type( typ ), 
                    h_render = h_render[b].type( typ ),
                    h_real_est = h_init[b].type( typ ))
                
                suc4,  h_pred_flow__gt_label = flow_to_trafo_PnP( 
                    real_br = real_br[b], 
                    real_tl = real_tl[b], 
                    ren_br = ren_br[b], 
                    ren_tl = ren_tl[b], 
                    flow_mask = gt_label_obj[b], 
                    u_map = flow[b, 0, :, :].type( typ ), 
                    v_map = flow[b, 1, :, :].type( typ ), 
                    K_ren = self.K_ren.type( typ ), 
                    K_real = K_real[b].type( typ ), 
                    render_d = render_d[b].type( typ ), 
                    h_render = h_render[b].type( typ ),
                    h_real_est = h_init[b].type( typ ))

                suc = suc1 and suc2 and suc3 and suc4
            
            calc = False
            if self.exp.get('visu', {}).get('always_calculate', False) or (self._mode == 'val' and self.exp.get('visu', {}).get('full_val', False) ) or self._mode == 'test': 
                target = torch.bmm( model_points, torch.transpose(h_gt[:,:3,:3], 1,2 ) ) + h_gt[:,:3,3][:,None,:].repeat(1,model_points.shape[1],1)

                # Compute ADD-S
                adds_h_gt_flow__gt_label = self.criterion_adds(target[b][None], model_points[b][None], idx[b][None], H = h_gt_flow__gt_label [None].type( target.dtype) )
                adds_h_pred_flow__flow_mask  = self.criterion_adds(target[b][None], model_points[b][None], idx[b][None], H = h_pred_flow__flow_mask[None].type( target.dtype))
                adds_h_pred_flow__pred_label = self.criterion_adds(target[b][None], model_points[b][None], idx[b][None], H = h_pred_flow__pred_label[None].type( target.dtype))
                adds_h_pred_flow__gt_label = self.criterion_adds(target[b][None], model_points[b][None], idx[b][None], H = h_pred_flow__gt_label[None].type( target.dtype))
                adds_init = self.criterion_adds(target[b][None], model_points[b][None], idx[b][None], H = h_init[None].type( target.dtype))
                
                # log scalars            
                log_scalars[f'adds_init'] = float(adds_init.detach())
                log_scalars[f'adds_gt_flow__gt_label'] = float(adds_h_gt_flow__gt_label.detach())
                log_scalars[f'adds_pred_flow__gt_label'] = float(adds_h_pred_flow__gt_label.detach())
                log_scalars[f'adds_pred_flow__flow_mask'] = float(adds_h_pred_flow__flow_mask.detach())
                log_scalars[f'adds_pred_flow__pred_label'] = float(adds_h_pred_flow__pred_label.detach())
                v1 = round(log_scalars['adds_init'],4)
                v2 = round(log_scalars[f'adds_gt_flow__gt_label'],4)
                v3 = round(log_scalars[f'adds_pred_flow__gt_label'],4)
                v4 = round(float(torch.mean(flow_loss_l2_stack[-1], dim=0).detach()),2 )
                print(f'LOS after iteration {iteration} | Init: {v1}, gt: {v2}, pred: {v3}, L_flow_l2: {v4}')
                if v3 > self.exp['visu'].get('on_pred_fail',9999):
                    calc = True  
            if (self.visu_forward or calc) and not len(batch) > 16:
                print('Will try to visu but the dataloader is in the wrong mode !!!')

            if self.visu_forward or calc:
                if self.visualizer is None:
                    self.visualizer = Visualizer(self.exp['model_path'] +
                                                    '/visu/', self.logger.experiment)
                b = 0
                # extract visu data 
                render_img_original =  batch[16]  
                depth_render_original = batch[17]                
                real_img_original = batch[18]
                depth_original = batch[19]
                label_original = batch[20] 

                self._k += 1
                self.counter_images_logged += 1
                mask = (flow_mask == True)
                self.visualizer.plot_estimated_pose(    tag = f"_",
                                            epoch = self.current_epoch,
                                            img= real_img_original[b].cpu().numpy(),
                                            points = copy.deepcopy(model_points[b].cpu().numpy()),
                                            store = True,
                                            K = K_real[b].cpu().numpy(),
                                            H = h_gt[b,:,:].cpu().numpy(),
                                            method='left')
                self.visualizer.plot_estimated_pose(    tag = f"Pose_estimate_(GT POSE, right pred_flow__gt_flow)_{self._mode}_nr_{self.counter_images_logged}",
                                            epoch = self.current_epoch,
                                            img= real_img_original[b].cpu().numpy(),
                                            points = copy.deepcopy(model_points[b].cpu().numpy()),
                                            store = True,
                                            K = K_real[b].cpu().numpy(),
                                            H = h_pred_flow__gt_label.detach().cpu().numpy(),
                                            method='right')
                self.visualizer.plot_estimated_pose(    tag = f"_",
                                            epoch = self.current_epoch,
                                            img= real_img_original[b].cpu().numpy(),
                                            points =copy.deepcopy(model_points[b].cpu().numpy()),
                                            store = True,
                                            K = K_real[b].cpu().numpy(),
                                            H = h_gt_flow__gt_label.cpu().numpy(), 
                                        method='left' )
                self.visualizer.plot_estimated_pose( tag = f"Pose_estimate_(left gt_flow__gt_label, right h_pred_flow__pred_label)_{self._mode}_nr_{self.counter_images_logged}",
                                            epoch = self.current_epoch,
                                            img= real_img_original[b].cpu().numpy(),
                                            points =copy.deepcopy(model_points[b].cpu().numpy()),
                                            store = True,
                                            K = K_real[b].cpu().numpy(),
                                            H = h_pred_flow__pred_label.detach().cpu().numpy(),
                                            method='right')

                self.visualizer.flow_to_gradient(tag = f'gt_flow_{self._mode}_nr_{self.counter_images_logged}', epoch= self.current_epoch,
                    img = real_img[0].cpu().type(torch.float32), flow =  uv_gt[0, :2, :, :].permute(1, 2, 0).cpu().type(torch.float32), 
                    mask = (gt_label_cropped[0] == idx[0]+1).cpu().type(torch.float32), #,tl=real_tl[0], br=real_br[0],
                    store=True, jupyter=False, method='left')
                self.visualizer.flow_to_gradient(tag = f'Flow_Gradient_Crop_left_gt__right_pred_{self._mode}_nr_{self.counter_images_logged}', epoch= self.current_epoch,
                    img = real_img[0].cpu().type(torch.float32), flow = flow[0, :2, :, :].permute(1, 2, 0).cpu().type(torch.float32), 
                    mask = (gt_label_cropped[0] == idx[0]+1).cpu().type(torch.float32), #,tl=real_tl[0], br=real_br[0],
                    store=True, jupyter=False, method='right')

                self.visualizer.flow_to_gradient(tag = f'gt_flow_{self._mode}_nr_{self.counter_images_logged}', epoch= self.current_epoch,
                    img = real_img_original[0].cpu().type(torch.float32), flow =  uv_gt[0, :2, :, :].permute(1, 2, 0).cpu().type(torch.float32), 
                    mask = (label_original[0] == idx[0]+1).cpu().type(torch.float32),tl=real_tl[0], br=real_br[0],
                    store=True, jupyter=False, method='left')
                self.visualizer.flow_to_gradient(tag = f'Flow_Gradient_left_gt__right_pred_{self._mode}_nr_{self.counter_images_logged}', epoch= self.current_epoch,
                    img = real_img_original[0].cpu().type(torch.float32), flow = flow[0, :2, :, :].permute(1, 2, 0).cpu().type(torch.float32), 
                    mask = (label_original[0] == idx[0]+1).cpu().type(torch.float32),tl=real_tl[0], br=real_br[0],
                    store=True, jupyter=False, method='right')

                if not self.exp['visu'].get('visu_fast', False):
                    self.visualizer.plot_translations(
                        tag = f'gt_votes_{self._mode}_nr_{self.counter_images_logged}',
                        epoch = self.current_epoch,
                        img = real_img[0].cpu(),
                        flow = uv_gt.permute(0,2,3,1)[0].cpu(),
                        mask=mask[0].cpu(),
                        store=True,
                        method= 'left')
                    self.visualizer.plot_translations(
                        tag = f'Predicted_votes_{self._mode}_nr_{self.counter_images_logged}',
                        epoch = self.current_epoch,
                        img = real_img[0].cpu(),
                        flow = flow[0, :2, :, :].permute(1, 2, 0).cpu(),
                        mask=mask[0].cpu(),
                        store=True,
                        method= 'right')
                    # self.visualizer.plot_segmentation(tag=f'_',
                    #                                     epoch=self.current_epoch,
                    #                                     label=flow_mask_ero[b].type(torch.bool).cpu(
                    #                                     ).numpy(),
                    #                                     store=True,
                    #                                     method='left')
                    
                    # self.visualizer.plot_segmentation(tag=f'Valid Flow_(gt flow eroded , right predicted label eroded)_{self._mode}_nr_{self.counter_images_logged}',
                    #                                     epoch=self.current_epoch,
                    #                                     label=valid_flow[b].type(torch.bool).cpu(
                    #                                     ).numpy(),
                    #                                     store=True,
                                                        # method='right')

                    seg_max = p_label.argmax(dim=1)
                    self.visualizer.plot_segmentation(tag=f'_',
                                                        epoch=self.current_epoch,
                                                        label=gt_label_cropped[b].cpu(
                                                        ).numpy(),
                                                        store=True,
                                                        method='left')
                    
                    self.visualizer.plot_segmentation(tag=f'Segmentation_(left gt , right predicted)_{self._mode}_nr_{self.counter_images_logged}',
                                                        epoch=self.current_epoch,
                                                        label=seg_max[b].cpu(
                                                        ).numpy(),
                                                        store=True,
                                                        method='right')
            
                    self.visualizer.plot_corrospondence(tag=f'_',
                                                        epoch=self.current_epoch,
                                                        u_map=u_map[b], 
                                                        v_map=v_map[b], 
                                                        flow_mask=flow_mask[b], 
                                                        real_img=real_img[b], 
                                                        render_img=render_img[b],
                                                        store=True,
                                                        method='left')
                    self.visualizer.plot_corrospondence(tag=f'Flow_(left gt , right predicted)_{self._mode}_nr_{self.counter_images_logged}',
                                                        epoch=self.current_epoch,
                                                        u_map= flow[b,0,:,:], 
                                                        v_map= flow[b,1,:,:], 
                                                        flow_mask=flow_mask[b], 
                                                        real_img=real_img[b], 
                                                        render_img=render_img[b],
                                                        store=True,
                                                        method='right')
                    self.visualizer.plot_estimated_pose(    tag = f"_",
                                                epoch = self.current_epoch,
                                                img= real_img_original[b].cpu().numpy(),
                                                points =copy.deepcopy(model_points[b].cpu().numpy()),
                                                store = True,
                                                K = K_real[b].cpu().numpy(),
                                                H = h_init[b].cpu().numpy(), 
                                                method='left' )
                    self.visualizer.plot_estimated_pose(    tag = f"Pose_estimate_(left Input Pose, right h_pred_flow__pred_label)_{self._mode}_nr_{self.counter_images_logged}",
                                                epoch = self.current_epoch,
                                                img= real_img_original[b].cpu().numpy(),
                                                points =copy.deepcopy(model_points[b].cpu().numpy()),
                                                store = True,
                                                K = K_real[b].cpu().numpy(),
                                                H = h_pred_flow__pred_label.detach().cpu().numpy(),
                                                method='right')
                    
        if self._mode == 'test':
            if not suc1:  log_scalars[f'adds_gt_flow__gt_label']= 999.0
            if not suc2:  log_scalars[f'adds_pred_flow__flow_mask']= 999.0
            if not suc3:  log_scalars[f'adds_pred_flow__pred_label']= 999.0
            if not suc4:  log_scalars[f'adds_pred_flow__gt_label']= 999.0

            try:
                col = ['ID'] + self.adds_mets 
                test_values = [log_scalars.get("adds_"+key) for key in self.adds_mets]
                test_values = [int( unique_desig[1])] + test_values
                res = {col[i]: test_values[i] for i in range(len(col))} 
                for i in range(0, len( flow_loss_l2_stack)):
                    res[f'L2_{i}'] = float(flow_loss_l2_stack[i][0].detach())

                self.df = self.df.append(res, ignore_index=True)
            except:
                print("Failed adding obj during testing")

        w_s = self.exp.get('loss', {}).get('weight_semantic_segmentation', 0.5)
        w_f = self.exp.get('loss', {}).get('weight_flow', 0.5)
 
        loss = w_s * focal_loss + w_f * loss_l2_sum
        
        log_scalars[f'loss_segmentation'] = float(
            torch.mean(focal_loss, dim=0).detach())
        log_scalars[f'loss_flow'] = float(torch.mean(flow_loss_l2_stack[-1], dim=0).detach())
        

        fl = log_scalars[f'loss_flow']
        if torch.any( torch.isnan(loss)):
            print(flow_loss_l2_stack)
            loss = torch.zeros(
                (bs,1), requires_grad=True, dtype=torch.float32, device=self.device)
            print('got NAN in loss')
            return loss, log_scalars, False
        
        if self.track:
            self.h_track =  h_pred_flow__gt_label.cpu().numpy()

        return loss, log_scalars, suc
    
    def on_epoch_start(self):
        self.counter_images_logged = 0
        self._mode = 'train'
        self.trainer.train_dataloader.dataset.visu = True

        if self.exp.get( 'training_params_limit', False):
            for j, block in enumerate(self.pixelwise_refiner.feature_extractor._blocks):
                for b in block.parameters():
                    if j < 15:
                        b.requires_grad = False
    
    def training_step(self, batch, batch_idx):
        st = time.time()
        unique_desig = batch[0][0]
        total_loss = 0
        total_dis = 0
        nr = self.exp.get('visu', {}).get('number_images_log_train', 1)
        if self.counter_images_logged < nr:
            self.visu_forward = True
        else:
            self.visu_forward = False
            self.trainer.train_dataloader.dataset.visu = False
        

        # forward
        dis, log_scalars, suc = self(batch[0])

        loss = torch.mean(dis)
        if not suc:
            return {'loss': loss}
        # # for epoch average logging
        try:
            self._dict_track['train_loss  [+inf - 0]'].append(float(loss))
        except:
            self._dict_track['train_loss  [+inf - 0]'] = [float(loss)]

        # tensorboard logging
        tensorboard_logs = {'train_loss': float(loss)}
        tensorboard_logs = {**tensorboard_logs, **log_scalars}
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': {'L_Seg': log_scalars['loss_segmentation'], 'L_Flow': log_scalars['loss_flow']}}

    def on_validation_epoch_start(self):
        self.counter_images_logged = 0
        self._mode = 'val'
        self.trainer.val_dataloaders[0].dataset.visu = True

    def validation_step(self, batch, batch_idx):
        st = time.time()
        self._mode = 'val'
        unique_desig = batch[0][0]
        total_loss = 0
        total_dis = 0

        nr = self.exp.get('visu', {}).get('number_images_log_val', 1)
        if self.counter_images_logged < nr:
            self.visu_forward = True
        else:
            self.visu_forward = False
            self.trainer.val_dataloaders[0].dataset.visu = False

        # forward
        dis, log_scalars, suc = self(batch[0])
        
        bs = dis.shape[0]
        
        # aggregate statistics per object (ADD-S sym and ADD non sym)
        loss = torch.mean(dis)
        if not suc:
            tensorboard_logs = {f'{self._mode}_disparity': float(loss)}

            return {f'{self._mode}_loss': loss, 'log': tensorboard_logs}
        try:
            self._dict_track[f'{self._mode}_disparity  [+inf - 0]'].append(float(loss))
        except:
            self._dict_track[f'{self._mode}_disparity  [+inf - 0]'] = [float(loss)]
        
        for i in range(0, bs):
            # object loss for each object
            obj = int(unique_desig[1][i])
            obj = self.obj_list[obj - 1]
            if f'{self._mode}_{obj}_avg_disparity_L2_dis  [+inf - 0]' in self._dict_track.keys():
                self._dict_track[f'{self._mode}_{obj}_avg_disparity_L2_dis  [+inf - 0]'].append(
                    float(dis[i]))
            else:
                self._dict_track[f'{self._mode}_{obj}_avg_disparity_L2_dis  [+inf - 0]'] = [
                    float(dis[i])]


        for n in self.adds_mets:
            try:
                obj = int(unique_desig[1][0])
                obj = self.obj_list[obj - 1]

                na = f'{self._mode}_{n}adds_dis (only for first obj in batch) [+inf - 0]' 
                na_obj =  f'{self._mode}_{obj}_{n}adds_dis (only for first obj in batch) [+inf - 0]' 

                value = log_scalars[f'adds_{n}'] 

                if na in self._dict_track.keys():
                    self._dict_track[na].append( value )
                else:
                    self._dict_track[na] = [value]

                if na_obj in self._dict_track.keys():
                    self._dict_track[na_obj].append( value )
                else:
                    self._dict_track[na_obj] = [value]
            except:
                pass            

        tensorboard_logs = {f'{self._mode}_disparity': float(loss)}
        tensorboard_logs = {**tensorboard_logs, **log_scalars}

        return {f'{self._mode}_loss': loss, 'log': tensorboard_logs}
   
    def on_test_epoch_start(self):
        self.counter_images_logged = 0
        self._mode = 'test'
        self._number_lost_track = 0
        self.trainer.test_dataloaders[0].dataset.visu = True

    def test_step(self, batch, batch_idx):
        st = time.time()
        
        total_loss = 0
        total_dis = 0

        nr = self.exp.get('visu', {}).get(f'number_images_log_{self._mode}', 1)
        if self.counter_images_logged < nr:
            self.visu_forward = True
        else:
            self.visu_forward = False
            self.trainer.test_dataloaders[0].dataset.visu = False


        # forward
        if not self.track: 
            unique_desig = batch[0][0]
            dis, log_scalars, suc = self(batch[0])
            loss = torch.mean(dis)        
            if not suc:
                return {f'{self._mode}_loss': loss}
            # aggregate statistics per object (ADD-S sym and ADD non sym)
            self.log_scalars_to_track_dict(log_scalars, dis, unique_desig)
        else:
            loss_ls = []
            # the first flag is used in the forward pass if it is true the start picture is loaded
            # if false the last frame prediction that set as the feedback is used
            self.first = True
            for j, frame in enumerate(batch):
                unique_desig = frame[0]
                dis, log_scalars, suc = self(frame)
                loss = torch.mean(dis)
                if not suc:
                    return {f'{self._mode}_loss': loss}

                self.log_scalars_to_track_dict(log_scalars, dis, unique_desig)
                
                loss_ls.append(loss)
                if self.first and j != 0 and log_scalars[f'adds_gt_flow__gt_label'] != 999:
                    print(f'{j} GOT TRACK AGAIN')
                    self.first = False
                if j == 0:
                    self.first = False
                
                print(f'{j} Forward, ADDs gt_flow_gt_label:', round(log_scalars[f'adds_gt_flow__gt_label'],4) , 'pred_flow_gt_label: ', round(log_scalars[f'adds_pred_flow__gt_label'],4) ) 

                if log_scalars[f'adds_gt_flow__gt_label'] == 999:
                    print(f'{j} LOST TRACK RESET first flag')
                    self._number_lost_track += 1
                    self.first = True
            loss = torch.mean( torch.tensor(loss_ls, device=loss.device))
        tensorboard_logs = {f'{self._mode}_disparity': float(loss)}
        tensorboard_logs = {**tensorboard_logs, **log_scalars}
        pb = {'L_Seg': log_scalars['loss_segmentation'], 'L_Flow': log_scalars['loss_flow'], 'ADD_S': log_scalars['adds_pred_flow__gt_label'] }
        
        return {'test_loss': loss, 'log': tensorboard_logs, 'progress_bar': pb }
   
    def log_scalars_to_track_dict(self, log_scalars, dis, unique_desig):
        loss = torch.mean(dis)
        bs = dis.shape[0]
        try:
            self._dict_track[f'{self._mode}_disparity  [+inf - 0]'].append(float(loss))
        except:
            self._dict_track[f'{self._mode}_disparity  [+inf - 0]'] = [float(loss)]
        
        for i in range(0, bs):
            # object loss for each object
            obj = int(unique_desig[1][i])
            obj = self.obj_list[obj - 1]
            if f'{self._mode}_{obj}_avg_disparity_L2_dis  [+inf - 0]' in self._dict_track.keys():
                self._dict_track[f'{self._mode}_{obj}_avg_disparity_L2_dis  [+inf - 0]'].append(
                    float(dis[i]))
            else:
                self._dict_track[f'{self._mode}_{obj}_avg_disparity_L2_dis  [+inf - 0]'] = [
                    float(dis[i])]


        for n in self.adds_mets:
            try:
                obj = int(unique_desig[1][0])
                obj = self.obj_list[obj - 1]

                na = f'{self._mode}_{n}_adds_dis [+inf - 0]' 
                na_obj =  f'{self._mode}_{obj}_{n}_adds_dis [+inf - 0]' 

                value = log_scalars[f'adds_{n}'] 

                if na in self._dict_track.keys():
                    self._dict_track[na].append( value )
                else:
                    self._dict_track[na] = [value]

                if na_obj in self._dict_track.keys():
                    self._dict_track[na_obj].append( value )
                else:
                    self._dict_track[na_obj] = [value]
            except:
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
        if not 'avg_val_disparity  [+inf - 0]' in avg_dict.keys():
            avg_dict['avg_val_disparity  [+inf - 0]'] = 999
        avg_val_disparity_float = float(avg_dict['avg_val_disparity  [+inf - 0]'])

        return {'avg_val_disparity_float': avg_val_disparity_float,
                'avg_val_disparity': torch.tensor( avg_dict['avg_val_disparity  [+inf - 0]'] ),
                'log': avg_dict}

    def training_epoch_end(self, outputs):
        self._k = 0
        self.counter_images_logged = 0  # reset image log counter
        avg_dict = {}
        for old_key in list(self._dict_track.keys()):
            if old_key.find('train') == -1:
                continue
            avg_dict['avg_' +
                     old_key] = float(np.mean(np.array(self._dict_track[old_key])))
            self._dict_track.pop(old_key, None)
        delta = time.time() - self.start
        v= self.exp['trainer']['limit_val_batches']
        t= self.exp['trainer']['limit_test_batches']
        bs = self.exp['loader']['batch_size']
        string = f'Time for one epoch: {int( delta/3600) }h, {int( (delta%3600)/60)}min  {int( delta%60) }s, Val: {v}, Test: {t}, BS: {bs}, Total:{(v+t)*bs}'
        logging.getLogger('lightning').info( string )

        self.start = time.time()
        avg_dict ['learning_rate'] = self.trainer.optimizers[0].param_groups[0]['lr']
        return {**avg_dict, 'log': avg_dict}

    def test_epoch_end(self, outputs):
        try:
            ("="*89)
            logging.info("="*89)
            logging.info("Mean: \n \n " + str(self.df['ID'].value_counts()))
            logging.info("="*89)
            logging.info("Count Working: \n \n " + str(self.df.groupby(['ID']).mean()))
            logging.info("="*89)
        except:
            pass
        try:
            os.mkdir( self.exp['model_path']+'/df')
        except:
            print('FAILED mkdir')
        self.df.to_pickle(self.exp['model_path']+'/df/test_df.pkl' )
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
                avg_dict[old_key[:p-1] + 'auc [0 - 100]'] = auc

            self._dict_track.pop(old_key, None)

       
        
        avg_test_dis_float = float(avg_dict['avg_test_gt_flow__gt_label_adds_dis [+inf - 0]'])
        
        for n in self.adds_mets:
            try:
                avg_in = filter_dict( avg_dict, n )
                df1 = dict_to_df(avg_in)
                # test_002_master_chef_can_res_gt_flow_auc [0 - 100]

                df2 = dict_to_df(get_df_dict(pre='test'))
                img = compare_df(df1, df2, key='auc [0 - 100]')
                tag = f'test_table_{n}_vs_df'
                img.save(self.exp['model_path'] +
                        f'/visu/{self.current_epoch}_{tag}.png')
                self.logger.experiment.add_image(tag, np.array(img).astype(
                    np.uint8), global_step=self.current_epoch, dataformats='HWC')
            except:
                pass
        logging.warning('Lost track during the full test epoch' + str(self._number_lost_track)+ ' times!' )
        return {f'avg_test_dis_float': avg_test_dis_float,
                f'avg_test_dis': avg_dict['avg_test_disparity  [+inf - 0]'],
                'log': avg_dict}

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

    def test_dataloader(self):
        dataset_test = GenericDataset(
            cfg_d=self.exp['d_test'],
            cfg_env=self.env)
        store = self.env['p_ycb'] + '/viewpoints_renderings'


        dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                      batch_size = 1,
                                                      num_workers = self.exp['loader']['num_workers'],
                                                      pin_memory= self.exp['loader']['pin_memory'],
                                                      shuffle= True)
        self.K_ren = torch.tensor( dataloader_test.dataset._backend.get_camera('data_syn/0019', K=True), device=self.device )
        self.obj_list = list( dataloader_test.dataset._backend._name_to_idx_full.keys()) 
        return dataloader_test

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
        self.K_ren = torch.tensor( dataloader_val.dataset._backend.get_camera('data_syn/0019', K=True), device=self.device )
        self.obj_list = list( dataloader_val.dataset._backend._name_to_idx_full.keys()) 

        return dataloader_val


