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
    test = torch.nonzero(img[:, :, :])
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
            input_channels=6, num_classes=21, growth_rate=16)

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
        points, choose, img, target, model_points, idx = batch[0:6]
        depth_img, label, img_orig, cam = batch[6:10]
        gt_rot_wxyz, gt_trans, unique_desig = batch[10:13]
        log_scalars = {}
        bs = points.shape[0]

        if len(batch) > 13:
            # check if skip
            if batch[13] is False:
                loss = torch.zeros(
                    bs.shape, requires_grad=True, dtype=torch.float32, device=self.device)
                pred_rot_wxyz = torch.zeros(
                    gt_rot_wxyz.shape, device=self.device)
                pred_rot_wxyz[:, 0] = 1
                pred_trans = torch.zeros(gt_trans.shape, device=self.device)
                return loss, pred_rot_wxyz.detach(), pred_trans.detach(), log_scalars

            real_img, render_img, real_d, render_d, gt_label_cropped, pred_rot_wxyz, pred_trans, pred_points = batch[
                13:]
            refine_iterations = 1
        else:
            pred_rot_wxyz, pred_trans, pred_points = forward_init_data(
                log_scalars)
            refine_iterations = get_ref_ite(self.exp)

        valid_samples = torch.ones((bs), device=self.device, dtype=torch.bool)

        for i in range(refine_iterations):
            if not(len(batch) > 13):
                real_img, render_img, real_d, render_d, gt_label_cropped = self.forward_prep_data(
                    idx, pred_rot_wxyz, pred_trans, pred_points, depth, cam, label)

            # stack the two images, might add additional mask as layer or depth info
            data = torch.cat([real_img, render_img], dim=1)

            # TODO idx is currently unused !!!!
            delta_v, rotations, p_label = self.pixelwise_refiner(
                data, idx)
            # delta_v = torch.zeros( delta_v.shape, device=self.device)

            pred_trans, pred_rot_wxyz, pred_points, delta_t = self.forward_pose_simple(delta_v, rotations, pred_trans,
                                                                                       pred_rot_wxyz, model_points, cam, idx, gt_label_cropped)

            if self.visu_forward and self.exp.get('visu', {}).get('network_input_batch', False):
                self._k += 1

                mask = gt_label_cropped[0] == (idx[0] + 1)

                print('delta_v_first 5', delta_v[0, :3, :, :][:, mask][:, :5])
                print('delta_v_last 5', delta_v[0, :3, :, :][:, mask][:, -5:])
                self.visualizer.plot_translations(
                    f'votes_image_plane_{self._mode}_nr_{self.counter_images_logged}',
                    self.current_epoch,
                    real_img[0].permute(1, 2, 0).cpu(),
                    delta_v[0, :2, :, :].permute(1, 2, 0).cpu(),
                    mask=mask.cpu(),
                    store=True)

                print('delta_t_first 5', delta_t[0, :3, :, :][:, mask][:, :5])
                print('delta_t_last 5', delta_t[0, :3, :, :][:, mask][:, -5:])
                self.visualizer.plot_translations(
                    f'votes_camera_coordinates_R3_{self._mode}_nr_{self.counter_images_logged}',
                    self.current_epoch,
                    real_img[0].permute(1, 2, 0).cpu(),
                    delta_t[0, :2, :, :].permute(1, 2, 0).cpu(),
                    mask=mask.cpu(),
                    store=True)

                seg_max = p_label.argmax(dim=1)
                self.visualizer.plot_segmentation(tag=f'gt_segmentation_cropped_{self._mode}_nr_{self.counter_images_logged}',
                                                  epoch=self.current_epoch,
                                                  label=gt_label_cropped[0].cpu(
                                                  ).numpy(),
                                                  store=True)
                # try:
                #     self.visualizer.plot_segmentation(tag=f'gt_segmentation_{self._mode}_nr_{self.counter_images_logged}',
                #                                       epoch=self.current_epoch,
                #                                       label=label[0].cpu(
                #                                       ).numpy(),
                #                                       store=True)
                # except:
                #     pass
                self.visualizer.plot_segmentation(tag=f'predicted_segmentation_{self._mode}_nr_{self.counter_images_logged}',
                                                  epoch=self.current_epoch,
                                                  label=seg_max[0].cpu(
                                                  ).numpy(),
                                                  store=True)
                # self.visualizer.visu_network_input(tag=f'network_input_{self._mode}_nr_{self.counter_images_logged}',
                #                                    epoch=self.current_epoch,
                #                                    data=data,
                #                                    max_images=10,
                #                                    store=True,
                #                                    jupyter=False)
                self.visualizer.visu_network_input_pred(tag=f'network_input_{self._mode}_nr_{self.counter_images_logged}',
                                                        epoch=self.current_epoch,
                                                        data=data,
                                                        images=img_orig,
                                                        target=pred_points,
                                                        cam=cam,
                                                        max_images=10,
                                                        store=True,
                                                        jupyter=False)
                # self.visualizer.plot_batch_projection(tag=f'batch_projection_{self._mode}_nr_{self.counter_images_logged}',
                #                                       epoch=self.current_epoch,
                #                                       images=img_orig,
                #                                       target=pred_points,
                #                                       cam=cam,
                #                                       max_images=10,
                #                                       store=True,
                #                                       jupyter=False)

        focal_loss = self.criterion_focal(
            p_label, gt_label_cropped)

        translation_loss = torch.norm(gt_trans - pred_trans, p=2, dim=1)

        # Compute ADD / ADD-S loss
        dis = self.criterion_adds(pred_r=pred_rot_wxyz, pred_t=pred_trans,
                                  target=target, model_points=model_points, idx=idx)
        # dis = dis  * valid_samples

        if exp.get('model', {}).get('df_load', False):
            log_scalars[f'df_ref_dis'] = float(
                torch.mean(df_ref_dis, dim=0).detach())
            if exp.get('model', {}).get('df_refine', False):
                log_scalars[f'df_ref_dis'] = float(
                    torch.mean(df_ref_dis, dim=0).detach())

        if torch.isnan(dis).any() or \
                torch.isnan(pred_rot_wxyz).any() or \
                torch.isnan(pred_trans).any():
            pass

        w_s = self.exp.get('loss', {}).get('weight_semantic_segmentation', 0.5)
        w_p = self.exp.get('loss', {}).get('weight_pose', 0.5)
        w_t = self.exp.get('loss', {}).get('weight_trans', 0.5)
        loss = w_s * focal_loss + w_p * dis + w_t * translation_loss

        log_scalars[f'loss_segmentation'] = float(
            torch.mean(focal_loss, dim=0).detach())
        log_scalars[f'loss_pose_add'] = float(torch.mean(dis, dim=0).detach())
        log_scalars[f'loss_translation'] = float(
            torch.mean(translation_loss, dim=0).detach())

        if torch.sum(valid_samples) == 0:
            loss = torch.zeros(
                dis.shape, requires_grad=True, dtype=torch.float32, device=self.device)

            pred_rot_wxyz = torch.zeros(
                pred_rot_wxyz.shape, device=self.device)
            pred_rot_wxyz[:, 0] = 1
            pred_trans = torch.zeros(pred_trans.shape, device=self.device)

        return loss, pred_rot_wxyz.detach(), pred_trans.detach(), log_scalars

    def forward_init_data(self, batch, log_scalars):
        for i in range(0, 10):
            if torch.isnan(batch[i]).any():
                raise Exception

        bs = points.shape[0]

        if self.exp.get('model', {}).get('df_load', False):
            # introduce new tensors for full tracking dense fusion
            st = time.time()

            num_points = exp['d_train']['num_points']

            tight_padded_img_batch = tight_image_batch(
                img, device=self.device)

            pred_r = torch.zeros((bs, 1000, 4), device=self.device)
            pred_t = torch.zeros((bs, 1000, 3), device=self.device)
            pred_c = torch.zeros((bs, 1000, 1), device=self.device)
            emb = torch.zeros((bs, 32, 1000), device=self.device)
            for i in range(bs):
                pred_r[i], pred_t[i], pred_c[i], emb[i], _ = self.df_pose_estimator(
                    ret_cropped_image(img[i])[None],
                    points[i][None],
                    choose[i][None],
                    idx[i][None])

            pred_r = pred_r / torch.norm(pred_r, dim=2)[:, :, None]
            w = self.exp.get('model', {}).get('df_w_normal', 0.015)
            refine_start = self.exp.get('model', {}).get(
                'df_refine', False)

            loss, df_ref_dis, new_points, new_target = self.df_criterion(
                pred_r, pred_t, pred_c,
                target, model_points, idx,
                points, w, refine_start, self.device)

            _, which_max = torch.max(pred_c, 1)
            which_max = which_max.squeeze(1)
            enum = torch.range(
                0, bs - 1, device=self.device, dtype=torch.long)
            pred_r_current = pred_r[enum, which_max, :]
            pred_t_current = pred_t[enum, which_max,
                                    :] + points[enum, which_max, :]
        else:
            pred_r_current = gt_rot_wxyz
            pred_t_current = gt_trans

            # Select the inital rotation and translation for iterative refinement
        mode = self.exp.get('model', {}).get(
            'inital_pose', {}).get('mode', {'TransNoise'})

        pred_rot_wxyz, pred_trans = get_inital(
            mode=mode,
            gt_rot_wxyz=gt_rot_wxyz,
            gt_trans=gt_trans,
            pred_r_current=pred_r_current,
            pred_t_current=pred_t_current,
            cfg=self.exp.get('model', {}).get(
                'inital_pose', {}), d=self.device)

        # apply pred_rot_wxyz, pred_trans (based on the mode) to get pred_points
        init_rot_mat = quat_to_rot(
            pred_rot_wxyz, 'wxyz', device=self.device).unsqueeze(1)
        init_rot_mat = init_rot_mat.view(-1, 3, 3).permute(0, 2, 1)
        pred_points = torch.add(
            torch.bmm(model_points, init_rot_mat), pred_trans.unsqueeze(1))
        store = pred_points.clone()
        w = 640
        h = 480
        bs = img.shape[0]

        return pred_rot_wxyz, pred_trans, pred_points

    def forward_prep_data(self, idx, pred_rot_wxyz, pred_trans, pred_points, depth, cam, label):
        w = 640
        h = 480
        bs = img.shape[0]

        # check if the current estimate of the objects position is within some bound
        # translation bounding:
        # deviation = torch.abs(torch.norm(pred_trans - gt_trans, dim=1))
        # for j in range(0, bs):
        #     if deviation[j] > self.exp.get('training', {}).get('trans_deviation_resample_inital_pose', 0.3):
        #         pred_trans[j] = torch.normal(
        #             mean=gt_trans[j], std=self.exp.get('training', {}).get('translation_noise_inital', 0.03))
        # deviation_post = torch.abs(
        #     torch.norm(pred_trans - gt_trans, dim=1))
        # if torch.sum(deviation) != torch.sum(deviation_post):
        #     # valid_samples[i] = False
        #     pass

        render_img = torch.zeros((bs, 3, h, w), device=self.device)
        render_d = torch.empty((bs, 1, h, w), device=self.device)
        # preper render data
        st = time.time()
        img_ren, depth, h_render = self.vm.get_closest_image_batch(
            i=idx, rot=pred_rot_wxyz, conv='wxyz')

        bb_lsd = get_bb_from_depth(depth)
        for j, b in enumerate(bb_lsd):
            tl, br = b.limit_bb()
            if br[0] - tl[0] < 30 or br[1] - tl[1] < 30 or b.violation():
                valid_samples[j] = False
                b.set_max()
                print("Depth BoundingBox violated the min size constrain", j)

            center_ren = backproject_points(
                h_render[j, :3, 3].view(1, 3), fx=cam[j, 2], fy=cam[j, 3], cx=cam[j, 0], cy=cam[j, 1])
            center_ren = center_ren.squeeze()
            b.move(-center_ren[1], -center_ren[0])
            b.expand(1.1)
            b.expand_to_correct_ratio(w, h)
            b.move(center_ren[1], center_ren[0])
            crop_ren = b.crop(img_ren[j]).unsqueeze(0)

            crop_ren = torch.transpose(crop_ren, 1, 3)
            crop_ren = torch.transpose(crop_ren, 2, 3)
            render_img[j] = self.up(crop_ren)
            crop_d = b.crop(depth[j, :, :, None].type(
                torch.float32))[None, :, :, :, ]
            crop_d = torch.transpose(crop_d, 1, 3)
            crop_d = torch.transpose(crop_d, 2, 3)
            render_d[j] = self.up(crop_d)

        # prepare real data
        real_img = torch.empty((bs, 3, h, w), device=self.device)
        real_d = torch.empty((bs, 1, h, w), device=self.device)

        if self.exp['model'].get('sem_seg', False):
            gt_label_cropped = torch.zeros(
                (bs, h, w), device=self.device, dtype=torch.long)
        # update the target to get new bb
        bb_ls = get_bb_real_target(pred_points, cam)
        for j, b in enumerate(bb_ls):

            tl, br = b.limit_bb()
            if br[0] - tl[0] < 30 or br[1] - tl[1] < 30 or b.violation():
                valid_samples[j] = False
                b.set_max()
                print("Real BoundingBox violated the min size constrain", j)

            center_real = backproject_points(
                pred_trans[j].view(1, 3), fx=cam[j, 2], fy=cam[j, 3], cx=cam[j, 0], cy=cam[j, 1])
            center_real = center_real.squeeze()
            b.move(-center_real[0], -center_real[1])
            b.expand(1.1)
            b.expand_to_correct_ratio(w, h)
            b.move(center_real[0], center_real[1])
            crop_real = b.crop(img_orig[j]).unsqueeze(0)
            up = torch.nn.UpsamplingBilinear2d(size=(h, w))
            crop_real = torch.transpose(crop_real, 1, 3)
            crop_real = torch.transpose(crop_real, 2, 3)
            real_img[j] = self.up(crop_real)

            crop_d = b.crop(depth_img[j, :, :, None].type(
                torch.float32))[None, :, :, :, ]
            crop_d = torch.transpose(crop_d, 1, 3)
            crop_d = torch.transpose(crop_d, 2, 3)
            real_d[j] = self.up(crop_d)

            if self.exp['model'].get('sem_seg', False):
                tmp = torch.transpose(torch.transpose(
                    b.crop(label[j].unsqueeze(2)), 0, 2), 1, 2)
                gt_label_cropped[j] = torch.round(up(tmp.type(
                    torch.float32).unsqueeze(0))).clamp(0, self.exp['d_train']['objects'] - 1).squeeze(2)

        return real_img, render_img, real_d, render_d, gt_label_cropped

    def forward_pose(self, delta_v, rotations, pred_trans, pred_rot_wxyz, model_points, cam, idx, gt_label_cropped):
        bs, _, h, w = delta_v.shape
        delta_v[:, :2, :, :] = delta_v[:, :2, :, :] * 100

        if torch.isinf(delta_v).any():
            base_new = quat_to_rot(
                pred_rot_wxyz.clone(), 'wxyz', device=self.device).unsqueeze(1)
            base_new = base_new.view(-1, 3, 3).permute(0, 2, 1)
            pred_points = torch.add(
                torch.bmm(model_points, base_new), pred_trans.unsqueeze(1))
            return pred_trans, pred_rot_wxyz, pred_points, torch.zeros(delta_v.shape, device=self.device)

        idx_v = torch.abs(delta_v[:, :2]) > 100
        while idx_v.any():
            idx_v = torch.abs(delta_v[:, :2]) > 100
            delta_v[:, :2][idx_v] = delta_v[:, :2][idx_v] * 0.5

        idx_z = torch.abs(delta_v[:, 2]) > 0.2
        while idx_z.any():
            idx_z = torch.abs(delta_v[:, 2]) > 0.2
            delta_v[:, 2][idx_z] = delta_v[:, 2][idx_z] * 0.5

        # TODO current bug useing ground truth semantic segmenation. Maybe use the predicted semantic segmenation. Only use the predicted semnatic segmentaion for testing when it got percie good.
        # 1. Update translation prediction
        # delta_v2 = delta_v.permute(1, 0, 2, 3).reshape(3, -1).T

        # pred_trans_batch = pred_trans[:, :, None, None].repeat(
        #     1, 1, h, w).permute(1, 0, 2, 3).reshape(3, -1).T

        cam_batch = cam[:, :, None, None].repeat(
            1, 1, h, w).permute(1, 0, 2, 3).reshape(4, -1)
        # image coordinates to euclidean distance
        pred_trans_new = get_delta_t_in_euclidean(
            delta_v.permute(1, 0, 2, 3).reshape(3, -1).T,
            t_src=pred_trans[:, :, None, None].repeat(
                1, 1, h, w).permute(1, 0, 2, 3).reshape(3, -1).T,
            fx=cam[:, :, None, None].repeat(1, 1, h, w).permute(
                1, 0, 2, 3).reshape(4, -1)[2, :][:, None],
            fy=cam[:, :, None, None].repeat(1, 1, h, w).permute(
                1, 0, 2, 3).reshape(4, -1)[3, :][:, None],
            device=self.device)

        delta_t = pred_trans_new - pred_trans[:, :, None, None].repeat(
            1, 1, h, w).permute(1, 0, 2, 3).reshape(3, -1).T
        delta_t = delta_t.T.reshape(3, bs, h, w).permute(1, 0, 2, 3)

        pred_trans_new = pred_trans_new.T.reshape(
            3, bs, h, w).permute(1, 0, 2, 3)
        # delta_t can be used for bookkeeping to keep track of the translation
        # limit delta_t to be within 10cm
        # val = self.exp.get('training', {}).get(
        #     'clamp_delta_t_pred', 0.1)
        # delta_t_clamp = torch.clamp(delta_t, -val, val)
        # pred_trans_batch = pred_trans_batch.view(
        #     delta_v.shape) + delta_t

        # _h, _w, _b = [0, 10, 15, 50], [0, 100, 200, 240], [0, 0, 1, 1]
        # for i in range(0, len(_h)):
        #     res = get_delta_t_in_euclidean(
        #         delta_v[_b[i], :, _h[i], _w[i]
        #                 ][None], t_src=pred_trans[_b[i], :][None],
        #         fx=cam[_b[i], 2].view(1, 1), fy=cam[_b[i], 3].view(1, 1), device=self.device)

        #     print('RESULT', res, 'Pre Compute',
        #           pred_trans_new[_b[i], :, _h[i], _w[i]])

        # calculate based on predicted semantic segmentation the pred_trans_mean
        true_res = idx[:, :, None, None].repeat(1, 1, h, w) + 1
        object_cor = gt_label_cropped == idx[:, :, None].repeat(1, h, w)

        valid_sum = torch.sum(object_cor.view(bs, -1), dim=1)
        valid_sum = torch.clamp(valid_sum, 1, 10.0e6)
        if (valid_sum == torch.ones(valid_sum.shape, device=self.device)).type(torch.uint8).any():
            logging.warning('Valid sum is 1')

        pred_trans_batch_valid = object_cor[:, None, :, :].repeat(
            1, 3, 1, 1) * pred_trans_new
        pred_trans_mean = torch.sum(
            pred_trans_batch_valid.view(bs, 3, -1), dim=2) / valid_sum[:, None].repeat(1, 3)
        pred_trans = pred_trans_mean

        # 2. Update rotation
        # Quaternion averageing base on http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf
        # https://github.com/christophhagen/averaging-quaternions
        # We expect similar orientation of the quaternions therfore mean averaging and then normalization !
        identity = torch.zeros(rotations.shape, device=self.device)
        identity[:, 0, :, :] = 1
        rotations_valid = (rotations + identity) * \
            object_cor[:, None, :, :].repeat(1, 4, 1, 1).type(torch.float32)
        rotations_mean = torch.sum(rotations_valid, dim=(
            2, 3)) / torch.sum(object_cor, dim=(1, 2))[:, None].repeat(1, 4)
        pred_rot_wxyz = compose_quat(pred_rot_wxyz, rotations_mean)

        # 3. Update pred_points
        base_new = quat_to_rot(
            pred_rot_wxyz.clone(), 'wxyz', device=self.device).unsqueeze(1)
        base_new = base_new.view(-1, 3, 3).permute(0, 2, 1)
        pred_points = torch.add(
            torch.bmm(model_points, base_new), pred_trans.unsqueeze(1))

        return pred_trans, pred_rot_wxyz, pred_points, delta_t

    def forward_pose_simple(self, delta_v, rotations, pred_trans, pred_rot_wxyz, model_points, cam, idx, gt_label_cropped):
        # Update translation
        bs = model_points.shape[0]

        pred_trans_mean = torch.mean(delta_v, dim=(2, 3))
        pred_trans_mean = pred_trans_mean * 0.05
        pred_trans = pred_trans + pred_trans_mean

        # 2. Update rotation
        identity = torch.zeros(rotations.shape, device=self.device)
        identity[:, 0, :, :] = 1
        rotations_valid = rotations + identity

        rotations_mean = torch.mean(rotations_valid, dim=(2, 3))
        pred_rot_wxyz = compose_quat(pred_rot_wxyz, rotations_mean)

        # 3. Update pred_points
        base_new = quat_to_rot(
            pred_rot_wxyz.clone(), 'wxyz', device=self.device).unsqueeze(1)
        base_new = base_new.view(-1, 3, 3).permute(0, 2, 1)
        pred_points = torch.add(
            torch.bmm(model_points, base_new), pred_trans.unsqueeze(1))

        return pred_trans, pred_rot_wxyz, pred_points, delta_v

    def training_step(self, batch, batch_idx):
        self._mode = 'train'
        st = time.time()
        total_loss = 0
        total_dis = 0
        nr = self.exp.get('visu', {}).get('number_images_log_train', 1)
        if self.counter_images_logged < nr:
            self.visu_forward = True
        else:
            self.visu_forward = False

        # forward
        dis, pred_r, pred_t, log_scalars = self(batch[0])

        # aggregate statistics per object (ADD-S sym and ADD non sym)
        bs = batch[0][0].shape[0]
        thr = self.exp.get('eval', {}).get('threshold_add', 0.02)
        # check if smaller ADD / ADD-s < 2cm
        within_add = torch.ge(torch.tensor(
            [thr] * bs, device=self.device), dis)

        loss = torch.mean(dis)
        self.visu_step(nr, batch, pred_r, pred_t, batch_idx)

        # # for epoch average logging
        try:
            self._dict_track['train_loss  [+inf - 0]'].append(float(loss))
            self._dict_track[f'train_adds_2cm  [0 - 1]'].append(
                float(torch.mean(within_add.type(torch.float32))))
        except:
            self._dict_track['train_loss  [+inf - 0]'] = [float(loss)]
            self._dict_track[f'train_adds_2cm  [0 - 1]'] = [
                float(torch.mean(within_add.type(torch.float32)))]

        # tensorboard logging
        tensorboard_logs = {'train_loss': float(loss)}

        tensorboard_logs = {**tensorboard_logs, **log_scalars}
        # 'dis': total_dis, 'log': tensorboard_logs,
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': {'L_Seg': log_scalars['loss_segmentation'], 'L_Add': log_scalars['loss_pose_add'], 'L_Tra': log_scalars[f'loss_translation']}}

    def validation_step(self, batch, batch_idx):
        self._mode = 'val'
        st = time.time()

        total_loss = 0
        total_dis = 0

        nr = self.exp.get('visu', {}).get('number_images_log_val', 1)
        if self.counter_images_logged < nr:
            self.visu_forward = True
        else:
            self.visu_forward = False

        # forward
        dis, pred_r, pred_t, log_scalars = self(batch[0])

        # aggregate statistics per object (ADD-S sym and ADD non sym)
        bs = batch[0][0].shape[0]
        unique_desig = batch[0][12]
        thr = self.exp.get('eval', {}).get('threshold_add', 0.02)
        # check if smaller ADD / ADD-s < 2cm
        within_add = torch.ge(torch.tensor(
            [thr] * bs, device=self.device), dis)

        loss = torch.mean(torch.sum(dis))

        self.visu_step(nr, batch, pred_r, pred_t, batch_idx)

        try:
            for _i in range(0, bs):
                self._dict_track['val_adds_dis  [+inf - 0]'].append(
                    float(dis[_i]))

            self._dict_track['val_loss  [+inf - 0]'].append(float(loss))
            self._dict_track[f'val_adds_2cm  [0 - 1]'].append(
                float(torch.mean(within_add.type(torch.float32))))
        except:
            self._dict_track['val_adds_dis  [+inf - 0]'] = [float(dis[0])]
            for _i in range(1, bs):
                self._dict_track['val_adds_dis  [+inf - 0]'].append(
                    float(dis[_i]))

            self._dict_track['val_loss  [+inf - 0]'] = [float(loss)]
            self._dict_track[f'val_adds_2cm  [0 - 1]'] = [
                float(torch.mean(within_add.type(torch.float32)))]

        for i in range(0, bs):
            # object loss for each object
            obj = int(unique_desig[1][i])
            obj = list(
                self.trainer.val_dataloaders[0].dataset._backend._name_to_idx.keys())[obj - 1]
            if f'val_{obj}_adds_dis  [+inf - 0]' in self._dict_track.keys():
                self._dict_track[f'val_{obj}_adds_dis  [+inf - 0]'].append(
                    float(dis[i]))
                self._dict_track[f'val_{obj}_adds_2cm  [0 - 1]'].append(
                    float(within_add[i]))
            else:
                self._dict_track[f'val_{obj}_adds_dis  [+inf - 0]'] = [
                    float(dis[i])]
                self._dict_track[f'val_{obj}_adds_2cm  [0 - 1]'] = [
                    float(within_add[i])]

        tensorboard_logs = {'val_loss': float(loss)}
        tensorboard_logs = {**tensorboard_logs, **log_scalars}

        val_loss = loss
        val_dis = loss
        return {'val_loss': val_loss, 'val_dis': val_dis, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        self._mode = 'test'
        total_loss = 0
        total_dis = 0

        st = time.time()

        nr = self.exp.get('visu', {}).get('number_images_log_test', 1)
        if self.counter_images_logged < nr:
            self.visu_forward = True
        else:
            self.visu_forward = False

        print(
            f'Visu Forward {self.visu_forward}, already logged {self.counter_images_logged}')

        # forward
        dis, pred_r, pred_t, log_scalars = self(batch[0])
        # aggregate statistics per object (ADD-S sym and ADD non sym)
        bs = batch[0][0].shape[0]
        unique_desig = batch[0][12]
        thr = self.exp.get('eval', {}).get('threshold_add', 0.02)
        # check if smaller ADD / ADD-s < 2cm
        within_add = torch.ge(torch.tensor(
            [thr] * bs, device=self.device), dis)

        loss = torch.mean(torch.sum(dis))

        self.visu_step(nr, batch, pred_r, pred_t, batch_idx)

        try:
            for _i in range(0, bs):
                self._dict_track['test_adds_dis  [+inf - 0]'].append(
                    float(dis[_i]))

            self._dict_track['test_loss  [+inf - 0]'].append(float(loss))
            self._dict_track[f'test_adds_2cm  [0 - 1]'].append(
                float(torch.mean(within_add.type(torch.float32))))
        except:
            self._dict_track['test_adds_dis  [+inf - 0]'] = [float(dis[0])]
            for _i in range(1, bs):
                self._dict_track['test_adds_dis  [+inf - 0]'].append(
                    float(dis[_i]))

            self._dict_track['test_loss  [+inf - 0]'] = [float(loss)]
            self._dict_track[f'test_adds_2cm  [0 - 1]'] = [
                float(torch.mean(within_add.type(torch.float32)))]

        for i in range(0, bs):
            # object loss for each object
            obj = int(unique_desig[1][i])
            obj = list(
                self.trainer.test_dataloaders[0].dataset._backend._name_to_idx.keys())[obj - 1]
            if f'test_{obj}_adds_dis  [+inf - 0]' in self._dict_track.keys():
                self._dict_track[f'test_{obj}_adds_dis  [+inf - 0]'].append(
                    float(dis[i]))
                self._dict_track[f'test_{obj}_adds_2cm  [0 - 1]'].append(
                    float(within_add[i]))
            else:
                self._dict_track[f'test_{obj}_adds_dis  [+inf - 0]'] = [
                    float(dis[i])]
                self._dict_track[f'test_{obj}_adds_2cm  [0 - 1]'] = [
                    float(within_add[i])]
        for key in log_scalars.keys():
            try:
                self._dict_track[f'test_{key}'].append(
                    float(log_scalars[key]))
            except:
                self._dict_track[f'test_{key}'] = [
                    float(log_scalars[key])]

        tensorboard_logs = {'test_loss': float(loss)}
        tensorboard_logs = {**tensorboard_logs, **log_scalars}

        test_loss = loss
        test_dis = loss
        return {'test_loss': test_loss, 'test_dis': test_dis, 'log': tensorboard_logs}

    def visu_step(self, nr, batch, pred_r, pred_t, batch_idx):
        if self.visu_forward:
            self.counter_images_logged += 1
            points, choose, img, target, model_points, idx = batch[0][0:6]
            depth_img, label_img, img_orig, cam = batch[0][6:10]
            gt_rot_wxyz, gt_trans, unique_desig = batch[0][10:13]
            self.visu_pose(batch_idx, pred_r[0], pred_t[0],
                           target[0], model_points[0], cam[0], img_orig[0], unique_desig, idx[0])

    def validation_epoch_end(self, outputs):
        avg_dict = {}
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
        if self.visualizer is None:
            self.visualizer = Visualizer(self.exp['model_path'] +
                                         '/visu/', self.logger.experiment)
        points = copy.deepcopy(target.detach().cpu().numpy())
        img = img_orig.detach().cpu().numpy()
        if self.exp['visu'].get('visu_gt', False):
            self.visualizer.plot_estimated_pose(tag='gt_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
                                                epoch=self.current_epoch,
                                                img=img,
                                                points=points,
                                                cam_cx=float(cam[0]),
                                                cam_cy=float(cam[1]),
                                                cam_fx=float(cam[2]),
                                                cam_fy=float(cam[3]),
                                                store=store)
            self.visualizer.plot_contour(tag='gt_contour_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
                                         epoch=self.current_epoch,
                                         img=img,
                                         points=points,
                                         cam_cx=float(cam[0]),
                                         cam_cy=float(cam[1]),
                                         cam_fx=float(cam[2]),
                                         cam_fy=float(cam[3]),
                                         store=store)

        t = pred_t.detach().cpu().numpy()
        r = pred_r.detach().cpu().numpy()

        rot = R.from_quat(re_quat(r, 'wxyz'))

        self.visualizer.plot_estimated_pose(tag='pred_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
                                            epoch=self.current_epoch,
                                            img=img,
                                            points=copy.deepcopy(
            model_points[:, :].detach(
            ).cpu().numpy()),
            trans=t.reshape((1, 3)),
            rot_mat=rot.as_matrix(),
            cam_cx=float(cam[0]),
            cam_cy=float(cam[1]),
            cam_fx=float(cam[2]),
            cam_fy=float(cam[3]),
            store=store)

        self.visualizer.plot_contour(tag='pred_contour_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
                                     epoch=self.current_epoch,
                                     img=img,
                                     points=copy.deepcopy(
            model_points[:, :].detach(
            ).cpu().numpy()),
            trans=t.reshape((1, 3)),
            rot_mat=rot.as_matrix(),
            cam_cx=float(cam[0]),
            cam_cy=float(cam[1]),
            cam_fx=float(cam[2]),
            cam_fy=float(cam[3]),
            store=store)

        render_img, depth, h_render = self.vm.get_closest_image_batch(
            i=idx.unsqueeze(0), rot=pred_r.unsqueeze(0), conv='wxyz')
        # get the bounding box !
        w = 640
        h = 480

        real_img = torch.zeros((1, 3, h, w), device=self.device)
        # update the target to get new bb

        base_inital = quat_to_rot(
            pred_r.unsqueeze(0), 'wxyz', device=self.device).squeeze(0)
        base_new = base_inital.view(-1, 3, 3).permute(0, 2, 1)
        pred_points = torch.add(
            torch.bmm(model_points.unsqueeze(0), base_inital.unsqueeze(0)), pred_t)
        # torch.Size([16, 2000, 3]), torch.Size([16, 4]) , torch.Size([16, 3])
        bb_ls = get_bb_real_target(
            pred_points, cam.unsqueeze(0))

        for j, b in enumerate(bb_ls):
            if not b.check_min_size():
                pass
            c = cam.unsqueeze(0)
            center_real = backproject_points(
                pred_t.view(1, 3), fx=c[j, 2], fy=c[j, 3], cx=c[j, 0], cy=c[j, 1])
            center_real = center_real.squeeze()
            b.move(-center_real[0], -center_real[1])
            b.expand(1.1)
            b.expand_to_correct_ratio(w, h)
            b.move(center_real[0], center_real[1])
            crop_real = b.crop(img_orig).unsqueeze(0)
            up = torch.nn.UpsamplingBilinear2d(size=(h, w))
            crop_real = torch.transpose(crop_real, 1, 3)
            crop_real = torch.transpose(crop_real, 2, 3)
            real_img[j] = up(crop_real)
        inp = real_img[0].unsqueeze(0)
        inp = torch.transpose(inp, 1, 3)
        inp = torch.transpose(inp, 1, 2)
        data = torch.cat([inp, render_img], dim=3)
        data = torch.transpose(data, 1, 3)
        data = torch.transpose(data, 2, 3)
        self.visualizer.visu_network_input(tag='render_real_comp_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
                                           epoch=self.current_epoch,
                                           data=data,
                                           max_images=1, store=store)

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

        store = self.env['p_ycb'] + '/viewpoints_renderings'
        if self.vm is None:
            self.vm = ViewpointManager(
                store=store,
                name_to_idx=dataset_train._backend._name_to_idx,
                nr_of_images_per_object=self.exp.get(
                    'vm', {}).get('nr_of_images_per_object', 1000),
                device=self.device,
                load_images=self.exp.get('vm', {}).get('load_images', False))

        return dataloader_train

    def test_dataloader(self):
        dataset_test = GenericDataset(
            cfg_d=self.exp['d_test'],
            cfg_env=self.env)
        store = self.env['p_ycb'] + '/viewpoints_renderings'
        if self.vm is None:
            self.vm = ViewpointManager(
                store=store,
                name_to_idx=dataset_test._backend._name_to_idx,
                nr_of_images_per_object=self.exp.get(
                    'vm', {}).get('nr_of_images_per_object', 1000),
                device=self.device,
                load_images=self.exp.get('vm', {}).get('load_images', False))

        dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                      **self.exp['loader'])
        return dataloader_test

    def val_dataloader(self):
        dataset_val = GenericDataset(
            cfg_d=self.exp['d_train'],
            cfg_env=self.env)

        store = self.env['p_ycb'] + '/viewpoints_renderings'
        if self.vm is None:
            self.vm = ViewpointManager(
                store=store,
                name_to_idx=dataset_val._backend._name_to_idx,
                nr_of_images_per_object=self.exp.get(
                    'vm', {}).get('nr_of_images_per_object', 1000),
                device=self.device,
                load_images=self.exp.get('vm', {}).get('load_images', False))

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

    # default used by the Trainer
    # TODO create early stopping callback
    # https://github.com/PyTorchLightning/pytorch-lightning/blob/63bd0582e35ad865c1f07f61975456f65de0f41f/pytorch_lightning/callbacks/base.py#L12
    early_stop_callback = EarlyStopping(
        monitor='avg_val_dis_float',
        patience=exp.get('early_stopping_cfg', {}).get('patience', 100),
        strict=True,
        verbose=True,
        mode='min'
    )

    # DEFAULTS used by the Trainer
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
        callbacks=[checkpoint_callback],
        early_stop_callback=early_stop_callback,
        default_root_dir=exp['model_path'])


        if exp.get('model_mode', 'fit') == 'fit':
            # lr_finder = trainer.lr_find(
            #     model, min_lr=0.0000001, max_lr=0.001, num_training=500, early_stop_threshold=100)
            # Results can be found in
            # lr_finder.results
            # lr_finder.suggestion()
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
