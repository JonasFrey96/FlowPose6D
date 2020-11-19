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
import math

# misc
import numpy as np
import pandas as pd
import random
from math import pi
from math import ceil
import coloredlogs
import datetime
from pathlib import Path
import yaml


sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))
sys.path.append(os.path.join(os.getcwd() + '/lib'))
# src modules
from helper import pad
import torch
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_DeepIM import TrackNet6D
coloredlogs.install()


from loaders_v2 import ConfigLoader

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

  parser = argparse.ArgumentParser()
  # parser.add_argument('--exp', type=file_path, default='/home/jonfrey/PLR3/yaml/exp/exp_ws_deepim_debug_natrix.yml',  # required=True,
  #                     help='The main experiment yaml file.')
  parser.add_argument('--env', type=file_path, default='yaml/env/env_natrix_jonas.yml',
                      help='The environment yaml file.')
  
  args = parser.parse_args()
  env_cfg_path = args.env
  
  # exp_cfg_paths = [
  # '/media/scratch1/jonfrey/models/runs/efficient_disparity_scaled_24/2020-11-10T18:22:10_b6_depth_no_skip_l1/exp99.yml',
  # '/media/scratch1/jonfrey/models/runs/evaluation/flow-disp/2020-11-14T00:11:35_lr-5/exp3.yml']
  
  base_path = '/media/scratch1/jonfrey/models/runs/cluster_final_report/24h_training'

  exp_cfg_paths = [str(p) for p in Path(base_path).rglob('*.yml') if str(p).find('env') == -1]

  eval_results_folder = '/media/scratch1/jonfrey/models/evaluate/24h_training/perfect_init_estimate'

  new_exps = []
  for j, e in enumerate(exp_cfg_paths):
    with open(e) as f:
      doc = yaml.load(f, Loader=yaml.FullLoader) 

    p =  eval_results_folder + '/'.join( e.split('/')[-3:] )
    search_dir_for_load =  '/'.join( e.split('/')[:-1] ) 
    possible_checkpoints = [str(p) for p in Path(search_dir_for_load).rglob('*.ckpt')]
    e_max = -1
    best_ckp = ''
    for ckp in possible_checkpoints:
      if ckp.find( 'last.ckp') != -1:
        best_ckp = ckp
      # s = ckp.find( 'epoch=')
      # e = ckp.find( '-avg_val_disparity_float=')
      # print(ckp)
      # print(ckp[s:])
      # print(ckp[:e])
      # epoch = int(ckp[s+6:e])
      # if epoch > e_max: 
      #   e_max = epoch
      #   best_ckp = ckp

    
    doc['checkpoint_load'] = best_ckp
    doc['checkpoint_restore'] = True
    doc['trainer']['limit_test_batches'] = 1.0
    doc['visu']['number_images_log_train'] = 0
    doc['visu']['number_images_log_test'] = 0
    doc['visu']['number_images_log_val'] = 0 
    doc['visu']['always_calulate'] = True
    doc['visu']['full_val'] = True
    doc['visu']['log_to_file'] = True
    p =  eval_results_folder + '/'.join( e.split('/')[-3:-1] )
    doc['model_path'] = p
    doc['loader']['batch_size'] = 1
    doc['loader']['num_workers'] = 0
    doc['loader']['shuffle'] = True
    doc['loader']['pin_memory'] = True
    doc['d_test'] = copy.deepcopy(doc['d_train'])
    # doc['d_test']['batch_list_cfg']['mode'] = 'dense_fusion_test'
    doc['d_test']['batch_list_cfg']['mode'] = 'test'
    doc['d_test']['batch_list_cfg']['seq_length'] = 10
    doc['d_test']['batch_list_cfg']['sub_sample'] = 5
    doc['d_test']['batch_list_cfg']['add_syn_to_train'] = False

    doc['visu']['on_pred_fail'] = 0.03
    doc['d_test']['noise_translation'] = 0.001
    doc['d_test']['noise_rotation'] = 1
    
    new_file = p+f'/model{j}.yml'
    new_exps.append(new_file)
    try:
      os.makedirs(p)
    except:
      pass
    with open(new_file, 'w') as f:
      print(f'Created exp {j} at {new_file}')
      yaml.dump(doc, f, default_flow_style=False, sort_keys=False)



  
  env = ConfigLoader().from_file(env_cfg_path).get_FullLoader()
  env_cfg_path = args.env
  for exp_cfg_path in new_exps:
    seed_everything(42)
  # for exp_cfg_path in exp_cfg_paths:
    exp = ConfigLoader().from_file(exp_cfg_path).get_FullLoader()
    model_path= exp['model_path']
    # copy config files to model path
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print((pad("Generating network run folder")))
    else:
        print((pad("Network run folder already exits")))

    exp_cfg_fn = os.path.split(exp_cfg_path)[-1]
    env_cfg_fn = os.path.split(env_cfg_path)[-1]

    exp, env = move_dataset_to_ssd(env, exp)
    exp, env = move_background(env, exp)
    dic = {'exp': exp, 'env': env}
    model = TrackNet6D(**dic)

    early_stop_callback = EarlyStopping(
        monitor='avg_val_disparity',
        patience=exp.get('early_stopping_cfg', {}).get('patience', 100),
        strict=False,
        verbose=True,
        mode='min',
        min_delta = exp.get('early_stopping_cfg', {}).get('min_delta', -0.1)
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=exp['model_path'] + '/{epoch}-{avg_val_disparity_float:.4f}',
        verbose=True,
        monitor="avg_val_disparity",
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
        default_root_dir=exp['model_path'],
        callbacks=[early_stop_callback])

    trainer.test(model)
