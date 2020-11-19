
import os 
import sys 
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))
sys.path.append(os.path.join(os.getcwd() + '/lib'))

import shutil
import datetime
import argparse
import signal
import coloredlogs
coloredlogs.install()

import torch
from pytorch_lightning import seed_everything,Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from .lightning import TrackNet6D
from helper import pad
from loaders_v2 import ConfigLoader
from helper import move_dataset_to_ssd
from helper import move_background


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)

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
    parser.add_argument('--exp', type=file_path, default='/home/jonfrey/PLR3/yaml/exp/breaks.yml',  # required=True,
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

    if exp.get('model_mode', 'fit') == 'fit':
        trainer.fit(model)

    elif exp.get('model_mode', 'fit') == 'test':
        trainer.test(model)

    else:
        print("Wrong model_mode defined in exp config")
        raise Exception
