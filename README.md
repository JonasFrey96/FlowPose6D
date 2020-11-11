# TrackThis

![](doc/TrackThis%20Kalman%20Filter.png)

## Instructions
How to run the network: 
```
cd ~ 
git clone -b pixelwise-refiner https://github.com/JonasFrey96/PLR3.git
```
Install track3 conda enviorment track3:
```
cd ~/PLR3 && conda env create -f scripts/environment.yml
```

Install efficientnet backbone
```
cd ~ && git clone https://github.com/JonasFrey96/EfficientNet-PyTorch.git
cd /EfficientNet-PyTorch
pip install -e .
```

Downloading the EfficientNet pertrained weights: 
Dont start a bjobs !!!
```
conda activate track3 
pip install opencv-python 
cd ~/PLR3 && python src/model/efficient_disparity.py
```


Schedule Job on leonhard: (The exp flag is the path to a folder relative to PLR3/yaml/exp containg mutiple .yml exp files)
Time is either in the format 4 24 or as a string 10:11
```
conda activate track3 
python scripts/leonhard/schedule_all_exp_in_folder.py --exp=t24h --time=24
```

The only thing that might need some adaption is the `scripts/leonhard/submit.sh' script which contains the direct path to the conda enviroment. 
Somehow i ran into issues with my installation and this fixed the bug. You might need to change tha last line to 
```
python tools/lightning_DeepIM.py $@
```
or give your conda python track3 path directly as I did. ( but i guess its not necesarry.)


For debugging do the flowing: 
```
module load python_gpu/3.7.4 gcc/6.3.0
```

```
bsub -Is -n 20 -W 3:59 -R "rusage[mem=3000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -R "rusage[scratch=16000]" bash
conda activate track3
```

Run the network with:
```
conda activate track3
```

Replace the exp file. In the exp file put the visu log_to_file = False so the network logs to the terminal. 
```
cd ~/PLR3 && python tools/lightning_DeepIM.py --env=yaml/env/env_leonhard_jonas.yml --exp=yaml/exp/t4h/exp1.yml
```


### General

This is the base setup to performe experiments for 6D object detection.
The main file to run and evaluate your network can be found in `tools/lightning.py`

In `tools/lightning.py` implement your training procedure.
Two config files have to be passed into this file the enviroment `env` and experiment `exp` file.

- `env` defines your local setup and global paths outside of the repository.
- `exp` contains all hyperparameters and training steup needed to train your model.

Both files are passed into the lightning module should be stored in the `yaml/`-folder.

### Installation:

Prerequests:

- Cuda Version: 10.2

Install conda env:

```
cd PLR2
conda env create -f environment.yml
```

Install KNN (not tested):

```
conda activate track
cd PLR2/lib/knn
python setup.py build
cp -r lib.linux-x86_64-3.7/* ./
```

Setting up global variables:
Got to:
`yaml/env/env_ws.yml`
and edit the following pathts:

#### Natrix Uusefull Commands:

```
nohup python tools/lightning_DeepIM.py > nohup/learn_segmentation_faster_lr.log &
```

#### Leonhard Usefull Commands:

```
conda activate track2

du -sh
conda env export --name machine_perception > machine_perception.yml
tar -cvf data_syn.tar ./data_syn


module load eth_proxy python_gpu/3.7.7 gcc/6.3.0
python tools/lightning_DeepIM.py --env=yaml/env/env_leonhard_yash.yml
/cluster/work/riner/users/PLR-2020/jonfrey/conda/envs/track2/bin/python --env=yaml/env/env_leonhard_yash.yml

#not debug
cd ~/PLR3 && /cluster/work/riner/users/PLR-2020/jonfrey/conda/envs/track2/bin/python tools/lightning_DeepIM.py --env=yaml/env/env_leonhard_jonas.yml

#debug
cd ~/PLR3 && /cluster/work/riner/users/PLR-2020/jonfrey/conda/envs/track2/bin/python tools/lightning_DeepIM.py --env=yaml/env/env_leonhard_jonas.yml --exp=yaml/exp/exp_ws_deepim_debug_leon.yml

# Copying model to Natrix from Leonhard:
scp jonfrey@login.leonhard.ethz.ch:/cluster/work/riner/users/PLR-2020/jonfrey/models/runs/efficient_disparity_b1/2020-10-28T00:29:24_non_overfit_24h_lr-4_obj2_loaded24h/* /media/scratch1/jonfrey/models/runs/cluster/

# Interactive with good GPU:
bsub -Is -n 16 -W 3:59 -R "rusage[mem=3000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -R "rusage[scratch=20000]" bash
# Interactive with 4-GPU:
bsub -Is -n 16 -W 3:59 -R "rusage[mem=3000,ngpus_excl_p=4]" -R "select[gpu_mtotal0>=10240]" -R "rusage[scratch=20000]" bash

bsub -Is -n 20 -W 3:59 -R "rusage[mem=3000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -R "rusage[scratch=16000]" bash

bsub -Is -n 40 -W 3:59 -R "rusage[mem=1500,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -R "rusage[scratch=8000]" bash

bsub -Is -n 40 -W 3:59 -R "rusage[mem=1500,ngpus_excl_p=4]" -R "select[gpu_mtotal0>=10240]" -R "rusage[scratch=8000]" bash

cd ~/PLR3 && /cluster/home/jonfrey/miniconda3/envs/track3/bin/python tools/lightning_DeepIM.py --env=yaml/env/env_leonhard_jonas.yml --exp=yaml/exp/t4h/exp1.yml

cd ~/PLR3 && CUDA_LAUNCH_BLOCKING=1 /cluster/home/jonfrey/miniconda3/envs/track3/bin/python tools/lightning_DeepIM.py --env=yaml/env/env_leonhard_jonas.yml --exp=yaml/exp/exp_ws_deepim_debug_leon.yml
CUDA_LAUNCH_BLOCKING=1 python tools/lightning_DeepIM.py --env=yaml/env/env_leonhard_jonas.yml --exp=yaml/exp/exp_ws_deepim_debug_leon.yml


# Final sub:
bsub -n 16 -W 3:59 -R "rusage[mem=3000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -R "rusage[scratch=20000]" ./scripts/leonhard/submit.sh --exp=yaml/exp/exp/exp1.yml --env=yaml/env/env_leonhard_jonas.yml 

```

Starting Tensorboard:
`./scripts/leonhard/start_tensorboard.sh deep_im`

### Lightning Module:

#### Dataloaders:

Can be configured for training, validation and testing.
The test data is a complete hold-out-data and should only be used before we publish our paper.

The validation data is used to tune hyperparameters.

The configuration of the dataloader looks as follows:

```
d_val:
  name: "ycb"
  objects: 21
  num_points: 1000
  num_pt_mesh_small: 500
  num_pt_mesh_large: 2300
  obj_list_fil: null
  obj_list_sym:
    - 12
    - 15
    - 18
    - 19
    - 20
  batch_list_cfg:
    sequence_names: null #either null or what name should be in the sequence for example to only plot hard interactions
    seq_length: 3
    fixed_length: true
    sub_sample: 1
    mode: "test" #dense_fusion_test
    add_syn_to_train: false
  noise_cfg:
    status: false
    noise_trans: 0.0
  output_cfg:
    force_one_object_visible: true
    status: false
    refine: false
    add_depth_image: false
    visu:
      status: true
      return_img: true
```

**batch_list_cfg:** configures the data that is used:

5 Options exists:

- dense_fusion_test: (auto sequence length of 1, exact same data as for DenseFusion)
- dense_fusion_train: (auto sequence length of 1, exact same data as for DenseFusion)
- test: (sequence data can be generated length > 1, same sequences as for DenseFusion selected)
- train: (sequence data can be generated length > 1, same sequences as for DenseFusion selected (only real data used) )
- train_inc_syn: (real and synthetic data used)

**output_cfg:** specifies what is returned by the dataloader

**Important:**
In the lightning module the validation and training data are seperated via **_sklearn.model_selection.train_test_split_** in `def train_dataloader(self):` Since everything is seeded with 42 the train and validation split is reproducable. In the curent validation setuo the validation data constists out of synthetic and real data.

#### Logging:

Each run is stored in a seperated folder.
The folder is configured in: `exp_ws` \_model_path\_

self.hparams are automatically stored as a yaml. In our case we add the env and exp to reproduce exactly our experiment.

Logging is done via Tensorboard:
When using VS-Code this automatically does all the port forwarding for you:
`tensorboard --logdir=./ --host=0.0.0.0`

Have a look at the PyTorch lightning Tensorboard integration.
Simply return in `def validation_epoch_end` or `def validation_step` a **'log': tensorboard_logs** (tensorboard_logs contains a dict with all metrices that should be logged)

#### Visu:

Images can be added via self.logger.experiment which is a TensorBoard Summary writer.

#### ModelCheckpoints and Loading:

Check PytorchLightning Documentation,

### Extending the Network:

Add modules to the `src` folder or files to `lib`.

## Keypoint labeling

The subdirectory `labeling/` contains a small tool to label `.obj` meshes. The tool can be built using:

```
mkdir -p labeling/build/
cd labeling/build/
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

Once built, it can be run with `./labeler <path-to-data-dir>`. Currently, it assumes the data directory is in the same format as `ycb_video_models`. That is, it contains one subdirectory for each object, which then again contain an obj file called `textured.obj` and a texture file `texture_map.png`.

## License

Licensed under the [MIT License](LICENSE)
