#!/bin/bash
# Load modules
#  #hdf5/1.10.1 > /dev/null
# Install dependencies
module load python_gpu/3.7.4 gcc/6.3.0
source ~/.bashrc
conda activate track2
cd $HOME/PLR3
/cluster/work/riner/users/PLR-2020/jonfrey/conda/envs/track2/bin/python tools/lightning_DeepIM.py $@
