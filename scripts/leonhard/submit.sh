#!/bin/bash
# Load modules
#  #hdf5/1.10.1 > /dev/null
# Install dependencies
module load python_gpu/3.7.4 gcc/6.3.0
source ~/.bashrc
conda activate track3
cd $HOME/PLR3
$HOME/miniconda3/envs/track3/bin/python tools/main.py $@
