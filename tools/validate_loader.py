
import os 
import sys 
import yaml
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))
sys.path.append(os.path.join(os.getcwd() + '/lib'))

from loaders_v2 import ConfigLoader
from loaders_v2 import GenericDataset
import torch
import time
import datetime
exp = ConfigLoader().from_file('yaml/exp/exp_natrix.yml').get_FullLoader()
env = ConfigLoader().from_file('yaml/env/env_natrix_jonas.yml').get_FullLoader()

dataset_test = GenericDataset(
            cfg_d=exp['d_test'],
            cfg_env=env)
store = env['p_ycb'] + '/viewpoints_renderings'
dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                      batch_size = 1,
                                      num_workers = 15,
                                      pin_memory= False,
                                      shuffle= False)
print(len(dataloader_test))
st = time.time()
for j, b in enumerate( dataloader_test ): 
  if j % 50 == 0 and j != 0:
    ti = (time.time()-st)/j *len(dataset_test)
    t1 = str(datetime.timedelta(seconds=(time.time()-st) ))
    left = str(datetime.timedelta(seconds=ti))
    
    print(f' {j}/{len(dataset_test)}, Time {t1}, Left: {left} ,Idx: {b[0][0]}')