import yaml
import numpy as np
import collections

import os
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
import copy
import shutil


### edit this path
plr_path = "/cluster/home/jonfrey/PLR3"
template_path =plr_path + '/yaml/exp/exp_ws_deepim.yml'
save_dir = plr_path+'/yaml/auto'
model_base_path = '/cluster/work/riner/users/PLR-2020/jonfrey/models/runs'
#pruge ansible folder first
shutil.rmtree(save_dir)


#open the template
with open(template_path) as f:
  data = yaml.load(f, Loader=yaml.FullLoader)
  print(data)
jobs = []


########################### EDIT YOUR EXPERIMENTS HERE #####################
#4h_load_<MODELNAME>_exp_<you_conduct>
folder_name = 'deep_im_lr'
tag = 'TAG'
host = 'leonhard' #'jonas' yash
ram = 64
scratch = 350
cores = 20
gpus = 1
time = '3:59' # '23:59' 

os.makedirs(f'{save_dir}/{folder_name}', exist_ok=True)
#lr = np.linspace(start=0.005, stop=0.00001, num=6).tolist()
ls_lr = np.logspace(start=-6, stop=-8, num=3,base=10).tolist()

i = [0]
def send(i,tag,data):
  #baseline
  with open(f'{save_dir}/{folder_name}/{i[0]}_{tag}.yml' , 'w') as f:
    jobs.append( {'exp': f'{save_dir}/{folder_name}/{i[0]}_{tag}.yml'} )
    data['model_path'] =  f'{model_base_path}/{folder_name}/{i[0]}_{tag}'
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    i[0] = i[0] +1

for lr in ls_lr:
  data['lr'] = lr
  send(i,f'lr_{lr}',data)


################################## DONE ################################
bsub = f'bsub -n {int(cores)} -W {time} -R "rusage[mem={int(ram*1000/cores)},ngpus_excl_p={int(gpus)}]" -R "rusage[scratch={int(scratch*1000/cores)}]" $HOME/PLR3/scripts/leonhard/submit.sh '
for job in jobs:
  exp = job['exp']
  arg = f' --env=yaml/env/env_leonhard_jonas.yml --exp {exp}'
  print("Send command: ", bsub+arg, "\n \n")
  os.system(bsub+arg)

"""
#create path file 
store = {'jobs':jobs}
with open(experiment_file, 'w') as f:
	yaml.dump(store, f)

print ("\n \n Submitting %d Jobs, Host: %s, Folder: %s"%(i[0],host , data['tensorboard_path']) )
#send jobs to cluster via ansible

if host == 'jonas':
	bashCommand = "cd "+  plr_path + " && sudo ansible-playbook scripts/ansible_auto/ansible_auto.yml"
elif  host == 'yash':
	bashCommand = "cd "+  plr_path + " && sudo ansible-playbook scripts/ansible_auto/ansible_auto_yash.yml"
os.system(bashCommand)
"""