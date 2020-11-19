from pathlib import Path
import os
import argparse
import yaml
from os.path import expanduser
"""
For execution run:
python scripts/leonhard/schedule_all_exp_in_folder.py --exp=t24h --time=24

"""
parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='t24h',  required=True,
                    help='Folder containing experiment yaml file.')
parser.add_argument('--time', default=24, required=True,
                    help='Runtime.')
parser.add_argument('--mem', default=10240, help='Min GPU Memory')
parser.add_argument('--model_base_path', default=None, help='Path tp models/runs folder')
parser.add_argument('--out', default=True)

args = parser.parse_args()
out = args.out
mp = args.model_base_path 
print(args.time)
mem = args.mem
if args.time == '120':
  s1 = '119:59'
elif args.time == '24':
  s1 = '23:59'
elif args.time == '4':
  s1 = '3:59'
  print('Working')
elif  isinstance( args.time, str):
  s1 = args.time
else:
  raise Exception

home = expanduser("~")

p = f'{home}/PLR3/yaml/exp/{args.exp}/'
exps = [str(p) for p in Path(p).rglob('*.yml')]


model_paths = []

for j,e in enumerate(exps):
  print(e)
  with open(e) as f:
    doc = yaml.load(f, Loader=yaml.FullLoader) 
  doc['visu']['log_to_file'] = True
  if not (mp is None):
    p =  doc['model_path'].find('/models/runs') #+ int( len('models/runs') )
    doc['model_path'] =  mp + doc['model_path'][p:] 

  with open(e, 'w+') as f:
    model_paths.append( doc['model_path'] )
    yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
  

for j, e in enumerate(exps):
  if out:
    p = model_paths[j].split('/')
    p = '/'.join(p[:-1])
    Path(p).mkdir(parents=True, exist_ok=True)

    name = model_paths[j].split('/')[-1] + str(j) + '.out'
    o = f""" -oo {p}/{name} """
  else:
    o = ' '
  cmd = f"""cd {home}/PLR3 && bsub{o}-n 20 -W {s1} -R "rusage[mem=3000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>={mem}]" -R "rusage[scratch=16000]" ./scripts/leonhard/submit.sh --exp={e} --env=yaml/env/env_leonhard_jonas.yml"""
  os.system(cmd)
  print(f'Run: {j}, Exp: {e}, Time: {s1}h')