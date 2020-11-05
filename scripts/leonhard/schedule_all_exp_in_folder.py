from pathlib import Path
import os
import argparse
"""
For execution run:
python scripts/leonhard/schedule_all_exp_in_folder.py --exp=t24h --time=24

"""


parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='t24h',  required=True,
                    help='Folder containing experiment yaml file.')
parser.add_argument('--time', default=24, required=True,
                    help='Runtime.')
args = parser.parse_args()
print(args.time)

if args.time == '24':
  s1 = '23:59'
elif args.time == '4':
  s1 = '3:59'
  print('Working')
elif  isinstance( args.time, str):
  s1 = args.time
else:
  raise Exception

p = f'/cluster/home/jonfrey/PLR3/yaml/exp/{args.exp}/'
exps = [str(p) for p in Path(p).rglob('*.yml')]

import yaml
for j,e in enumerate(exps):
  print(e)
  with open(e) as f:
      doc = yaml.load(f, Loader=yaml.FullLoader) 
  doc['visu']['log_to_file'] = True
  with open(e, 'w+') as f:
      yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
  

for j, e in enumerate(exps):
  
  cmd = f"""cd /cluster/home/jonfrey/PLR3 && bsub -n 20 -W {s1} -R "rusage[mem=3000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -R "rusage[scratch=16000]" ./scripts/leonhard/submit.sh --exp={e} --env=yaml/env/env_leonhard_jonas.yml"""
  os.system(cmd)
  print(f'Run: {j}, Exp: {e}, Time: {s1}h')