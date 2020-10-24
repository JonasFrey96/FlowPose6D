from pathlib import Path
import os

p = '/cluster/home/jonfrey/PLR3/yaml/exp/t/'
exps = [str(p) for p in Path(p).rglob('*.yml')]


for j, e in enumerate(exps):
  cmd = f"""cd /cluster/home/jonfrey/PLR3 && bsub -n 16 -W 3:59 -R "rusage[mem=3000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -R "rusage[scratch=25000]" ./scripts/leonhard/submit.sh --exp={e} --env=yaml/env/env_leonhard_jonas.yml"""
  os.system(cmd)
  print(f'Run {j}, Exp {e}')
