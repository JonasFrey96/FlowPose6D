import yaml
import numpy as np
import collections

import os
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
import copy
import shutil
from pathlib import Path
import time
p = '/media/scratch1/jonfrey/models/evaluate/tracking/tracking_videos_low_noise_model/visu'
images = [str(_p) for _p in Path(p).rglob('*.jpg')]
# 

tags = ['0_Flow_Gradient_left_gt__right_pred_test_nr_', '0_Flow_Gradient_Crop_left_gt__right_pred_test_nr_', """0_Pose_estimate_(GT POSE, right pred_flow__gt_flow)_test_nr_""" ,"""0_Pose_estimate_(left gt_flow__gt_label, right h_pred_flow__pred_label)_test_nr_"""]

for tag in tags:
  for i in images:
    start= i.find(tag) + len(tag)
    stop = i.find('.jpg')
    index = i[start:stop]
    index = '0'* (6- len( i[start:stop])) + index 
    new = i[:start] + index + '.jpg'

    if i != new:
      i = i.replace('(','\(')
      i = i.replace(')','\)')
      i = i.replace(' ','\ ')

      new = new.replace('(','_')
      new = new.replace(')','_')
      new = new.replace(' ','_')

      os.system(f'mv {i} {new}')


  
  tag = tag.replace('(','_')
  tag = tag.replace(')','_')
  tag = tag.replace(' ','_')
  os.system("""cd """ +p+ """ && ffmpeg -framerate 5 -f image2 -pattern_type glob -framerate 12 -i '*"""+ tag+ """*.jpg' """+p[:-5]+"""/000"""+tag+ """movie.avi""")