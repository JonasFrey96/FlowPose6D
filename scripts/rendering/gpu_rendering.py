
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # 'egl' osmesa

import pyglet
#pyglet.options['headless'] = True
import trimesh
import numpy as np
import trimesh
import pyrender
# print(pyglet)
# from random import choice
# import random
# import PIL

from math import pi
from PIL import Image
# import imageio
import copy
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
# import gc
import cv2
import scipy.io as scio

obj = '005_tomato_soup_can'
model = '/media/scratch1/jonfrey/datasets/YCB_Video_Dataset/models'
base = '/media/scratch1/jonfrey/datasets/YCB_Video_Dataset/data/0003'
idx1 = '000010'
# im_src = cv2.imread(f'{base}/{idx1}-color.png', cv2.IMREAD_COLOR)
# depth_src = (cv2.imread(f'{base}/{idx1}-depth.png',
#                        cv2.IMREAD_UNCHANGED).astype(np.float32) / 10000)
# pose_src = scio.loadmat(f'{base}/{idx1}-meta.mat')['poses'][:, :, 0]
# trimeshes = [trimesh.load(x + '/geometry.ply') fosud
# trimesh.load(x + '/geometry.ply')) for x in seq_paths]
# vertices, indices = data.objload(f'{model}/{obj}/textured.obj', rescale=False)
# texture_map = data.load(f'{model_folder}/texture_map.png')[::-1, :, :])

bill_trimesh = trimesh.load(f'{model}/{obj}/textured.obj')
mesh = pyrender.Mesh.from_trimesh(bill_trimesh, smooth=True, wireframe=False)
scene = pyrender.Scene(bg_color=(0, 255, 0, 255))
pose_obj = np.eye(4)

# pose_obj[:3,
#          3] = [0, 0, 1]
# scene.add(mesh, pose=copy.copy(pose_obj))

# pose_obj[:3,
#          3] = [0, 1, 0]
# scene.add(mesh, pose=copy.copy(pose_obj))

# pose_obj[:3,
#          3] = [1, 0, 0]
# scene.add(mesh, pose=copy.copy(pose_obj))

pose_obj[:3, 3] = [0, 0, -0.3]
pose_obj[:3, :3] = R.from_euler('xyz', [90, 0, 0]).as_matrix()

scene.add(mesh, pose=copy.copy(pose_obj))

# pose_obj[:3,
#          3] = [0, -1, 0]
# scene.add(mesh, pose=copy.copy(pose_obj))

# pose_obj[:3,
#          3] = [-1, 0, 0]
# scene.add(mesh, pose=copy.copy(pose_obj))

# pose_obj[:3,
#          3] = [0, 0, 1]
# scene.add(mesh, pose=copy.copy(pose_obj))

ren = pyrender.OffscreenRenderer(640, 480, point_size=1)

# Set camera and lighting
cx = 312.9869
cy = 241.3109
fx = 1066.778
fy = 1067.487
camera = pyrender.IntrinsicsCamera(
    fx, fy, cx, cy, znear=0.1, zfar=2, name=None)
camera_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0.0, 0],
    [0.0, 0, 1, 0],
    [0.0, 0.0, 0.0, 1.0],
])

scene.add(camera, pose=camera_pose)
light = pyrender.SpotLight(color=np.ones(3), intensity=35.0,
                           innerConeAngle=np.pi * 0.1)
scene.add(light, pose=camera_pose)
color, depth = ren.render(scene)
#print(f'Rendering took {stop-start}s')
fig = plt.figure()
plt.axis("off")
fig.add_subplot(1, 2, 1)
plt.imshow(depth)
fig.add_subplot(1, 2, 2)
plt.imshow(color)

plt.savefig('scripts/color-depth.png', dpi=600)
# print("render_time:", time.time() - t_render)
# Show the images
# t_render = time.time()
# plt.axis('off')
# print(color)
