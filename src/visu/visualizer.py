import sys
import os
import random
if __name__ == "__main__":
    # load data
    os.chdir('/home/jonfrey/PLR2')
    sys.path.append('src')
    sys.path.append('src/dense_fusion')

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R
import copy
import k3d
import cv2
import io


from visu import save_image
from helper import re_quat
from helper import BoundingBox
from matplotlib import cm
from torchvision import transforms
import math
from math import pi
jet = cm.get_cmap('jet')
SEG_COLORS = (np.stack([jet(v)
                        for v in np.linspace(0, 1, 22)]) * 255).astype(np.uint8)
SEG_COLORS_BIN = (np.stack([jet(v)
                        for v in np.linspace(0, 1, 2)]) * 255).astype(np.uint8)

def backproject_points(p, fx, fy, cx, cy):
    """
    p.shape = (nr_points,xyz)
    """
    # true_divide
    u = torch.round((torch.div(p[:, 0], p[:, 2]) * fx) + cx)
    v = torch.round((torch.div(p[:, 1], p[:, 2]) * fy) + cy)

    if torch.isnan(u).any() or torch.isnan(v).any():
        u = torch.tensor(cx).unsqueeze(0)
        v = torch.tensor(cy).unsqueeze(0)
        print('Predicted z=0 for translation. u=cx, v=cy')
        # raise Exception

    return torch.stack([v, u]).T


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# if useing multiplot decorater make sure to catch if method is def not to plot
def multiplot(func):
    def wrap(*args, **kwargs):
        if kwargs.get('method', 'def') == 'def':
            return func(*args,**kwargs)
        
        elif kwargs.get('method', 'def') == 'left':
            res = func(*args,**kwargs)
            args[0].storage_left = res
            
        elif kwargs.get('method', 'def') == 'right':
            res = func(*args,**kwargs)
            args[0].storage_right = res
        if args[0].storage_right is not None and args[0].storage_left is not None:
            s = args[0].storage_right.shape
            img_f = np.zeros( (int(s[0]),int(s[1]*2), s[2]), dtype=np.uint8 )
            img_f[:,:s[1]] = args[0].storage_left
            img_f[:,s[1]:] = args[0].storage_right
            args[0].storage_left = None
            args[0].storage_right = None
            if kwargs.get('store', True):
                save_image(img_f, tag=str(kwargs.get('epoch', 'Epoch_Is_Not_Defined_By_Pos_Arg')) + '_' + kwargs.get('tag', 'Tag_Is_Not_Defined_By_Pos_Arg') , p_store=args[0].p_visu)

            if args[0].writer is not None:
                args[0].writer.add_image(kwargs.get('tag', 'Tag_Is_Not_Defined_By_Pos_Arg') , 
                    img_f.astype(np.uint8), 
                    global_step=kwargs.get('epoch', 'Epoch_Is_Not_Defined_By_Pos_Arg'), 
                    dataformats='HWC')
            if kwargs.get('jupyter', False):
                display(Image.fromarray(img_f.astype(np.uint8)))
        return func(*args,**kwargs)
    return wrap

class Visualizer():
    def __init__(self, p_visu, writer=None):
        if p_visu[-1] != '/':
            p_visu = p_visu + '/'
        self.p_visu = p_visu
        self.writer = writer

        if not os.path.exists(self.p_visu):
            os.makedirs(self.p_visu)

        # for mutliplot decorator 
        self.storage_left = None
        self.storage_right = None

        self.flow_scale= 1000
        Nc = int( np.math.pi*2 * self.flow_scale)
        cmap = plt.cm.get_cmap('hsv', Nc)
        self.flow_cmap = [cmap(i) for i in range(cmap.N)]
        

    @multiplot
    def flow_to_gradient(self, tag, epoch,
        img, flow, mask,tl=[0,0], br=[479,639],
        store=False, jupyter=False, method='def'):
        """
        img torch.tensor(h,w,3)
        flow torch.tensor(h,w,2)
        mask torch.tensor(h,w) BOOL
        call with either: 
        flow_to_gradient( np.uint8( real_img.numpy() ), flow.clone(), (gt_label_cropped == idx+1) )
        flow_to_gradient( np.uint8( img_orig.numpy() ), flow.clone(), (label_img == idx+1),  real_tl, real_br)
        """


        flow = flow
        amp = torch.norm(flow, p=2, dim=2)
        amp = amp / (torch.max(amp)+1.0e-6)  # normalize the amplitude
        dir_bin = torch.atan2(flow[:, :, 0], flow[:, :, 1])
        dir_bin *= self.flow_scale
        dir_bin = dir_bin.type(torch.long)

        h,w = 480,640
        arr = np.zeros( (h,w,4), dtype=np.uint8)
        arr_img = np.ones( (h,w,4), dtype=np.uint8) *255
        arr_img[:,:,:3] = img
        for u_ ,u in enumerate( np.linspace( float( tl[0] ) , float( br[0] ), num=h).tolist() ):
            u = int(u)
            for v_, v in enumerate( np.linspace( float( tl[1] ) , float( br[1] ), num=w).tolist()):
                v = int(v)
                if u > 0 and u < h and v > 0 and v < w:
                    arr[u,v] = np.uint8( np.array(self.flow_cmap[dir_bin[u_,v_]])*255 )

        mask = mask[:,:,None].repeat(1,1,4).type(torch.bool).numpy()
        arr_img[mask] = arr[ mask]
        pil_img = Image.fromarray(arr_img[:,:,:3],'RGB')
        if method != 'def':
            return np.array(pil_img).astype(np.uint8)
        if jupyter:
            display(pil_img)
        if store:
            pil_img.save(self.p_visu + str(epoch) +
                         '_' + tag + '.png')
        if self.writer is not None:
            img_np = np.array(pil_img).astype(np.uint8)
            self.writer.add_image(
                tag, img_np, global_step=epoch, dataformats='HWC')

    @multiplot
    def plot_translations(self,
                          tag,
                          epoch,
                          img,
                          flow,
                          mask,
                          store=False,
                          jupyter=False,
                          method='def',
                          min_points=50):
        """
        img torch.tensor(h,w,3)
        flow torch.tensor(h,w,2)
        mask torch.tensor(h,w) BOOL
        """
        flow = flow * \
            mask.type(torch.float32)[:, :, None].repeat(1, 1, 2)
        # flow '[+down/up-], [+right/left-]'

        def bin_dir_amplitude(flow):
            amp = torch.norm(flow, p=2, dim=2)
            amp = amp / (torch.max(amp)+1.0e-6)  # normalize the amplitude
            dir_bin = torch.atan2(flow[:, :, 0], flow[:, :, 1])
            nr_bins = 8
            bin_rad = 2 * pi / nr_bins
            dir_bin = torch.round(dir_bin / bin_rad) * bin_rad
            return dir_bin, amp

        rot_bin, amp = bin_dir_amplitude(flow)
        s = 20

        while torch.sum(mask[::s,::s]) < min_points and s > 1:
            s -= 1

        a = 2 if s > 15 else 1
        pil_img = Image.fromarray(img.numpy().astype(np.uint8), 'RGB')
        draw = ImageDraw.Draw(pil_img)
        txt = f"""Horizontal, pos right | neg left:
  max = {torch.max(flow[mask][:,0])}
  min = {torch.min(flow[mask][:,0])}
  mean = {torch.mean(flow[mask][:,0])}
Vertical, pos down | neg up:
  max = {torch.max(flow[mask][:,1])}
  min = {torch.min(flow[mask][:,1])}
  mean = {torch.mean(flow[mask][:,1])}"""
        draw.text((10, 60), txt, fill=(201, 45, 136, 255))
        col = (0, 255, 0)
        grey = (207, 207, 207)
        for u in range(int(flow.shape[0] / s) - 2):
            u = int(u * s)
            for v in range(int(flow.shape[1] / s) - 2):
                v = int(v * s)
                if mask[u, v] == True:
                    du = round(math.cos(rot_bin[u, v])) * s / 2 * amp[u, v]
                    dv = round(math.sin(rot_bin[u, v])) * s / 2 * amp[u, v]
                    try:
                        draw.line([(v, u), (v + dv, u + du)],
                                  fill=col, width=2)
                        draw.ellipse([(v - a, u - a), (v + a, u + a)],
                                     outline=grey, fill=grey, width=2)
                    except:
                        pass
        if method != 'def':
            return np.array(pil_img).astype(np.uint8)
        if jupyter:
            display(pil_img)
        if store:
            pil_img.save(self.p_visu + str(epoch) +
                         '_' + tag + '.png')
        if self.writer is not None:
            img_np = np.array(pil_img).astype(np.uint8)
            self.writer.add_image(
                tag, img_np, global_step=epoch, dataformats='HWC')
    @multiplot
    def plot_contour(self,
                     tag,
                     epoch,
                     img,
                     points,
                     cam_cx=0,
                     cam_cy=0,
                     cam_fx=0,
                     cam_fy=0,
                     trans=[[0, 0, 0]],
                     rot_mat=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                     store=False,
                     jupyter=False,
                     thickness=2,
                     color=(0, 255, 0),
                     method='def'):
        """
        tag := tensorboard tag 
        epoch := tensorboard epoche
        store := ture -> stores the image to standard path
        path := != None creats the path and store to it path/tag.png
        img:= original_image, [widht,height,RGB], torch
        points:= points of the object model [length,x,y,z]
        trans: [1,3]
        rot: [3,3]
        """
        rot_mat = np.array(rot_mat)
        trans = np.array(trans)
        img_f = copy.deepcopy(img).astype(np.uint8)
        points = np.dot(points, rot_mat.T)
        points = np.add(points, trans[0, :])
        h = img_f.shape[0]
        w = img_f.shape[1]
        acc_array = np.zeros((h, w, 1), dtype=np.uint8)

        # project pointcloud onto image
        for i in range(0, points.shape[0]):
            p_x = points[i, 0]
            p_y = points[i, 1]
            p_z = points[i, 2]
            u = int(((p_x / p_z) * cam_fx) + cam_cx)
            v = int(((p_y / p_z) * cam_fy) + cam_cy)
            try:
                a = 10
                acc_array[v - a:v + a + 1, u - a:u + a + 1, 0] = 1
            except:
                pass

        kernel = np.ones((a * 2, a * 2, 1), np.uint8)
        erosion = cv2.erode(acc_array, kernel, iterations=1)

        try:  # problem cause by different cv2 version > 4.0
            contours, hierarchy = cv2.findContours(
                np.expand_dims(erosion, 2), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        except:  # version < 4.0
            _, contours, hierarchy = cv2.findContours(
                np.expand_dims(erosion, 2), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        out = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.drawContours(out, contours, -1, (0, 255, 0), 3)

        for i in range(h):
            for j in range(w):
                if out[i, j, 1] == 255:
                    img_f[i, j, :] = out[i, j, :]

        if method != 'def':
            return img_f.astype(np.uint8)

        if jupyter:
            display(Image.fromarray(img_f))

        if store:
            save_image(img_f, tag=str(epoch) + '_' + tag, p_store=self.p_visu)

        if self.writer is not None:
            self.writer.add_image(tag, img_f.astype(
                np.uint8), global_step=epoch, dataformats='HWC')
    @multiplot
    def plot_segmentation(self, tag, epoch, label, store, method='def', jupyter= False):
        if label.dtype == np.bool:
            col_map = SEG_COLORS_BIN
        else:
            col_map = SEG_COLORS


        if label.dtype == np.float32:
            label = label.round()
        image_out = np.zeros(
            (label.shape[0], label.shape[1], 3), dtype=np.uint8)
        for h in range(label.shape[0]):
            for w in range(label.shape[1]):
                image_out[h, w, :] = col_map[int(label[h, w])][:3]
        
        if method != 'def':
            return image_out.astype(np.uint8)

        if store:
            save_image(
                image_out, tag=f"{epoch}_{tag}", p_store=self.p_visu)
        if self.writer is not None:
            self.writer.add_image(
                tag, image_out, global_step=epoch, dataformats="HWC")
        if jupyter:
            display(Image.fromarray( image_out.astype(np.uint8) ))
            


    @multiplot
    def plot_estimated_pose(self,
                            tag,
                            epoch,
                            img,
                            points,
                            trans=[[0, 0, 0]],
                            rot_mat=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            cam_cx=0, cam_cy=0, cam_fx=0, cam_fy=0,
                            store=False, jupyter=False, w=2, K = None, H=None, method='def'):
        """
        tag := tensorboard tag 
        epoch := tensorboard epoche
        store := ture -> stores the image to standard path
        path := != None creats the path and store to it path/tag.png
        img:= original_image, [widht,height,RGB]
        points:= points of the object model [length,x,y,z]
        trans: [1,3]
        rot: [3,3]
        """
        if K is not None: 
            cam_cx = K [0,2]
            cam_cy = K [1,2] 
            cam_fx = K [0,0]
            cam_fy = K [1,1]
        if H is not None:
            rot_mat = H[:3,:3]
            trans = H[:3,3][None,:]
            if H[3,3] != 1:
                raise Exception
            if H[3,0] != 0 or H[3,1] != 0 or H[3,2] != 0:
                raise Exception

            
        if type(rot_mat) == list:
            rot_mat = np.array(rot_mat)
        if type(trans) == list:
            trans = np.array(trans)

        img_d = copy.deepcopy(img)
        points = np.dot(points, rot_mat.T)
        points = np.add(points, trans[0, :])
        for i in range(0, points.shape[0]):
            p_x = points[i, 0]
            p_y = points[i, 1]
            p_z = points[i, 2]
            u = int(((p_x / np.clip(p_z, a_min= 0.0001, a_max=None)) * cam_fx) + cam_cx)
            v = int(((p_y / np.clip(p_z, a_min= 0.0001,a_max=None)) * cam_fy) + cam_cy)
            try:
                img_d[v - w:v + w + 1, u - w:u + w + 1, 0] = 0
                img_d[v - w:v + w + 1, u - w:u + w + 1, 1] = 255
                img_d[v - w:v + w + 1, u - w:u + w + 1, 0] = 0
            except:
                #print("out of bounce")
                pass
        if method != 'def':
            return img_d.astype(np.uint8)

        if jupyter:
            display(Image.fromarray(img_d.astype(np.uint8)))

        if store:
            #store_ar = (img_d* 255).round().astype(np.uint8)
            #print("IMAGE D:" ,img_d,img_d.shape )
            save_image(img_d, tag=str(epoch) + '_' + tag, p_store=self.p_visu)
        if self.writer is not None:
            self.writer.add_image(tag, img_d.astype(
                np.uint8), global_step=epoch, dataformats='HWC')
    
    @multiplot
    def plot_estimated_pose_on_bb(  self,
                                    tag,
                                    epoch,
                                    img,
                                    points,
                                    tl,
                                    br,
                                    trans=[[0, 0, 0]],
                                    rot_mat=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                    cam_cx=0, cam_cy=0, cam_fx=0, cam_fy=0,
                                    store=False, jupyter=False, w=2, K = None, H=None, method='def'):
        """
        tag := tensorboard tag 
        epoch := tensorboard epoche
        store := ture -> stores the image to standard path
        path := != None creats the path and store to it path/tag.png
        img:= original_image, [widht,height,RGB]
        points:= points of the object model [length,x,y,z]
        trans: [1,3]
        rot: [3,3]
        """
        if K is not None: 
            cam_cx = K [0,2]
            cam_cy = K [1,2] 
            cam_fx = K [0,0]
            cam_fy = K [1,1]
        if H is not None:
            rot_mat = H[:3,:3]
            trans = H[:3,3][None,:]
            if H[3,3] != 1:
                raise Exception
            if H[3,0] != 0 or H[3,1] != 0 or H[3,2] != 0:
                raise Exception

            
        if type(rot_mat) == list:
            rot_mat = np.array(rot_mat)
        if type(trans) == list:
            trans = np.array(trans)

        img_d = copy.deepcopy(img)
        points = np.dot(points, rot_mat.T)
        points = np.add(points, trans[0, :])
        width = int( br[1] - tl[1] )
        height = int( br[0] - tl[0] )
        off_h = int( tl[0] ) 
        off_w = int( tl[1] )
        
        for i in range(0, points.shape[0]):
            p_x = points[i, 0]
            p_y = points[i, 1]
            p_z = points[i, 2]

            u = int( (int(((p_x / p_z) * cam_fx) + cam_cx) - off_w) / width * 640 )
            v = int( (int(((p_y / p_z) * cam_fy) + cam_cy) - off_h) / height * 480 )

            try:
                img_d[v - w:v + w + 1, u - w:u + w + 1, 0] = 0
                img_d[v - w:v + w + 1, u - w:u + w + 1, 1] = 255
                img_d[v - w:v + w + 1, u - w:u + w + 1, 0] = 0
            except:
                #print("out of bounce")
                pass
        if method != 'def':
            return img_d.astype(np.uint8)

        if jupyter:
            display(Image.fromarray(img_d.astype(np.uint8)))

        if store:
            #store_ar = (img_d* 255).round().astype(np.uint8)
            #print("IMAGE D:" ,img_d,img_d.shape )
            save_image(img_d, tag=str(epoch) + '_' + tag, p_store=self.p_visu)
        if self.writer is not None:
            self.writer.add_image(tag, img_d.astype(
                np.uint8), global_step=epoch, dataformats='HWC')

    @multiplot
    def plot_bounding_box(self, tag, epoch, img, rmin=0, rmax=0, cmin=0, cmax=0, str_width=2, store=False, jupyter=False, b=None, method='def'):
        """
        tag := tensorboard tag 
        epoch := tensorboard epoche
        store := ture -> stores the image to standard path
        path := != None creats the path and store to it path/tag.png
        img:= original_image, [widht,height,RGB]

        """

        if isinstance(b, dict):
            rmin = b['rmin']
            rmax = b['rmax']
            cmin = b['cmin']
            cmax = b['cmax']

        # ToDo check Input data
        img_d = np.array(copy.deepcopy(img))

        c = [0, 0, 255]
        rmin_mi = max(0, rmin - str_width)
        rmin_ma = min(img_d.shape[0], rmin + str_width)

        rmax_mi = max(0, rmax - str_width)
        rmax_ma = min(img_d.shape[0], rmax + str_width)

        cmin_mi = max(0, cmin - str_width)
        cmin_ma = min(img_d.shape[1], cmin + str_width)

        cmax_mi = max(0, cmax - str_width)
        cmax_ma = min(img_d.shape[1], cmax + str_width)

        img_d[rmin_mi:rmin_ma, cmin:cmax, :] = c
        img_d[rmax_mi:rmax_ma, cmin:cmax, :] = c
        img_d[rmin:rmax, cmin_mi:cmin_ma, :] = c
        img_d[rmin:rmax, cmax_mi:cmax_ma, :] = c
        print("STORE", store)
        img_d = img_d.astype(np.uint8)
        if method != 'def':
            return img_d.astype(np.uint8)

        if store:
            #store_ar = (img_d* 255).round().astype(np.uint8)
            save_image(img_d, tag=str(epoch) + tag, p_store=self.p_visu)
        if jupyter:
            display(Image.fromarray(np.uint8(img_d) ))
        if self.writer is not None:
            self.writer.add_image(tag, img_d.astype(
                np.uint8), global_step=epoch, dataformats='HWC')
    @multiplot
    def plot_batch_projection(self, tag, epoch,
                              images, target, cam,
                              max_images=10, store=False, jupyter=False, method='def'):

        num = min(max_images, target.shape[0])
        fig = plt.figure(figsize=(7, num * 3.5))
        for i in range(num):
            masked_idx = backproject_points(
                target[i], fx=cam[i, 2], fy=cam[i, 3], cx=cam[i, 0], cy=cam[i, 1])

            for j in range(masked_idx.shape[0]):
                try:
                    images[i, int(masked_idx[j, 0]), int(
                        masked_idx[j, 1]), 0] = 0
                    images[i, int(masked_idx[j, 0]), int(
                        masked_idx[j, 1]), 1] = 255
                    images[i, int(masked_idx[j, 0]), int(
                        masked_idx[j, 1]), 2] = 0
                except:
                    pass

            min1 = torch.min(masked_idx[:, 0])
            max1 = torch.max(masked_idx[:, 0])
            max2 = torch.max(masked_idx[:, 1])
            min2 = torch.min(masked_idx[:, 1])

            bb = BoundingBox(p1=torch.stack(
                [min1, min2]), p2=torch.stack([max1, max2]))

            bb_img = bb.plot(
                images[i, :, :, :3].cpu().numpy().astype(np.uint8))
            fig.add_subplot(num, 2, i * 2 + 1)
            plt.imshow(bb_img)

            fig.add_subplot(num, 2, i * 2 + 2)
            real = images[i, :, :, :3].cpu().numpy().astype(np.uint8)
            plt.imshow(real)
        
        if method != 'def':
            a = get_img_from_fig(fig).astype(np.uint8)
            plt.close()
            return a

        if store:
            #store_ar = (img_d* 255).round().astype(np.uint8)
            plt.savefig(
                f'{self.p_visu}/{str(epoch)}_{tag}_project_batch.png', dpi=300)
            #save_image(img_d, tag=str(epoch) + tag, p_store=self.p_visu)
        if jupyter:
            plt.show()
        if self.writer is not None:
            # you can get a high-resolution image as numpy array!!
            plot_img_np = get_img_from_fig(fig)
            self.writer.add_image(
                tag, plot_img_np, global_step=epoch, dataformats='HWC')
        
    @multiplot
    def visu_network_input(self, tag, epoch, data, max_images=10, store=False, jupyter=False, method='def'):
        num = min(max_images, data.shape[0])
        fig = plt.figure(figsize=(7, num * 3.5))

        for i in range(num):

            n_render = f'batch{i}_render.png'
            n_real = f'batch{i}_real.png'
            real = np.transpose(
                data[i, :3, :, :].cpu().numpy().astype(np.uint8), (1, 2, 0))
            render = np.transpose(
                data[i, 3:, :, :].cpu().numpy().astype(np.uint8), (1, 2, 0))

            # plt_img(real, name=n_real, folder=folder)
            # plt_img(render, name=n_render, folder=folder)

            fig.add_subplot(num, 2, i * 2 + 1)
            plt.imshow(real)
            plt.tight_layout()
            fig.add_subplot(num, 2, i * 2 + 2)
            plt.imshow(render)
            plt.tight_layout()
        
        if method != 'def':
            a = get_img_from_fig(fig).astype(np.uint8)
            plt.close()
            return  a  

        if store:
            #store_ar = (img_d* 255).round().astype(np.uint8)
            plt.savefig(
                f'{self.p_visu}/{str(epoch)}_{tag}_network_input.png', dpi=300)
                
            #save_image(img_d, tag=str(epoch) + tag, p_store=self.p_visu)
        if jupyter:
            plt.show()
        if self.writer is not None:
            # you can get a high-resolution image as numpy array!!
            plot_img_np = get_img_from_fig(fig)
            self.writer.add_image(
                tag, plot_img_np, global_step=epoch, dataformats='HWC')
        plt.close()
    @multiplot

    def plot_corrospondence(self, tag, epoch, u_map, v_map, flow_mask, real_img, render_img, store=False, jupyter=False, coloful = False, method='def', res_h =30, res_w=30, min_points=50):
        """Plot Matching Points on Real and Render Image

        Args:
            tag ([string]): 
            epoch (int): 
            u_map (torch.tensor dtype float): H,W 
            v_map (torch.tensor dtype float): H,W
            flow_mask (torch.tensor dtype bool): H,W
            real_img (torch.tensor dtype float): H,W,3
            render_img (torch.tensor dtype float): H,W,3
        """     
        cropped_comp = np.concatenate( [real_img.cpu().numpy(), render_img.cpu().numpy() ], axis=1).astype(np.uint8)
        cropped_comp_img = Image.fromarray(cropped_comp)
        draw = ImageDraw.Draw(cropped_comp_img)

        m = flow_mask != 0
        txt = f"""Flow in Height:
  max = {torch.max(u_map[m].type(torch.float32))}
  min = {torch.min(u_map[m].type(torch.float32))}
  mean = {torch.mean(u_map[m].type(torch.float32))}
Flow in Vertical:
  max = {torch.max(v_map[m].type(torch.float32))}
  min = {torch.min(v_map[m].type(torch.float32))}
  mean = {torch.mean(v_map[m].type(torch.float32))}"""
        draw.text((10, 60), txt, fill=(201, 45, 136, 255))

        Nc = 20
        cmap = plt.cm.get_cmap('gist_rainbow', Nc)
        cmaplist = [cmap(i) for i in range(cmap.N)]
        

        w = 640
        h = 480
        col = (0,255,0)

        while torch.sum(flow_mask[::res_h,::res_w]) < min_points and res_h > 1:
            res_w -= 1
            res_h -= 1

        for _w in range(0,w,res_w):
            for _h in range(0,h,res_h): 

                if flow_mask[_h,_w] != 0:
                    try:
                        delta_h = u_map[_h,_w]
                        delta_w = v_map[_h,_w]
                        if coloful:
                            col = random.choice(cmaplist)[:3]
                            col = (int( col[0]*255 ),int( col[1]*255 ),int( col[2]*255 ))
                        draw.line([(int(_w), int(_h)), (int(_w + w - delta_w ), int( _h - delta_h))],
                        fill=col, width=2)
                    except:
                        print('failed')
        if method != 'def':
            return np.array( cropped_comp_img ).astype(np.uint8)
        if store:
            cropped_comp_img.save(f'{self.p_visu}/{str(epoch)}_{tag}_corrospondence.png')
        if jupyter:
             display(cropped_comp_img)
        if self.writer is not None:
            plot_img_np = np.array( cropped_comp_img )
            self.writer.add_image(
                tag, plot_img_np, global_step=epoch, dataformats='HWC')
       
    @multiplot
    def visu_network_input_pred(self, tag, epoch, data, images, target, cam, max_images=10, store=False, jupyter=False, method='def'):
        num = min(max_images, data.shape[0])
        fig = plt.figure(figsize=(10.5, num * 3.5))

        for i in range(num):
            # real render input
            n_render = f'batch{i}_render.png'
            n_real = f'batch{i}_real.png'
            real = np.transpose(
                data[i, :3, :, :].cpu().numpy().astype(np.uint8), (1, 2, 0))
            render = np.transpose(
                data[i, 3:, :, :].cpu().numpy().astype(np.uint8), (1, 2, 0))
            fig.add_subplot(num, 3, i * 3 + 1)
            plt.imshow(real)
            plt.tight_layout()
            fig.add_subplot(num, 3, i * 3 + 2)
            plt.imshow(render)
            plt.tight_layout()

            # prediction
            masked_idx = backproject_points(
                target[i], fx=cam[i, 2], fy=cam[i, 3], cx=cam[i, 0], cy=cam[i, 1])
            for j in range(masked_idx.shape[0]):
                try:
                    images[i, int(masked_idx[j, 0]), int(
                        masked_idx[j, 1]), 0] = 0
                    images[i, int(masked_idx[j, 0]), int(
                        masked_idx[j, 1]), 1] = 255
                    images[i, int(masked_idx[j, 0]), int(
                        masked_idx[j, 1]), 2] = 0
                except:
                    pass
            min1 = torch.min(masked_idx[:, 0])
            max1 = torch.max(masked_idx[:, 0])
            max2 = torch.max(masked_idx[:, 1])
            min2 = torch.min(masked_idx[:, 1])
            bb = BoundingBox(p1=torch.stack(
                [min1, min2]), p2=torch.stack([max1, max2]))
            bb_img = bb.plot(
                images[i, :, :, :3].cpu().numpy().astype(np.uint8))
            fig.add_subplot(num, 3, i * 3 + 3)
            plt.imshow(bb_img)
            # fig.add_subplot(num, 2, i * 2 + 4)
            # real = images[i, :, :, :3].cpu().numpy().astype(np.uint8)
            # plt.imshow(real)
        if method != 'def':
            a = get_img_from_fig(fig).astype(np.uint8)
            plt.close()
            return a

        if store:
            #store_ar = (img_d* 255).round().astype(np.uint8)
            plt.savefig(
                f'{self.p_visu}/{str(epoch)}_{tag}_network_input_and_prediction.png', dpi=300)
            #save_image(img_d, tag=str(epoch) + tag, p_store=self.p_visu)
        if jupyter:
            plt.show()
        if self.writer is not None:
            # you can get a high-resolution image as numpy array!!
            plot_img_np = get_img_from_fig(fig)
            self.writer.add_image(
                tag, plot_img_np, global_step=epoch, dataformats='HWC')
        plt.close()


def plot_pcd(x, point_size=0.005, c='g'):
    """[summary]

    Args:
        x ([type]): point_nr,3
        point_size (float, optional): [description]. Defaults to 0.005.
        c (str, optional): [description]. Defaults to 'g'.
    """    
    if c == 'b':
        k = 245
    elif c == 'g':
        k = 25811000
    elif c == 'r':
        k = 11801000
    elif c == 'black':
        k = 2580
    else:
        k = 2580
    colors = np.ones(x.shape[0]) * k
    plot = k3d.plot(name='points')
    plt_points = k3d.points(x, colors.astype(np.uint32), point_size=point_size)
    plot += plt_points
    plt_points.shader = '3d'
    plot.display()


def plot_two_pcd(x, y, point_size=0.005, c1='g', c2='r'):
    if c1 == 'b':
        k = 245
    elif c1 == 'g':
        k = 25811000
    elif c1 == 'r':
        k = 11801000
    elif c1 == 'black':
        k = 2580
    else:
        k = 2580

    if c2 == 'b':
        k2 = 245
    elif c2 == 'g':
        k2 = 25811000
    elif c2 == 'r':
        k2 = 11801000
    elif c2 == 'black':
        k2 = 2580
    else:
        k2 = 2580

    col1 = np.ones(x.shape[0]) * k
    col2 = np.ones(y.shape[0]) * k2
    plot = k3d.plot(name='points')
    plt_points = k3d.points(x, col1.astype(np.uint32), point_size=point_size)
    plot += plt_points
    plt_points = k3d.points(y, col2.astype(np.uint32), point_size=point_size)
    plot += plt_points
    plt_points.shader = '3d'
    plot.display()


class SequenceVisualizer():
    def __init__(self, seq_data, images_path, output_path=None):
        self.seq_data = seq_data
        self.images_path = images_path
        self.output_path = output_path

    def plot_points_on_image(self, seq_no, frame_no, jupyter=False, store=False, pose_type='filtered'):
        seq_data = self.seq_data
        images_path = self.images_path
        output_path = self.output_path
        frame = seq_data[seq_no][frame_no]
        unique_desig = frame['dl_dict']['unique_desig'][0]

        if pose_type == 'ground_truth':
            # ground truth
            t = frame['dl_dict']['gt_trans'].reshape(1, 3)
            rot_quat = re_quat(copy.deepcopy(
                frame['dl_dict']['gt_rot_wxyz'][0]), 'wxyz')
            rot = R.from_quat(rot_quat).as_matrix()
        elif pose_type == 'filtered':
            # filter pred
            t = np.array(frame['filter_pred']['t']).reshape(1, 3)
            rot_quat = re_quat(copy.deepcopy(
                frame['filter_pred']['r_wxyz']), 'wxyz')
            rot = R.from_quat(rot_quat).as_matrix()
        elif pose_type == 'final_pred_obs':
            # final pred
            t = np.array(frame['final_pred_obs']['t']).reshape(1, 3)
            rot_quat = re_quat(copy.deepcopy(
                frame['final_pred_obs']['r_wxyz']), 'wxyz')
            rot = R.from_quat(rot_quat).as_matrix()
        else:
            raise Exception('Pose type not implemented.')

        w = 2
        if type(unique_desig) != str:
            im = np.array(Image.open(
                images_path + unique_desig[0] + '-color.png'))  # ycb
        else:
            im = np.array(Image.open(
                images_path + unique_desig + '.png'))  # laval
        img_d = copy.deepcopy(im)

        dl_dict = frame['dl_dict']
        points = copy.deepcopy(
            seq_data[seq_no][0]['dl_dict']['model_points'][0, :, :])
        points = np.dot(points, rot.T)
        points = np.add(points, t[0, :])

        cam_cx = dl_dict['cam_cal'][0][0]
        cam_cy = dl_dict['cam_cal'][0][1]
        cam_fx = dl_dict['cam_cal'][0][2]
        cam_fy = dl_dict['cam_cal'][0][3]
        for i in range(0, points.shape[0]):
            p_x = points[i, 0]
            p_y = points[i, 1]
            p_z = points[i, 2]
            u = int(((p_x / p_z) * cam_fx) + cam_cx)
            v = int(((p_y / p_z) * cam_fy) + cam_cy)
            try:
                img_d[v - w:v + w + 1, u - w:u + w + 1, 0] = 0
                img_d[v - w:v + w + 1, u - w:u + w + 1, 1] = 255
                img_d[v - w:v + w + 1, u - w:u + w + 1, 0] = 0
            except:
                #print("out of bounds")
                pass

        img_disp = Image.fromarray(img_d)
        if jupyter:
            display(img_disp)

        if store:
            outpath = output_path + \
                '{}_{}_{}.png'.format(pose_type, seq_no, frame_no)
            img_disp.save(outpath, "PNG", compress_level=1)
            print("Saved image to {}".format(outpath))

    def save_sequence(self, seq_no, pose_type='filtered', name=''):
        for fn in range(len(self.seq_data)):
            self.plot_points_on_image(seq_no, fn, False, True, pose_type)
        if name:
            video_name = '{}_{}_{}'.format(name, pose_type, seq_no)
        else:
            video_name = '{}_{}'.format(pose_type, seq_no)
        cmd = "cd {} && ffmpeg -r 10 -i ./filtered_{}_%d.png -vcodec mpeg4 -y {}.mp4".format(
            self.output_path, seq_no, video_name)
        os.system(cmd)


def load_sample_dict():
    # load data
    os.chdir('/home/jonfrey/PLR2')
    sys.path.append('src')
    sys.path.append('src/dense_fusion')

    from loaders_v2 import ConfigLoader, GenericDataset

    exp_cfg = ConfigLoader().from_file('/home/jonfrey/PLR2/yaml/exp/exp_ws_deepim.yml')
    env_cfg = ConfigLoader().from_file(
        '/home/jonfrey/PLR2/yaml/env/env_natrix_jonas.yml')
    generic = GenericDataset(
        cfg_d=exp_cfg['d_train'],
        cfg_env=env_cfg)
    img = Image.open(
        '/media/scratch1/jonfrey/datasets/YCB_Video_Dataset/data/0000/000001-color.png')
    out = generic[0]
    generic.visu = True
    names = ['cloud', 'choose', 'img_masked', 'target', 'model_points',
             'idx', 'add_depth', 'add_mask', 'img', 'cam', 'rot', 'trans', 'desig']

    sample = {}
    print(len(out[0]))
    for i, o in enumerate(out[0]):
        sample[names[i]] = o

    return sample


if __name__ == "__main__":

    sample = load_sample_dict()

    p = "/home/jonfrey/tmp"
    vis = Visualizer(p_visu=p, writer=None)
    vis.plot_contour(tag="visu_contour_test",
                     epoch=0,
                     img=sample['img'],
                     points=sample['target'],
                     cam_cx=sample['cam'][0],
                     cam_cy=sample['cam'][1],
                     cam_fx=sample['cam'][2],
                     cam_fy=sample['cam'][3],
                     store=True)

    vis.plot_estimated_pose(tag="visu_estimated_test",
                            epoch=0,
                            img=sample['img'],
                            points=sample['target'],
                            cam_cx=sample['cam'][0],
                            cam_cy=sample['cam'][1],
                            cam_fx=sample['cam'][2],
                            cam_fy=sample['cam'][3],
                            store=True)

    images = sample['img']
    images = images.unsqueeze(0)
    images = images.repeat(10, 1, 1, 1)

    target = sample['target']
    target = target.unsqueeze(0)
    target = target.repeat(10, 1, 1)

    cam = sample['cam']
    cam = cam.unsqueeze(0)
    cam = cam.repeat(10, 1)

    vis.plot_batch_projection(tag='batch_projection', epoch=0,
                              images=images, target=target, cam=cam,
                              max_images=10, store=True, jupyter=False)
    images = torch.transpose(images, 1, 3)
    images = torch.transpose(images, 2, 3)
    data = torch.cat([images, images], dim=1)

    vis.visu_network_input(tag="network_input",
                           epoch=0, data=data,
                           max_images=10,
                           store=True,
                           jupyter=False)

