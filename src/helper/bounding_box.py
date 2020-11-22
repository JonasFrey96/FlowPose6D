import numpy as np
import matplotlib.pyplot as plt
import copy
import torch

from .camera import backproject_points_batch, backproject_points, backproject_point


class BoundingBox():
    def __init__(self, p1, p2):
        "p1 = u,v  u=height v=widht starting top_left 0,0"
        if p1[0] < p2[0] and p1[1] < p2[1]:
            # print("p1 = top_left")
            self.tl = p1
            self.br = p2
        elif p1[0] > p2[0] and p1[1] > p2[1]:
            # print("p1 = bottom_right")
            self.br = p1
            self.tl = p2
        elif p1[0] > p2[0] and p1[1] < p2[1]:
            # print("p1 = bottom_left")
            self.tl = copy.copy(p1)
            self.tl[0] = p2[0]
            self.br = p2
            self.br[0] = p1[0]
        else:
            # print("p1 = top_right")
            self.br = copy.copy(p1)
            self.br[0] = p2[0]
            self.tl = p2
            self.tl[0] = p1[0]

    def __str__(self):
        w = self.width()
        h = self.height()
        return f'TL Cor: {self.tl}, BR Cor: {self.br}, Widht: {w}, Height: {h}'

    def height(self):
        return (self.br[0] - self.tl[0])

    def width(self):
        return (self.br[1] - self.tl[1])

    def check_min_size(self, min_h=40, min_w=40):
        if 0 < self.height() < min_h or 0 < self.width() < min_w:
            return False
        else:
            return True

    def violation(self):
        return bool(torch.isinf(self.br[0]) or torch.isinf(self.br[1]) or torch.isinf(self.tl[0]) or torch.isinf(self.tl[1]))

    def move(self, u, v):
        self.br[0] += u
        self.tl[0] += u
        self.br[1] += v
        self.tl[1] += v

    def expand(self, r):
        r = r - 1
        self.br[0] = int(self.br[0] + self.height() * r)
        self.tl[0] = int(self.tl[0] - self.height() * r)
        self.br[1] = int(self.br[1] + self.height() * r)
        self.tl[1] = int(self.tl[1] - self.height() * r)

    def add_margin(self, u, v):
        self.br[0] += u
        self.tl[0] -= u
        self.br[1] += v
        self.tl[1] -= v

    def set_max(self, max_height=480, max_width=640):
        self.tl[0] = 0
        self.tl[1] = 0
        self.br[0] = max_height
        self.br[1] = max_width

    def limit_bb(self, max_height=480, max_width=640, store=False):
        if store:
            if self.tl[0] < 0:
                self.tl[0] = 0
            elif self.tl[0] > max_height:
                self.tl[0] = max_height

            if self.br[0] < 0:
                self.br[0] = 0
            elif self.br[0] > max_height:
                self.br[0] = max_height

            if self.tl[1] < 0:
                self.tl[1] = 0
            elif self.tl[1] > max_width:
                self.tl[1] = max_width
            if self.br[1] < 0:
                self.br[1] = 0
            elif self.br[1] > max_width:
                self.br[1] = max_width
        else:
            br = self.br.clone()
            tl = self.tl.clone()
            if self.tl[0] < 0:
                tl[0] = 0
            elif self.tl[0] > max_height:
                tl[0] = max_height

            if self.br[0] < 0:
                br[0] = 0
            elif self.br[0] > max_height:
                br[0] = max_height

            if self.tl[1] < 0:
                tl[1] = 0
            elif self.tl[1] > max_width:
                tl[1] = max_width
            if self.br[1] < 0:
                br[1] = 0
            elif self.br[1] > max_width:
                br[1] = max_width
            return tl, br

    def crop(self, img, scale=False, mode='nearest', max_height=480, max_width=640):
        """
            img: torch.tensor H,W,C
            scale: bool return the Image scaled up to H=480 W=640
            mode: nearest, bilinear
        """
        if self.valid():
            res = img[int(self.tl[0]):int(self.br[0]), int(self.tl[1]):int(self.br[1]), :]
        else:
            h = img.shape[0]
            w = img.shape[1]
            img_pad = torch.zeros((int(h + 2*max_height), int(w + 2*max_width), img.shape[2]), dtype=img.dtype, device=img.device)
            
            img_pad[max_height:max_height+h,max_width:max_width+w] = img
            res = img_pad[ int(max_height+self.tl[0]) : int(max_height+self.br[0]), int(max_width+self.tl[1]) : int(max_width+self.br[1])] # H W C
        if scale:
            res = res.permute(2,0,1)[None] #BS C H W
            if mode == 'bilinear':
                res  = torch.nn.functional.interpolate(res, size=(480,640), mode=mode, align_corners=False)  
            else:
                res  = torch.nn.functional.interpolate(res, size=(480,640), mode=mode)  
            res = res[0].permute(1,2,0) # H W C
            
        return res

    def add_noise(self, std_h, std_w):
        # std_h is the variance that is added to the top corrner position and, bottom_corner position
        self.br = np.random.normal(self.br, np.array(
            [std_h, std_w])).astype(dtype=np.int32)
        self.tl = np.random.normal(self.tl, np.array(
            [std_h, std_w])).astype(dtype=np.int32)

    def valid(self, w=640, h=480):
        return self.tl[0] >= 0 and self.tl[1] >= 0 and self.br[0] <= h and self.br[1] <= w

    def expand_to_correct_ratio(self, w, h):
        if self.width() / self.height() > w / h:
            scale_ratio = h / self.height()
            h_set = self.width() * (h / w)
            add_w = 0
            add_h = int((h_set - self.height()) / 2)
        else:
            scale_ratio = h / self.height()
            w_set = self.height() * (w / h)
            add_h = 0
            add_w = int((w_set - self.width()) / 2)

        self.add_margin(add_h, add_w)

    def plot(self, img, w=5, ret_array=True, debug_plot=False):
        test = copy.deepcopy(img)

        if self.valid():
            c = [0, 255, 0]
        else:
            c = [255, 0, 0]

        _tl, _br = self.limit_bb()
        w = 5
        test[int(_tl[0]):int(_br[0]), int(_tl[1]) -
             w: int(_tl[1]) + w] = c
        test[int(_tl[0]):int(_br[0]), int(_br[1]) -
             w: int(_br[1]) + w] = c

        test[int(_tl[0]) - w:int(_tl[0]) + w,
             int(_tl[1]): int(_br[1])] = c
        test[int(_br[0]) - w:int(_br[0]) + w,
             int(_tl[1]): int(_br[1])] = c

        if ret_array:
            return test

        if debug_plot:
            fig = plt.figure()
            fig.add_subplot(1, 1, 1)
            plt.imshow(test)

            plt.axis("off")
            plt.savefig('/home/jonfrey/Debug/test.png')
            plt.show()


def get_bb_from_depth(depth):
    bb_lsd = []
    for d in depth:
        masked_idx = torch.nonzero( d != 0, as_tuple=False)
        min1 = torch.min(masked_idx[:, 0]).type(torch.float32)
        max1 = torch.max(masked_idx[:, 0]).type(torch.float32)
        min2 = torch.min(masked_idx[:, 1]).type(torch.float32)
        max2 = torch.max(masked_idx[:, 1]).type(torch.float32)
        bb_lsd.append(BoundingBox(p1=torch.stack(
            [min1, min2]), p2=torch.stack([max1, max2])))
    return bb_lsd


def get_bb_real_target(target, K):
    bb_ls = []
    for i in range(target.shape[0]):
        # could not find a smart alternative to avoide looping
        masked_idx = backproject_points(
            target[i], K=K[i])
        min1 = torch.min(masked_idx[:, 0])
        max1 = torch.max(masked_idx[:, 0])
        max2 = torch.max(masked_idx[:, 1])
        min2 = torch.min(masked_idx[:, 1])

        bb = BoundingBox(p1=torch.stack(
            [min1, min2]), p2=torch.stack([max1, max2]))

        bb_ls.append(bb)

    return bb_ls
