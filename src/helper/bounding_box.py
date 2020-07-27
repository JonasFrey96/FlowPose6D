import numpy as np
import matplotlib.pyplot as plt
import copy
import torch


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

    def width(self):

        return (self.br[1] - self.tl[1])

    def height(self):
        return (self.br[0] - self.tl[0])

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

    def crop(self, img):
        if self.valid():
            return img[int(self.tl[0]):int(self.br[0]), int(self.tl[1]):int(self.br[1]), :]
        else:
            # to find way not to create a new tensor with doube the size (maybe just translate the original image and then pad with 0)
            h = img.shape[0]
            w = img.shape[1]
            img_pad = torch.zeros((int(h * 3), int(w * 3), img.shape[2]))

            off_h = int(h)
            off_w = int(w)
            img_pad[:, :, 0] = 0
            img_pad[:, :, 1] = 255
            img_pad[:, :, 2] = 0
            _tl, _br = self.limit_bb()
            img_pad[off_h + int(_tl[0]): off_h + int(_br[0]), off_w + int(_tl[1]):off_w + int(
                _br[1]), :] = img[int(_tl[0]):int(_br[0]), int(_tl[1]):int(_br[1]), :]

            return img_pad[off_h + int(self.tl[0]):off_h + int(self.br[0]), off_w + int(self.tl[1]):off_w + int(self.br[1]), :]

    def add_noise(self, std_h, std_w):
        # std_h is the variance that is added to the top corrner position and, bottom_corner position
        self.br = np.random.normal(self.br, np.array(
            [std_h, std_w])).astype(dtype=np.int32)
        self.tl = np.random.normal(self.tl, np.array(
            [std_h, std_w])).astype(dtype=np.int32)

    def valid(self, w=640, h=480):
        return self.tl[0] >= 0 and self.tl[1] >= 0 and self.br[0] < h and self.br[1] < w

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
        w = 5
        test[int(self.tl[0]):int(self.br[0]), int(self.tl[1]) -
             w: int(self.tl[1]) + w] = [0, 255, 0]
        test[int(self.tl[0]):int(self.br[0]), int(self.br[1]) -
             w: int(self.br[1]) + w] = [0, 255, 0]

        test[int(self.tl[0]) - w:int(self.tl[0]) + w,
             int(self.tl[1]): int(self.br[1])] = [0, 255, 0]
        test[int(self.br[0]) - w:int(self.br[0]) + w,
             int(self.tl[1]): int(self.br[1])] = [0, 255, 0]

        if ret_array:
            return test

        if debug_plot:
            fig = plt.figure()
            fig.add_subplot(1, 1, 1)
            plt.imshow(test)

            plt.axis("off")
            plt.savefig('/home/jonfrey/Debug/test.png')
            plt.show()
