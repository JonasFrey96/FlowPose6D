# move this to seperate file
import matplotlib.pyplot as plt
import numpy as np
import torch


def plt_img(img, name='plt_img.png', folder='/home/jonfrey/Debug'):

    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(folder + '/' + name)


def plt_torch(data, name='torch.png', folder='/home/jonfrey/Debug'):
    img = np.transpose(
        data[:, :, :].cpu().numpy().astype(np.uint8), (2, 1, 0))
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(folder + '/' + name)
