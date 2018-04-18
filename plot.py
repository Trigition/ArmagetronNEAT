#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('arrays', help='Numpy array files to render', nargs='+')
    return parser.parse_args()


args = get_arguments()
master_array = [np.load(filename) for filename in args.arrays]
master_array = np.concatenate(master_array)

fig = plt.figure()
im = plt.imshow(master_array[0], animated=True)

def animate(i):
    im.set_array(master_array[i])
    return im,


anim = animation.FuncAnimation(fig, animate, blit=True, interval=60, frames=master_array.shape[0] - 1)

plt.show()
