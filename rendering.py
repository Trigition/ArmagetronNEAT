#!/usr/bin/env python
# -*- coding: utf-8 -*-

from queue import Queue
from threading import Thread

import numpy as np
import os

from PIL import Image
from scipy.ndimage import zoom


class Renderer(Thread):

    """The Renderer Class saves and renders simulation
    and agent sensor numpy matrices"""

    def __init__(self, buffer_len=1024, img_dir='images'):
        """Initializes a renderer

        :buffer_len: How many jobs can be queued at once before
        the Renderer will block any more inputs

        """
        Thread.__init__(self)

        self.img_dir = img_dir

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        self.buffer = Queue(buffer_len)
        self.daemon = True
        self.start()

    def run(self):
        """Runs the Renderer Thread

        """
        while True:
            job = self.buffer.get()
            filename = '%s/%s.jpg' % (self.img_dir, job['filename'])
            img = insert_lines(job['matrix'], job['scale'], thickness=3)
            img = Image.fromarray(img)
            img.convert('RGB').save(filename)
            self.buffer.task_done()

    def wait_till_done(self):
        self.queue.join()


def insert_lines(a, scale, value=128, thickness=1):
    """Inserts grid lines into an array so individual
    spaces are more easily seen.
    :a: The 2D array
    :scale: The amount the array has been scaled
    :value: The value to place
    :thickness: The number of rows/columns to insert
    :returns: The modified array

    """
    a = np.copy(a)
    a[a > 0] = 255
    
    if scale > 1:
        a = zoom(a, scale, order=0)

    width, height = a.shape

    # Typecheck dimensions with scale
    if width % scale != 0:
        print('Warning: Width %d is not a multiple of specified \
              scale %d. Borders may not be rendered' %
              (width, scale))

    if height % scale != 0:
        print('Warning: Height %d is not a multiple of specified \
              scale %d. Borders may not be rendered' %
              (height, scale))

    # Strange off by one errors occure and the grid becomes misaligned
    # on coordinates far from the origin

    #a = np.insert(a, range(0, width+1, scale), value, axis=0)
    #a = np.insert(a, range(0, height+1, scale), value, axis=1)

    return a
