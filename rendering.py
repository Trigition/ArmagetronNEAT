#!/usr/bin/env python
# -*- coding: utf-8 -*-

from queue import Queue
from threading import Thread

import numpy as np
from PIL import Image
from scipy.ndimage import zoom


class Renderer(Thread):

    """The Renderer Class saves and renders simulation
    and agent sensor numpy matrices"""

    def __init__(self, buffer_len=1024):
        """Initializes a renderer

        :buffer_len: How many jobs can be queued at once before
        the Renderer will block any more inputs

        """
        Thread.__init__(self)

        self.buffer = Queue(buffer_len)
        self.daemon = True
        self.start()

    def run(self):
        """Runs the Renderer Thread

        """
        while True:
            job = self.buffer.get()
            filename = '%s.jpg' % job['filename']
            img = insert_lines(job['matrix'], job['scale'], thickness=3)
            img = Image.fromarray(img)
            img.convert('RGB').save(filename)
            self.buffer.task_done()

    def wait_till_done(self):
        self.queue.join()


def insert_lines(a, scale, value=0, thickness=1):
    """Inserts grid lines into an array so individual
    spaces are more easily seen.
    :a: The 2D array
    :scale: The amount the array has been scaled
    :value: The value to place
    :thickness: The number of rows/columns to insert
    :returns: The modified array

    """
    new_a = np.copy(a)
    new_a[new_a > 0] = 255
    if scale > 1:
        new_a = zoom(new_a, scale, order=0)

    width, height = new_a.shape
    # Typecheck dimensions with scale
    if width % scale != 0:
        print('Warning: Width %d is not a multiple of specified \
              scale %d. Borders may not be rendered' %
              (width, scale))
    if height % scale != 0:
        print('Warning: Height %d is not a multiple of specified \
              scale %d. Borders may not be rendered' %
              (height, scale))

    # Insert
    for i in reversed(range(0, width, scale)):
        new_a = np.insert(new_a, i, value, axis=0)
    for i in reversed(range(0, height, scale)):
        new_a = np.insert(new_a, i, value, axis=1)

    return new_a
