#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def sigmoid(x):
    """Sigmoid activation function

    :x: TODO
    :returns: TODO

    """
    return 1 / (1+np.exp(-x))
