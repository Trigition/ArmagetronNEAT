#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def sigmoid(x):
    """Sigmoid activation function

    :x: The input value
    :returns: The value passed through
    the sigmoid activation function

    """
    return 1 / (1+np.exp(-x))
