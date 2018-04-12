#!/usr/bin/env python
# -*- coding: utf-8 -*-

def feature_scaling(array_like, a=0, b=1, inplace=False):
    """Scales the array values between min_val and
    max_val.

    :array_like: The array that is being scaled
    :a: The minimum value to scale to
    :b: The maximum value to scale to
    :inplace: If True, the array_like is modified. If False
    then a scaled array is returned.
    :returns: array_like but scaled

    """
    if a >= b:
        raise ValueError('Error: Feature scaling minimum \
                value %d is higher than %d' %
                (a, b))

    low = min(array_like)
    high= max(array_like)
    
    if inplace:
        # If in place, modify array_like
        for x in array_like:
            x = scale(x, low, high, a, b)
        return array_like
    else:
        # If not in place, return new array
        return [scale(x, low, high, a, b) for x in array_like]

def scale(x, min_val, max_val, a, b):
    """Scales an individual value between a and b

    :x: The value to scale
    :min_val: The minimum value that member x belongs to
    :max_val: The maximum value that member x belongs to
    :a: The minimum value in the desired scale
    :b: The maximum value in the desired scale
    :returns: The value x scaled between a and b

    """
    diff = min_val - max_val
    return a + ((x - min_val) * (b - a) / diff)
