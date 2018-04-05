#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import networkx as nx

from neat import NEAT_Pool, NEAT_Network

def test_pool_creation():
    pool = NEAT_Pool(None, (5,5), 3)

def test_network_creation():
    pool = NEAT_Pool(None, (5,5), 3)
    net = NEAT_Network(pool.starting_genome, pool)

    data = np.random.rand(5,5).flatten()

    results = net.feedforward(data)
    print(results)
