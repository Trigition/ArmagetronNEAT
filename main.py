#!/usr/bin/env python
# -*- coding: utf-8 -*-

from armagetron import Simulation

max_agents = 50

if __name__ == '__main__':
    i = 0
    sim = Simulation(1000, 50, n_threads=8)
    sim.simulate()
