#!/usr/bin/env python
# -*- coding: utf-8 -*-

from armagetron import Simulation

max_agents = 50

if __name__ == '__main__':
    i = 0
    #grid = Grid(100, 100, max_agents)
    #grid.simulate(1000)
    sim = Simulation(1000, 50, n_threads=4)
    sim.simulate()
