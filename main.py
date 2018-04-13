#!/usr/bin/env python
# -*- coding: utf-8 -*-

from armagetron import Simulation

max_agents = 50

if __name__ == '__main__':
    sim = Simulation(2000, 50, n_threads=16, sensor_radius=1)
    sim.simulate()
