#!/usr/bin/env python
# -*- coding: utf-8 -*-

from armagetron import Grid

max_agents = 50

if __name__ == '__main__':
    i = 0
    grid = Grid(100, 100, max_agents)
    grid.simulate(1000)
