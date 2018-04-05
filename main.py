#!/usr/bin/env python
# -*- coding: utf-8 -*-

from armagetron import Grid

max_agents = 3

if __name__ == '__main__':
    i=0
    grid = Grid(100,100, max_agents)
    # results = grid.simulate(epoch=i)
    # while True:
    #     try:
    #         i += 1
    #         grid = Grid(100,100, max_agents, results)
    #         results = grid.simulate(epoch=i)
    #     except KeyboardInterrupt:
    #         print('Goodbye')
    #         break
