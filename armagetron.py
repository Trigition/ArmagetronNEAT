#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from agent import Agent


class Grid():

    """A grid of pixels and agents"""

    def __init__(self, width, height, num_agents, seeding_agents=None):
        """Initializes the grid with a specific width
        and height

        :width: The width of the grid
        :height: The height of the grid
        :num_agents: The number of agents to simulate

        """
        if type(width) is not int or type(height) is not int:
            raise TypeError('Grid dimensions must be integers')
        if width < 1 or height < 1:
            raise ValueError('Grid dimensions must be positive values')

        self.width = width
        self.height = height

        # Create grid
        self.grid = np.zeros((width, height), dtype=np.uint8)

        self.active_agents = []
        self.my_agents = []

        if seeding_agents is None:
            self.register_agents([Agent(i + 1, self) for i in range(num_agents)])
        else:
            gen_agents = 0
            print('Performing crossover')
            while gen_agents <= num_agents:
                src1_i = (gen_agents // len(seeding_agents) % len(seeding_agents))
                src2_i = (gen_agents + 1) % len(seeding_agents)
                result = seeding_agents[src1_i] + seeding_agents[src2_i]

                result.grid = self
                gen_agents += 1
                self.active_agents.append(result)
                self.my_agents.append(result)
                self.randomly_place_agent(result)

    def register_agents(self, agents):
        for agent in agents:
            self.active_agents.append(agent)
            self.my_agents.append(agent)
            self.randomly_place_agent(agent)

    def randomly_place_agent(self, agent):
        pos_x = np.random.randint(0, self.width - 1, dtype=np.uint8)
        pos_y = np.random.randint(0, self.height - 1, dtype=np.uint8)
        agent.set_pos(pos_x, pos_y)
        agent.set_orientation(np.random.randint(0, 4, dtype=np.uint8))

    def step(self):
        for agent in self.active_agents:
            # Determine if any agents are now in walls/out of bounds
            if agent.x not in range(self.width - 1) or \
               agent.y not in range(self.height - 1):
                # Out of bounds
                self.active_agents.remove(agent)
            elif self.grid[agent.x][agent.y] != 0:
                # Collision into wall
                self.active_agents.remove(agent)
            # make a step
            self.grid[agent.x][agent.y] = agent.agent_id
            agent.step()

    def render_grid(self, scale=1):
        array = np.copy(self.grid)
        array[array > 0] = 255
        
        if scale > 0:
            array = zoom(array, scale, order=0)

        return Image.fromarray(array)

    def simulate(self, epoch=0):
        print('Simulating')
        iterations = 0
        while len(self.active_agents) > 0:
            self.step()
            #self.render_grid(4).save('%d-%08d.jpg' % (epoch, iterations))
            iterations += 1

        print('Simulation finished after %d steps' % iterations)
        print('Top Agents:')
        sorted_agents = sorted(self.my_agents, key=lambda x: x.lifetime, reverse=True)
        sorted_agents = sorted_agents[:len(sorted_agents) // 10] # Take top %10
        for agent in sorted_agents:
            print('\t%s: %d' % (str(agent), agent.lifetime))

        return sorted_agents
