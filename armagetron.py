#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from agent import Agent
from neat import NEAT_Pool


class Grid():

    """A grid of pixels and agents"""

    def __init__(self, width, height, num_agents, sensor_radius=5):
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
        self.num_agents = num_agents

        # Create grid
        self.grid = np.zeros((width, height), dtype=np.uint8)

        # Create pool
        dims = (sensor_radius+1, sensor_radius+1)
        self.pool = NEAT_Pool(self.grid, dims, 3)

        self.active_agents = []
        self.my_agents = []

        # self.register_agents([Agent(i + 1, self) for i in range(num_agents)])
        agents = []
        for i in range(num_agents):
            agents.append(Agent(i+1, self, self.pool))
        
        self.register_agents(agents)

        
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

    def simulate(self, num_epochs=5):
        print('Simulating')
        iterations = 0
        while len(self.active_agents) > 0:
            self.step()
            self.render_grid(4).save('%08d.jpg' % iterations)
            iterations += 1

        print('Simulation finished after %d steps' % iterations)
        print('Top Agents:')
        sorted_agents = sorted(self.my_agents, key=lambda x: x.lifetime, reverse=True)
        sorted_agents = sorted_agents[:len(sorted_agents) // 10] # Take top %10
        for agent in sorted_agents:
            print('\t%s: %d' % (str(agent), agent.lifetime))


        n_top = len(sorted_agents)
        new_agents = []
        for i in range(self.num_agents):
            parent1 = sorted_agents[i // n_top]
            parent2 = sorted_agents[(i+1) % n_top]
            offspring = parent1 + parent2

        #return sorted_agents
