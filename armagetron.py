#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
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
        self.__reset_grid__()

        # Create pool
        dims = (sensor_radius+1, sensor_radius+1)
        self.pool = NEAT_Pool(self.grid, dims, 3)

        self.active_agents = []
        self.my_agents = []

        agents = []
        for i in range(num_agents):
            agents.append(Agent(i+1, self, self.pool))
        
        self.register_agents(agents)
        self.global_iter = 0

        
    def register_agents(self, agents):
        for agent in agents:
            self.active_agents.append(agent)
            self.my_agents.append(agent)
            self.randomly_place_agent(agent)

    def randomly_place_agent(self, agent):
        pos_x = np.random.randint(0, self.width, dtype=np.uint8)
        pos_y = np.random.randint(0, self.height, dtype=np.uint8)
        agent.set_pos(pos_x, pos_y)
        agent.set_orientation(np.random.randint(0, 4, dtype=np.uint8))

    def step(self):
        for agent in self.active_agents:
            # Determine if any agents are now in walls/out of bounds
            if self.is_out_of_bounds(agent):
                # Out of bounds
                self.active_agents.remove(agent)
                continue
            elif self.grid[agent.x][agent.y] != 0:
                # Collision into wall
                self.active_agents.remove(agent)
                continue
            # make a step
            self.grid[agent.x][agent.y] = agent.agent_id
            agent.step()

    def is_out_of_bounds(self, agent):
        """Checks to see if an agent is out of bounds

        :agent: The agent being checked
        :returns: True if the agent is out of bounds

        """
        x = agent.x
        y = agent.y

        if x < 0 or x >= self.width:
            return True
        if y < 0 or y >= self.height:
            return True
        return False

    def render_grid(self, scale=1):
        array = np.copy(self.grid)
        array[array > 0] = 255
        
        if scale > 0:
            array = zoom(array, scale, order=0)

        return Image.fromarray(array)

    def epoch(self, n_trials=20):
        """Runs n_trials in an epoch. The top performing
        agents are returned along with their average score

        :n_trials: The number of trials to perform
        :returns: A dict of agent performances

        """

        # Create scoreboard to record agent's performance
        scoreboard = defaultdict(lambda: 0, {})

        for trial in range(n_trials):
            print('\tSimulating trial: %d' % trial)
            self.__reset_grid__()
            while len(self.active_agents) > 0:
                self.step()
                self.render_grid(scale = 4).convert('RGB').save('%010d.jpg' % self.global_iter)
                self.global_iter += 1

            # Update scoreboard
            for agent in self.my_agents:
                scoreboard[agent] += agent.lifetime

            agents = self.my_agents
            self.my_agents = []
            self.register_agents(agents)

        # Trials are finished, take average
        for agent in self.my_agents:
            scoreboard[agent] = float(scoreboard[agent]) / float(n_trials)

        return scoreboard


    def simulate(self, num_epochs=10):
        for epoch in range(num_epochs):
            print('Simulating Epoch: %d' % epoch)
            
            results = self.epoch()

            data = sorted(list(results.values()))
            print('Top %d. Low %d. Avg %d' % (data[0], data[len(data) - 1], float(sum(data)) / float(len(data))))
            
            self.breed(results)

    def breed(self, agent_performances):
        """Breeds agents after an epoch

        """
        sorted_agents = sorted(agent_performances, key=agent_performances.get)

        n_top = len(sorted_agents)
        new_agents = []

        for i in range(self.num_agents):
            parent1 = sorted_agents[(i // n_top) % n_top]
            parent2 = sorted_agents[(i+1) % n_top]
            offspring = parent1 + parent2
            new_agents.append(offspring)

        self.my_agents = []
        self.active_agents = []

        self.register_agents(new_agents)

    def __reset_grid__(self):
        self.grid = np.zeros((self.width, self.height), dtype=np.uint32)

