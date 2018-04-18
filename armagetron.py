#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import zoom
from neat import NEAT_Pool
from populations import Population
from worker_pool import Worker_Pool
from rendering import Renderer

class Simulation():

    """This class represents a simulation pool of grids
    and agents. Here is where simulation parameters are set
    and used."""

    def __init__(self,
            population_size,
            sim_population,
            sensor_radius=5,
            n_threads=1):
        """Initializes a simulation

        :population_size: The allowable population size per generation
        :sim_population: The allowable population size per grid simulation
        :n_threads: The number of threads to utilize

        """

        # Type checking
        if type(n_threads) is not int:
            raise TypeError('Thread count must be a positive integer')
        elif n_threads < 1:
            raise ValueError('Thread count must be above 0')

        self.n_threads = n_threads
        self.sensor_radius = sensor_radius

        # Create genetic pool for simulation
        dims = (sensor_radius+1, sensor_radius+1)
        self.pool = NEAT_Pool(dims, 3)
        self.population = Population(population_size, sim_population, self.pool)

        self.renderer = Renderer()

    def simulate(self, generations=1):
        """Evolves the Population until the specified generation

        :generations: TODO
        :returns: TODO

        """

        sim_dims = (100, 100)
        workers = Worker_Pool(self.n_threads)

        for generation in range(generations):
            print('Simulating Generation: %d' % generation)
            workers.reset_results()
            pops = [pop for pop in self.population]
            grids = []

            i =0
            for population in pops:
                g = Grid(*sim_dims, population, self.renderer.buffer, generation, i)
                grids.append(g)
                i += 1

            for grid in grids:
                workers.add_task(grid.simulate)
            workers.wait_for_completion()

            # Combine results
            scores = {}
            for d in workers.results:
                for k, v in d.items():
                    scores[k] = v

            self.population.breed(scores)

        self.renderer.wait_till_done()


class Grid():

    """A grid of pixels and agents"""

    def __init__(self,
            width,
            height,
            agents,
            image_queue,
            generation,
            population_number):
        """Initializes the grid with a specific width
        and height

        :width: The width of the grid
        :height: The height of the grid
        :num_agents: The number of agents to simulate
        :agents_per_sim: The maximum number of agents per
        simulation

        """
        if type(width) is not int or type(height) is not int:
            raise TypeError('Grid dimensions must be integers')
        if width < 1 or height < 1:
            raise ValueError('Grid dimensions must be positive values')

        self.width = width
        self.height = height
        self.num_agents = len(agents)

        # Create grid
        self.__reset_grid__()

        self.active_agents = []
        self.my_agents = []

        self.register_agents(agents)

        self.iteration = 0
        self.image_queue = image_queue
        self.generation = generation
        self.pop_num = population_number

    def __str__(self):
        """Returns a string representation of the Grid instance
        :returns: A string

        """
        s = 'Grid-%03d-%03d-%03d' % (self.generation, self.pop_num, self.iteration)
        return s

    def register_agents(self, agents):
        for agent in agents:
            agent.set_grid(self)
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
                # Punish agent for going out of bounds
                agent.lifetime /= 10.0
                self.active_agents.remove(agent)
                self.render_agent(agent)               
                continue
            elif self.grid[agent.x][agent.y] != 0:
                # Collision into wall
                self.active_agents.remove(agent)
                self.render_agent(agent)
                continue
            # make a step
            self.grid[agent.x][agent.y] = agent.agent_id
            agent.step()

        # self.render_grid()
        self.grid_history.append(np.copy(self.grid))
        self.iteration += 1

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

    def render_agent(self, agent):
        job = {}
        job['matrix'] = agent.sensor_history
        job['filename'] = str(agent)
        self.image_queue.put(job)

    def render_grid(self, scale=4):
        job = {}
        job['matrix'] = self.grid_history
        job['filename'] = str(self)
        self.image_queue.put(job)

    def simulate(self):
        while len(self.active_agents) > 0:
            self.step()

        scores = {}
        for agent in self.my_agents:
            scores[agent] = agent.lifetime

        self.render_grid()

        return scores

    def __reset_grid__(self):
        self.grid = np.zeros((self.width, self.height), dtype=np.uint32)
        self.grid_history = []
