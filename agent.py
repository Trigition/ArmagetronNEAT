#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from neat import NEAT_Network


class Agent():

    """An agent whic exists on the grid"""

    def __init__(self,
                 agent_id,
                 pool_ref,
                 genome=None,
                 agent_name='Agent',
                 sensor_radius=5):
        """Initializes an agent

        :agent_id: The agent id, must be unique to other agents
        :grid_ref: A reference to the grid the agent resides on
        :agent_name: The name of the agent
        :sensor_radius: How far the agent can see in either direction

        """

        self.agent_id = agent_id
        self.agent_name = agent_name
        self.lifetime = 0
        self.pool = pool_ref
        self.sensor_radius = sensor_radius
        self.grid = None
        self.sensor_history = []

        if genome:
            self.brain = NEAT_Network(genome, pool_ref)
        else:
            self.brain = NEAT_Network(pool_ref.starting_genome, pool_ref)
        self.left = 1
        self.center = 0
        self.right = 2

    def __str__(self):
        return '%s-%d' % (self.agent_name, self.agent_id)

    def __add__(self, other):
        """Overloaded addition operator. This allows
        two agents to crossover for genetic evolution

        :other: Reference to the comparator
        :returns: A new agent with a mixed genome

        """
        new_agent = Agent(0, self.pool, sensor_radius=self.sensor_radius)
        new_agent.brain = self.brain + other.brain
        return new_agent

    def set_grid(self, grid):
        self.grid = grid

    def get_complexity(self):
        """Returns the number of graph connections
        :returns: The number of neuron connections

        """
        return len(self.brain.network.edges)

    def set_pos(self, x, y):
        """Sets the position of the agent

        :x: The x coordinate
        :y: The y coordinate
        """
        self.x = x
        self.y = y

    def set_orientation(self, heading):
        self.heading = heading % 4

    def move_forward(self):
        delta_x = (self.heading % 2) * ((self.heading - 2) * -1)
        delta_y = ((self.heading + 1) % 2) * ((self.heading - 1) * -1)

        # Finally change position
        self.set_pos(self.x + delta_x, self.y + delta_y)

    def step(self):
        cur_sense = self.sense()

        result = self.brain.feedforward(cur_sense)
        result = result.index(max(result))

        if result == self.left:
            self.set_orientation(self.heading - 1)

        if result == self.right:
            self.set_orientation(self.heading + 1)

        self.move_forward()
        self.lifetime += 1

    def sense(self):
        max_x = self.x + self.sensor_radius
        min_x = self.x - self.sensor_radius
        max_y = self.y + self.sensor_radius
        min_y = self.y - self.sensor_radius

        sensor_dia = 2*self.sensor_radius + 1
        sensor_shape = (sensor_dia, sensor_dia)

        self.sensor = np.empty(sensor_shape)
        self.sensor.fill(255)

        x_range = (max(0, min_x), min(self.grid.width, max_x+1))
        y_range = (max(0, min_y), min(self.grid.height, max_y+1))

        for x in range(x_range[0], x_range[1]):
            for y in range(y_range[0], y_range[1]):
                i_x = self.sensor_radius + (x - self.x)
                i_y = self.sensor_radius + (y - self.y)
                if self.grid.grid[x][y] <= 0:
                    self.sensor[i_x][i_y] = 0

        # Rotate sensor so the agent is always going 'upwards'
        self.sensor = np.rot90(self.sensor, self.heading)

        self.sensor_history.append(np.copy(self.sensor))
        return self.sensor

    def render_sensor(self, scale=5):
        array = np.copy(self.sensor)
        array[array > 0] = 255
        if scale > 1:
            array = zoom(array, scale, order=0)
        return Image.fromarray(array).convert('RGB')

    @staticmethod
    def check_type(variable, dtype):
        if type(variable) is not dtype:
            raise TypeError('%s is not of type %s' % (variable, dtype))
