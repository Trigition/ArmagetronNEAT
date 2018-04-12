#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random
import numpy as np
from agent import Agent


class Population():

    """This class holds a collection of agents
    together as a discrete population"""

    def __init__(self, max_population, sim_population, genetic_pool):
        """Initializes a population

        :max_population: The maximum population at any
        particular instance
        :sim_population: The maximum population allowed
        in any simulation instance
        :genetic_pool: A reference to NEAT's genetic pool

        """

        self.max_population = max_population
        self.sim_population = sim_population
        self.genetic_pool = genetic_pool
        self.cur_id = 0
        
        self.current_population = []
        self.current_generation = 0
        self.__set_new_population__()

    def __iter__(self):
        """Returns the iterator for Populations

        """
        return self

    def __next__(self):
        """Returns the next population to simulate
        :returns: A list of agents to simulate

        """
        if len(self.agents_waiting_for_sim) == 0:
            raise StopIteration
        try:
            sample = random.sample(self.agents_waiting_for_sim, self.sim_population)
            #self.agents_waiting_for_sim -= sample
            tmp = self.agents_waiting_for_sim
            self.agents_waiting_for_sim = [item for item in tmp if item not in sample]
            return sample
        except ValueError:
            # Agent queue is smaller than sim_population
            sample = self.agents_waiting_for_sim
            self.agents_waiting_for_sim = []
            return sample

    def __set_new_population__(self, population=None):
        """Sets the new population

        :population: A list of agents to be the new population.
        If this list is none a brand new set of agents are created

        """
        if population:
            # Do some Value checks
            if len(population) != self.max_population:
                raise ValueError('Provided population does not match\
                        this Population instance\'s maximum population')
            self.current_population = population
        else:
            for i in range(self.max_population):
                agent = Agent(self.cur_id, self.genetic_pool)
                self.current_population.append(agent)
                self.cur_id += 1

        self.agents_waiting_for_sim = self.current_population

    def get_next_sim_population(self):
        """Randomly select the next simulation population
        :returns: A list of agents to simulate

        """
        try:
            sample = random.sample(self.agents_waiting_for_sim, self.sim_population)
            self.agents_waiting_for_sim -= sample
        except ValueError:
            # Agent queue smaller than sim_population
            sample = self.sim_population
            self.agents_waiting_for_sim = []

        return sample

    def breed(self, agent_scores, percentile=80, n_parents=2):
        """Breeds agents for the next generation.
        New agents will replace the current population
        list.

        :agent_scores: A dict of agent scores
        :percentile: The percentile allowed to
        be considered 'elite' and allowed to be
        primary sources of genetic crossover
        :n_parents: The number of biological parents
        a child is allowed to have.

        """
        # Perform some typechecking
        if type(n_parents) is not int:
            raise TypeError('Only integer value are allowed for number of parents')
        elif n_parents < 1:
            raise ValueError('Offspring must have 2 or more parents')

        if percentile < 0:
            raise ValueError('Cannot compute negative percentiles...')
        elif percentile > 100:
            raise ValueError('Cannot compute 100+ percentiles...')

        # Prepare for population selection
        cutoff = np.percentile(list(agent_scores.values()), percentile)
        elites, commoners = split_population(agent_scores, cutoff)

        # Elites get priority when it comes to breeding, for each elite
        # select a mate(s)
        n_elites = len(elites)
        n_commoners = len(commoners)
        next_generation = []
        n_generated = 0

        # Generate offspring
        while n_generated < self.max_population:
            # Generate offspring
            elite = elites[n_generated % n_elites]
            mates = random.sample(commoners, n_parents - 1)
            parents = mates + [elite]
            # Mate
            offspring = parents[0]
            for parent in parents[1:]:
                offspring += parent

            offspring.agent_id = self.cur_id

            next_generation.append(offspring)
            self.cur_id += 1

        # Set the new population
        self.__set_new_population(next_generation)
        self.current_generation += 1


def split_population(agent_scores, cutoff):
    """Splits the population into two sets: an elite population
    which surpasses the cutoff and a common population which is
    at or below the cutoff

    :agent_scores: A dict containing agents and their scores
    :cutoff: TODO
    :returns: TODO

    """
    elite_population = []
    common_population = []

    for agent, score in agent_scores.items():
        if score > cutoff:
            elite_population.append(agent)
        else:
            common_population.append(agent)

    return (elite_population, common_population)
