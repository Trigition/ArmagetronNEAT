#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1.0 - y)


class MLP_NN(object):

    """Basic Neural Network"""

    def __init__(self, n, hidden, output):
        """ Initializes a network

        :n: The number of input neurons
        :hidden: The number of hidden neurons
        :output: The number of output neurons

        """
        # Add 1 for bias node
        self.n = n + 1
        self.hidden = hidden
        self.output = output

        # Set up array of 1s for activations
        self.ai = [1.0] * self.n
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output

        # Randomize weights
        self.wi = np.random.rand(self.n, self.hidden)
        self.wo = np.random.rand(self.hidden, self.output)

        # Create zero-arrays for changes
        self.ci = np.zeros((self.n, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedforward(self, inputs):
        """Feeds input through the network

        :inputs: TODO
        :returns: TODO

        """
        if len(inputs) != self.n - 1:
            raise ValueError('Feed input != neural input length: %d != %d' % (len(inputs), self.n))

        for i in range(self.n - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.hidden):
            net_sum = 0.0
            for i in range(self.n):
                net_sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(net_sum)

        # output activation
        for k in range(self.output):
            net_sum = 0.0
            for j in range(self.hidden):
                net_sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(net_sum)

        return self.ao[:]

    @staticmethod
    def crossover(nn0, nn1):
        """Performs genetic cross over on the weights
        within the network

        :nn0: First parent's network
        :nn1: Second parent's network
        :returns: A new network with 50% crossover

        """
        if nn0.n != nn1.n or \
           nn0.hidden != nn1.hidden or \
           nn0.output != nn1.output:
               raise ValueError('Crossover networks must have same dimensions')

        child_network = MLP_NN(nn0.n - 1, nn0.hidden, nn0.output)

        input_mask = np.random.randint(0,1,size=nn0.wi.shape)
        output_mask = np.random.randint(0,1,size=nn0.wo.shape)

        result_ih = np.zeros(nn0.wi.shape)
        result_ho = np.zeros(nn0.wo.shape)

        for x in range(input_mask.shape[0]):
            for y in range(input_mask.shape[1]):
                # Perform crossover
                result_ih[x][y] = nn0.wi[x][y] if input_mask[x][y] else nn1.wi[x][y]

        # Mutate
        w, h = result_ih.shape
        result_ih = np.add(result_ih, ((np.random.rand(w,h) * 2) - 1) * 0.05)

        for x in range(output_mask.shape[0]):
            for y in range(output_mask.shape[1]):
                w, h = result_ho.shape
                # Perform crossover
                result_ho[x][y] = nn0.wo[x][y] if output_mask[x][y] else nn1.wo[x][y]
        # Mutate
        w, h = result_ho.shape
        result_ho = np.add(result_ho, ((np.random.rand(w,h) * 2) - 1) * 0.05)

        child_network.wi = result_ih
        child_network.wo = result_ho

        return child_network
