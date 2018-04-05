#!/usr/bin/env python
# -*- coding: utf-8 -*-


import networkx as nx
from activation import sigmoid

class Node():

    """This class represents a node"""

    def __init__(self, label, node_type):
        """Initializes a node

        :type: The node's type: input, hidden, output

        """

        self.type = node_type
        self.label = label

class NEAT_Pool():

    """A class which holds a pool of genomes"""

    def __init__(self, input_src, input_dims, output_dim):
        """Initializes a Pool"""
        self.innovation_number = 0

        self.nodes = {}
        self.node_num = 0

        self.input_nodes = []
        self.output_nodes = []

        # Create a base genome for every individual to start out with
        # To start this we need to generate a base topology with a complete
        # graph from input neurons to output neurons
        for x in range(input_dims[0]):
            for y in range(input_dims[1]):
                # Create an input node for each pixel source
                self.node_num += 1
                node = Node(self.node_num, 'input')
                self.nodes[node.label] = node
                self.input_nodes.append(node)

        for x in range(output_dim):
            self.node_num += 1
            node = Node(self.node_num, 'output')
            self.nodes[node.label] = node
            self.output_nodes.append(node)

        # Generate connections
        initial_genome = {}
        for input_node in self.input_nodes:
            for output_node in self.output_nodes:
                edge = create_connection(input_node, output_node, self)
                initial_genome[edge['innovation']] = edge

        self.starting_genome = initial_genome


class NEAT_Network():

    """An instance of a genome"""

    def __init__(self, genome, pool_ref):
        """Initializes a network with the desired
        genome

        :genome: The genome to construct the network

        """

        self.genome = genome
        self.network = nx.DiGraph()
        self.pool = pool_ref
        self.__load_genome__(genome)

    def __load_genome__(self, genome):
        for innov, gene in genome.items():
            if gene['enabled']:
                self.network.add_node(gene['in'], value=0.0)
                self.network.add_node(gene['out'], value=0.0)
                self.network.add_edge(gene['in'],\
                                      gene['out'],\
                                      weight=gene['weight'])

    def feedforward(self, data):
        # Load data

        output = []

        flat_data = data.flatten()
        i = 0
        for node in self.pool.input_nodes:
            self.network.nodes[node]['value'] = flat_data[i]
            i += 0

        # For each output node, sum the previous nodes in the graph
        for node in self.pool.output_nodes:
            output.append(get_weighted_sum(node, self.network))

        return output

    def innovate_node(self):
        pass


def get_weighted_sum(cur_node, network):
    """Gets the node's weighted sum

    :cur_node: The reference to the current node
    :network: A reference to a network to load values
    :returns: The weighted sum (with activation function)

    """
    w_sum = 0.0
    for node in network.predecessors(cur_node):
        w_sum += get_weighted_sum(node, network)

    if network.in_degree(cur_node) == 0:
        w_sum = network.nodes[cur_node]['value']
    else:
        w_sum = sigmoid(w_sum)

    return w_sum


def create_connection(input_node, output_node, pool):
    """Creates a connection between two nodes

    :input_node: The input node label
    :output_node: The output node label
    :returns: The created connection

    """

    pool.innovation_number += 1

    edge = {}
    edge['in'] = input_node
    edge['out'] = output_node
    edge['weight'] = 1.0
    edge['enabled'] = True
    edge['innovation'] = pool.innovation_number

    return edge
