#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random

import networkx as nx
from networkx.algorithms.shortest_paths import has_path

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

    def new_hidden_node(self):
        self.node_num += 1
        node = Node(self.node_num, 'hidden')
        self.nodes[node.label] = node
        return node

    def new_gene(self, input_node, output_node):
        """Constructs a new gene from an input node
        and an output node. It's innovation number is
        assigned. Note that any new gene's weight is
        zero.

        :input_node: Node to read values from
        :output_node: Node to write values to

        """
        edge = create_connection(input_node, output_node, self)
        return edge


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
        for node in self.pool.input_nodes:
            self.network.add_node(node)
        for innov, gene in genome.items():
            if gene['enabled']:
                if gene['in'] not in self.pool.nodes:
                    self.network.add_node(gene['in'], value=0.0)
                if gene['out'] not in self.pool.nodes:
                    self.network.add_node(gene['out'], value=0.0)
                self.network.add_edge(gene['in'],
                                      gene['out'],
                                      weight=gene['weight'],
                                      enabled=gene['enabled'],
                                      innovation=gene['innovation'])

    def __add__(self, other):
        """Overrides the '+' operator for easy crossover

        :other: The other network
        :returns: TODO

        """
        if type(other) is not NEAT_Network:
            raise TypeError('You can only add Networks to other Networks')

        new_genome = {}

        for i in range(self.pool.innovation_number):
            left_gene = self.genome[i] if i in self.genome else None
            right_gene = other.genome[i] if i in other.genome else None

            if left_gene is not None and right_gene is not None:
                # Randomly choose a gene
                chosen_gene = random.choice([left_gene, right_gene])
                new_genome[i] = chosen_gene.copy()
            elif left_gene is not None:
                # Either disjoint or excess gene
                # Use this gene
                new_genome[i] = left_gene.copy()
            elif right_gene is not None:
                new_genome[i] = right_gene.copy()
            else:
                # Neither parent contains the gene for this innovation number
                continue

            if new_genome[i]['enabled']:
                # Random chance of disabling node
                if random.uniform(0.0, 1.0) < 0.1:
                    new_genome[i]['enabled'] = False
            else:
                # Random chance of enabling node
                if random.uniform(0.0, 1.0) < 0.75:
                    new_genome[i]['enabled'] = True

            # Mutate weights
            new_genome[i]['weight'] += random.randrange(-1, 1)

        self.mutate()

        return NEAT_Network(new_genome, self.pool)

    def mutate(self):
        """Mutates the network

        """
        chance = random.uniform(0.0, 1.0)
        if chance > 0.7:
            self.innovate_node()
        chance = random.uniform(0.0, 1.0)
        if chance > 0.7:
            self.innovate_edge()

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
        # Choose an edge to mutate
        # desired_edge = random.choice(self.network.edges)
        edges = [edge for edge in self.network.edges(data=True)]
        desired_edge = random.choice(edges)

        # Ask pool to generate a hidden node
        new_node = self.pool.new_hidden_node()

        # Create connections
        gene1 = self.pool.new_gene(desired_edge[0], new_node)
        gene2 = self.pool.new_gene(new_node, desired_edge[1])
        
        gene2['weight'] = desired_edge[2]['weight']

        self.genome[gene1['innovation']] = gene1
        self.genome[gene2['innovation']] = gene2

        # Disable old edge
        desired_edge[2]['enabled'] = False

    def innovate_edge(self):
        """Innovates a new edge connection

        """
        # Choose random node
        choices = list(self.network.nodes)
        node1 = random.choice(choices)
        choices.remove(node1)
        node2 = random.choice(choices)

        if has_path(self.network, node1, node2):
            # There exists a path node1->node2
            # No not make a path from node2->node1
            src_node = node1
            dest_node = node2
        elif has_path(self.network, node2, node1):
            # There exists a path node2->node1
            src_node = node2
            dest_node = node1
        else:
            # There is no path between node1 node2
            return

        new_gene = self.pool.new_gene(src_node, dest_node)
        self.genome[new_gene['innovation']] = new_gene


def get_weighted_sum(cur_node, network):
    """Gets the node's weighted sum

    :cur_node: The reference to the current node
    :network: A reference to a network to load values
    :returns: The weighted sum (with activation function)

    """
    w_sum = 0.0
    try:
        for node in network.predecessors(cur_node):
            w_sum += get_weighted_sum(node, network) * network[node][cur_node]['weight']

        if network.in_degree(cur_node) == 0:
            w_sum = network.nodes[cur_node]['value']
        else:
            w_sum = sigmoid(w_sum)
    except nx.exception.NetworkXError:
        w_sum = 0.0
    finally:
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
    edge['weight'] = 0.0
    edge['enabled'] = True
    edge['innovation'] = pool.innovation_number

    return edge
