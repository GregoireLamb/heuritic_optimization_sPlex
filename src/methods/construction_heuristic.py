import math
from itertools import combinations

import networkx as nx
import random

from src.utils import is_s_plex, Instance
from src.solution import Solution


class ConstructionHeuristic:
    def __init__(self, params=None):

        assert params is not None, 'No params given. Must include at least the randomized boolean parameter'

        self._randomized = params['randomized']
        self.top_kth = params['top_kth']

        self._instance = None
        self._x = {}
        self._components = []
        self._s = -1

    def solve(self, instance: Instance) -> Solution:  # TODO put last_compoment_only to false by default
        """
        Solve the instance using the deterministic construction heuristic
        :param instance: instance to solve
        :return: solution
        """
        self._s = instance.s
        self._instance = instance

        self._x = {e: 0 for e in instance.edges}

        to_add = instance.nodes.copy()
        best_option_dict = {}  # component_index: (cost)

        for i in to_add:
            # if i % 10 == 0:
            #     print(f' - {i} nodes already added')

            # New component
            best_cost = self.compute_cost_of_new_component(i)
            best_component = None
            if self._randomized:
                best_option_dict[None] = best_cost

            # Existing component
            for index, k in enumerate(self._components):
                cost = self.compute_insertion_cost_to_component(i, k)
                if self._randomized:
                    best_option_dict[index] = cost
                if cost < best_cost:
                    best_cost = cost
                    best_component = index

            if self._randomized:
                best_option_list = sorted(best_option_dict.items(), key=lambda x: x[1])  # Sort by cost
                best_component = random.choice(best_option_list[:self.top_kth])[0]

            if best_component is None:
                self._components.append({i})
                # print(f'Creating new component with node {best_node}, cost = {best_cost}')
            else:
                self._components[best_component].add(i)
                # print(f'Adding node {best_node} to component {best_component}, cost = {best_cost}')

        for k in self._components:
            new_edges = self.get_edges_of_s_plex(k)
            self._x.update({e: 1 for e in new_edges})

        return Solution(instance, self._x, self._components)

    def compute_cost_of_new_component(self, i):
        """
        Compute the cost of creating a new component with node i
        :param i: node
        :return: cost
        """
        cost = 0
        for k in self._components:
            for j in k:
                if (i, j) in self._instance.edges_in_instance:
                    cost += self._instance.weight[(i, j)]
        return cost

    def compute_insertion_cost_to_component(self, i, k):
        """
        Compute the cost of inserting node i to component k
        :param i: node
        :param k: component
        :return: cost
        """
        G = self.create_component_graph(k, i)

        cost = 0
        cost = self.disconnecting_cost(cost, i, k)
        cost = self.make_s_plex(G, cost)

        return cost

    def get_edges_of_s_plex(self, k):
        """
        Get the edges of the s-plex of component k
        :param k: component
        :return: edges of the s-plex
        """
        G = self.create_component_graph(k)
        self.make_s_plex(G, 0)
        return set((min(i, j), max(i, j)) for i, j in G.edges())

    def make_s_plex(self, G, cost):
        """
        Make G an s-plex
        :param G: graph
        :param cost: current cost
        :return: cost of making G an s-plex
        """
        # cost = 0
        # if not is_s_plex(self._s, G):
        #     all_node_pairs = combinations(G.nodes(), 2)
        #     for node_pair in all_node_pairs:
        #         if node_pair not in G.edges():
        #             G.add_edge(*node_pair)
        #             cost += self._instance.weight[node_pair]
        while not is_s_plex(self._s, G):
            min_degree_node = min(G.nodes(), key=lambda x: G.degree(x))
            best_node_to_add = -1
            best_sup_cost = 1e10
            for node in G.nodes():
                if node != min_degree_node and (min_degree_node, node) not in G.edges():
                    sup_cost = self._instance.weight[min_degree_node, node]
                    if sup_cost < best_sup_cost:
                        best_sup_cost = sup_cost
                        best_node_to_add = node
            G.add_edge(min_degree_node, best_node_to_add)
            cost += best_sup_cost
        return cost

    def create_component_graph(self, k, i=None):
        """
        Create a graph with the nodes in component k and the pre-existing edges
        :param k: component
        :param i: node to add
        :return: graph
        """
        G = nx.Graph()
        G.add_nodes_from(k)
        if i is not None:
            G.add_node(i)
        for n in G.nodes():
            for m in G.nodes():
                if n < m and (n, m) in self._instance.edges_in_instance:
                    G.add_edge(n, m)
        return G

    def disconnecting_cost(self, cost, i, k):
        """
        Compute the disconnecting cost of adding node i to component k
        :param cost: current cost
        :param i: node
        :param k: component
        :return: disconnecting cost
        """
        for c in self._components:
            if c == k:
                continue
            for j in c:
                cost += self._instance.weight[(i, j)] * self._instance.in_instance[(i, j)]
        return cost
