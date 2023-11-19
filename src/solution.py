import os
from itertools import combinations
from typing import Set, List

import networkx as nx
from matplotlib import pyplot as plt

from src.config import Config
from src.utils import Instance, is_s_plex


class Solution:
    def __init__(self, instance: Instance, x: dict, components: list = None, obj: float = None):
        self.instance = instance
        self.x = x  # Dictionary with edge as key and 1 if edge is in final graph, 0 otherwise

        # List of components. Each component is a set of nodes
        self.components: List[Set] = components if components is not None else self._get_components()
        self.obj = obj  # Objective value of the solution. Can be None if not computed

    def _get_components(self):
        """
        Compute the components of the solution
        """
        G = nx.Graph()
        G.add_nodes_from(self.instance.nodes)
        G.add_edges_from([e for e in self.instance.edges if self.x[e] == 1])
        return list(nx.connected_components(G))

    def __repr__(self):
        G = nx.Graph()
        G.add_nodes_from(self.instance.nodes)
        G.add_edges_from([e for e in self.instance.edges if self.x[e] == 1])
        nx.draw(G, with_labels=True)
        plt.show()
        return f"Solution cost: {self.evaluate()}, number of connected components: {nx.number_connected_components(G)}"

    def evaluate(self):
        """
        Evaluate the solution
        """
        obj = 0
        for edge in self.instance.edges:
            if self.x[edge] != self.instance.in_instance[edge]:
                obj += self.instance.weight[edge]
        return obj

    def save(self, config: Config, path=None):
        """
        Save the solution in a txt file
        Only write the updated edges in format "i j" where i<j
        :param config: config object
        :param path: path to save the solution
        """
        if path is None:
            path = f"{config.solutions_dir}/{config.method}/" \
                   f"{'randomized' if config.method_params['randomized'] else 'deterministic'}"

        # Create directory if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f"{path}/{self.instance.name}.txt", "w") as f:
            f.write(f"{self.instance.name}\n")  # First line is the name of the instance

            for edge in self.instance.edges:
                if self.x[edge] != self.instance.in_instance[edge]:
                    f.write(f"{edge[0]} {edge[1]}\n")

    def generate_neighborhood(self, type: str, neighborhood_config: dict):
        """
        Generate a neighborhood of the current solution
        :param type: type of neighborhood
        :param neighborhood_config: dictionary with neighborhood parameters
        :return: list of neighbors
        """
        type, params = type.split('_', 1)
        params = params.split('_')
        if type == 'kflips':
            return self.kflips_neighborhood(neighborhood_config['kflips'], *params)
        elif type == 'swap':
            return self.swap_nodes(neighborhood_config['swap'], *params)
        elif type == 'movenodes':
            return self.move_nodes(neighborhood_config['movenodes'], *params)
        else:
            raise ValueError(f'Neighborhood type {type} does not exist')

    def kflips_neighborhood(self, config, k):
        k = int(k)
        edges = {e for e in self.instance.edges if self.x[e]}
        if not config['remove_edge_in_instance_allowed']:
            edges -= self.instance.edges_in_instance

        # Iterate over all possible k-choices of edges
        for edges_to_remove in combinations(edges, k):
            # Remove the k edges
            new_x = {e: 0 if e in edges_to_remove else self.x[e] for e in self.instance.edges}
            new_x, components = self._make_s_plex_on_full_graph(new_x, forbiden_edges=edges_to_remove)

            # Check if feasible solution was found
            if new_x is None:
                continue

            # TODO: pass the components to the solution
            # TODO: pass the objective value to the solution (delta evaluation)
            yield Solution(self.instance, new_x, components)

    def _make_s_plex_on_full_graph(self, new_x, forbiden_edges=None):
        if forbiden_edges is None:
            forbiden_edges = set()
        G = nx.Graph()
        G.add_edges_from([k for k, v in new_x.items() if v])
        components = list(nx.connected_components(G))
        for comp in components:
            new_x = self._make_s_plex_on_component(new_x, comp, forbiden_edges)
            if new_x is None:
                return None, None
        return new_x, components

    def _make_s_plex_on_component(self, new_x, comp, forbidden_edges):
        G_comp = nx.Graph()
        G_comp.add_nodes_from(comp)
        while not is_s_plex(self.instance.s, G_comp):
            min_degree_node = min(G_comp.nodes(), key=lambda x: G_comp.degree(x))
            candidates = [node for node in G_comp.nodes() if node != min_degree_node and
                          (min_degree_node, node) not in G_comp.edges() and
                          (min_degree_node, node) not in forbidden_edges]
            if not candidates:
                return None
            target_node = min(candidates, key=lambda x: self.instance.weight[min_degree_node, x])
            G_comp.add_edge(min_degree_node, target_node)
            new_x[min_degree_node, target_node] = 1
        return new_x

    def swap_nodes(self, config, n):
        pass

    def move_nodes(self, config, k, l):
        pass

