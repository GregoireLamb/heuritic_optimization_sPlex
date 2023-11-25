import os
from itertools import combinations
import random
from typing import Set, List, Generator

import networkx as nx
from matplotlib import pyplot as plt

from src.config import Config
from src.utils import Instance, is_s_plex


class Solution:
    def __init__(self, instance: Instance, x: dict, components: list = None, graph: nx.Graph = None, obj: float = None):
        self.instance = instance
        self.x = x  # Dictionary with edge as key and 1 if edge is in final graph, 0 otherwise

        if graph is None:
            self.graph = nx.Graph()
            self.graph.add_nodes_from(self.instance.nodes)
            self.graph.add_edges_from([k for k, v in self.x.items() if v])
        else:
            self.graph = graph

        # List of components. Each component is a set of nodes
        self.components: List[Set] = components if components is not None else list(nx.connected_components(self.graph))
        self.obj = obj  # Objective value of the solution. Can be None if not computed

    def _get_components(self):
        """
        Compute the components of the solution
        """
        return list(nx.connected_components(self.graph))

    def __repr__(self):
        nx.draw(self.graph, with_labels=True)
        plt.show()
        return f"Solution cost: {self.evaluate()}, number of connected components: {len(self.components)}"

    def evaluate(self):
        """
        Evaluate the solution
        """
        if self.obj is not None:
            return self.obj
        obj = 0
        for edge in self.instance.edges:
            if self.x[edge] != self.instance.in_instance[edge]:
                obj += self.instance.weight[edge]
        self.obj = obj
        return obj

    def is_feasible(self):
        """
        Check if the solution is feasible
        """
        feasible = all([is_s_plex(self.instance.s, nx.subgraph(self.graph, comp)) for comp in self.components])
        if feasible:
            return True
        # Find node with degree < n - s
        for comp in self.components:
            if is_s_plex(self.instance.s, nx.subgraph(self.graph, comp)):
                continue
            for node in comp:
                if self.graph.degree(node) < len(comp) - self.instance.s:
                    print(f"Node {node} has degree {self.graph.degree(node)}. "
                          f"Should be at least {len(comp) - self.instance.s} (s = {self.instance.s})")

        return False

    def save(self, config: Config, path=None):
        """
        Save the solution in a txt file
        Only write the updated edges in format "i j" where i<j
        :param config: config object
        :param path: path to save the solution
        """
        if path is None:
            path = f"{config.solutions_dir}/{config.method}"
            if config.method == 'construction_heuristic':
                path += f"/{config.det_or_random_construction}"

        # Create directory if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f"{path}/{self.instance.name}.txt", "w") as f:
            f.write(f"{self.instance.name}\n")  # First line is the name of the instance

            for edge in self.instance.edges:
                if self.x[edge] != self.instance.in_instance[edge]:
                    f.write(f"{edge[0]} {edge[1]}\n")

    def generate_neighborhood(self, type: str, neighborhood_config: dict) -> Generator['Solution', None, None]:
        """
        Generate a neighborhood of the current solution
        :param type: type of neighborhood
        :param neighborhood_config: dictionary with neighborhood parameters
        :return: list of neighbors
        """
        type, params = type.split('_', 1)
        params = params.split('_')
        neighborhood_config = neighborhood_config.get(type, {})
        if type == 'kflips':
            return self.kflips_neighborhood(neighborhood_config, *params)
        elif type == 'nodeswap':
            return self.swap_nodes_neighborhood(neighborhood_config, *params)
        elif type == 'movenodes':
            return self.move_nodes_neighborhood(neighborhood_config, *params)
        else:
            raise ValueError(f'Neighborhood type {type} does not exist')

    def get_random_neighbor(self, type: str, neighborhood_config: dict) -> 'Solution':
        """
        Generate a random neighbor of the current solution
        :param type: type of neighborhood
        :param neighborhood_config: dictionary with neighborhood parameters
        :return: random neighbor
        """
        type, params = type.split('_', 1)
        params = params.split('_')
        neighborhood_config = neighborhood_config.get(type, {})
        if type == 'kflips':
            return self.kflips_random_neighbor(neighborhood_config, *params)
        elif type == 'nodeswap':
            return self.swap_nodes_random_neighbor(neighborhood_config, *params)
        elif type == 'movenodes':
            return self.move_nodes_random_neighbor(neighborhood_config, *params)
        else:
            raise ValueError(f'Neighborhood type {type} does not exist')

    def kflips_neighborhood(self, config, k):
        k = int(k)
        edges = {e for e in self.instance.edges if self.x[e]}
        if not config['remove_edge_in_instance_allowed']:
            edges -= self.instance.edges_in_instance

        # Iterate over all possible k-choices of edges
        for edges_to_remove in combinations(edges, k):
            sol = self.remove_edges(edges_to_remove)
            if sol is None:
                continue

            # TODO: pass the components to the solution
            # TODO: pass the objective value to the solution (delta evaluation)
            yield sol

    def kflips_random_neighbor(self, config, k):
        k = int(k)
        edges = {e for e in self.instance.edges if self.x[e]}
        if not config['remove_edge_in_instance_allowed']:
            edges -= self.instance.edges_in_instance

        while True:
            edges_to_remove = random.sample(edges, k)
            sol = self.remove_edges(edges_to_remove)
            # Check if feasible solution was found
            if sol is not None:
                return sol

    def remove_edges(self, edges_to_remove):
        # Remove the k edges
        new_x = {e: 0 if e in edges_to_remove else self.x[e] for e in self.instance.edges}
        new_G = nx.Graph()
        new_G.add_nodes_from(self.instance.nodes)
        new_G.add_edges_from([e for e in self.instance.edges if new_x[e]])
        new_x, components, new_G = self._make_s_plex_on_full_graph(new_x, new_G, forbidden_edges=edges_to_remove)
        if new_x is None:
            return None
        return Solution(self.instance, new_x, components, new_G)

    def _make_s_plex_on_full_graph(self, new_x, new_G, forbidden_edges=None):
        if forbidden_edges is None:
            forbidden_edges = set()

        components = list(nx.connected_components(new_G))
        for comp in components:
            new_x, new_G = self._make_s_plex_on_component(new_x, comp, forbidden_edges)
            if new_x is None:
                return None, None, None

            new_G.add_edges_from([k for k, v in new_x.items() if v])
            assert is_s_plex(self.instance.s, nx.subgraph(new_G, comp)), f'Component {comp} is not an s-plex!'
        return new_x, components, new_G

    def _make_s_plex_on_component(self, new_x, comp, forbidden_edges):
        G_comp = nx.Graph()
        G_comp.add_nodes_from(comp)
        for n in comp:
            for m in comp:
                if n < m and new_x[n, m]:
                    G_comp.add_edge(n, m)
        while not is_s_plex(self.instance.s, G_comp):
            min_degree_node = min(G_comp.nodes(), key=lambda x: G_comp.degree(x))
            # candidates = [node for node in G_comp.nodes() if node != min_degree_node and
            #               (min_degree_node, node) not in G_comp.edges() and
            #               (min_degree_node, node) not in forbidden_edges]
            candidates = G_comp.nodes() - {min_degree_node} - set(G_comp.neighbors(min_degree_node)) - \
                         set(forbidden_edges)
            if not candidates:
                return None, None
            target_node = min(candidates, key=lambda x: self.instance.weight[min_degree_node, x])
            G_comp.add_edge(min_degree_node, target_node)
            new_x[min_degree_node, target_node] = 1

        return new_x, G_comp

    def swap_nodes_neighborhood(self, config, n):
        n = int(n)
        # Iterate over all choices of n nodes
        for nodes in combinations(self.instance.nodes, n):
            sol = self.apply_swap_nodes(nodes)
            yield sol

    def swap_nodes_random_neighbor(self, config, n):
        n = int(n)
        nodes = random.sample(self.instance.nodes, n)
        return self.apply_swap_nodes(nodes)

    def apply_swap_nodes(self, nodes):
        # We will switch n_1 -> n_2, ... n_n -> n_1
        new_x = self.x.copy()
        for a, b in zip(nodes, nodes[1:] + nodes[:1]):
            new_x = self._swap_nodes(new_x, a, b)
        new_G = nx.Graph()
        new_G.add_nodes_from(self.instance.nodes)
        new_G.add_edges_from([e for e in self.instance.edges if new_x[e]])
        new_x, components, new_G = self._make_s_plex_on_full_graph(new_x, new_G)

        return Solution(self.instance, new_x, components, new_G)

    def _swap_nodes(self, new_x, a, b):
        G = nx.Graph()
        G.add_edges_from([k for k, v in new_x.items() if v])
        neighbors_a = list(G.neighbors(a))
        neighbors_b = list(G.neighbors(b))

        n_a = len(neighbors_a)
        n_b = len(neighbors_b)

        G.remove_edges_from([(a, n) for n in neighbors_a])
        G.remove_edges_from([(b, n) for n in neighbors_b])

        G.add_edges_from([(a, n) for n in neighbors_b])
        G.add_edges_from([(b, n) for n in neighbors_a])

        assert len(list(G.neighbors(a))) == n_b, f'Number of neighbors of {a} is not {n_b}'
        assert len(list(G.neighbors(b))) == n_a, f'Number of neighbors of {b} is not {n_a}'

        new_x = {e: 1 if e in G.edges() else 0 for e in self.instance.edges}
        return new_x

    def move_nodes_neighborhood(self, config, n):
        n = int(n)
        # Iterate over all pairs of components
        for A, B in combinations(list(range(len(self.components))) + [-1], 2):
            if A == -1:
                continue

            sol = self.solution_from_move_nodes(A, B, n)

            yield sol

    def move_nodes_random_neighbor(self, config, n):
        n = int(n)
        A = random.choice(list(range(len(self.components))))
        B = random.choice(list(range(len(self.components))) + [-1])
        sol = self.solution_from_move_nodes(A, B, n)
        return sol

    def solution_from_move_nodes(self, A, B, n):
        A = self.components[A]
        B = self.components[B] if B != -1 else set()
        # Randomly choose n nodes from A (or the complete component if n is larger)
        nodes = random.sample(A, min(n, len(A)))
        if type(nodes) == int:
            nodes = [nodes]
        # We will move n_1 -> B, ... n_n -> B
        new_x = self.x.copy()
        new_G = self.graph.copy()
        subgraph_to_move = new_G.subgraph(nodes).copy()
        edges_to_remove = [(u, v) for u, v in new_G.edges(A) if v not in A]
        new_G.remove_edges_from(edges_to_remove)
        new_G.add_edges_from([(u, v) for u, v in subgraph_to_move.edges() if v in B])
        if B:
            # If B is not empty, connect one element to the nodes in B
            new_G.add_edge(nodes[0], random.choice(list(B)))
        new_x, components, new_G = self._make_s_plex_on_full_graph(new_x, new_G)
        sol = Solution(self.instance, new_x, nx.connected_components(new_G), new_G)
        return sol

    def _move_node(self, new_x, new_G, node, B):
        # Disconnect node
        neighbors = list(new_G.neighbors(node))
        for neighbor in neighbors:
            new_x[node, neighbor] = 0
            new_x[neighbor, node] = 0
            if node < neighbor:
                new_G.remove_edge(node, neighbor)
            else:
                new_G.remove_edge(neighbor, node)
        # Connect to random node in B if B is not empty
        if B:
            new_neighbor = random.choice(list(B))
            new_x[node, new_neighbor] = 1
            new_x[new_neighbor, node] = 1
            new_G.add_edge(node, new_neighbor)

        return new_x, new_G
