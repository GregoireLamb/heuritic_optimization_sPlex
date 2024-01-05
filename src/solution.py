import os
from itertools import combinations
import random
from typing import Set, List, Generator

import networkx as nx
from matplotlib import pyplot as plt

from src.config import Config
from src.utils import Instance, is_s_plex, AbstractSol


class Solution(AbstractSol):
    def __init__(self, instance: Instance, x: dict, components: list = None, graph: nx.Graph = None, obj: float = None):
        super().__init__()
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
        self.obj_val = obj  # Objective value of the solution. Can be None if not computed

    def _get_components(self):
        """
        Compute the components of the solution
        """
        return list(nx.connected_components(self.graph))

    def copy(self):
        return Solution(self.instance, self.x.copy(), self.components.copy(), self.graph.copy(), self.obj_val)

    def copy_from(self, other: 'Solution'):
        self.x = other.x.copy()
        self.components = other.components.copy()
        self.graph = other.graph.copy()
        self.obj_val = other.obj_val

    def __repr__(self):
        nx.draw(self.graph, with_labels=True)
        plt.show()
        return f"Solution cost: {self.evaluate()}, number of connected components: {len(self.components)}"

    def evaluate(self):
        """
        Evaluate the solution
        """
        if self.obj_val is not None:
            return self.obj_val
        obj = 0
        for edge in self.instance.edges:
            if self.x[edge] != self.instance.in_instance[edge]:
                obj += self.instance.weight[edge]
        self.obj_val = obj
        return obj

    obj = evaluate

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
            sol, added_edges = self.remove_edges(edges_to_remove)
            if sol is None:
                continue

            sol.obj_val = self.obj_val + self.compute_delta(added_edges=added_edges, removed_edges=edges_to_remove)
            yield sol

    def kflips_random_neighbor(self, config, k):
        k = int(k)
        edges = {e for e in self.instance.edges if self.x[e]}
        if not config['remove_edge_in_instance_allowed']:
            edges -= self.instance.edges_in_instance

        while True:
            edges_to_remove = random.sample(list(edges), k)
            sol, _ = self.remove_edges(edges_to_remove)
            # Check if feasible solution was found
            if sol is not None:
                return sol

    def remove_edges(self, edges_to_remove):
        # Remove the k edges
        edges_to_remove = set(edges_to_remove).union({(v, u) for u, v in edges_to_remove})
        new_x = {e: 0 if e in edges_to_remove else self.x[e] for e in self.instance.edges} # edges from sol - removed edges
        new_G = nx.Graph()
        new_G.add_nodes_from(self.instance.nodes)
        new_G.add_edges_from([e for e in self.instance.edges if new_x[e]])
        new_x, components, new_G, created_edges = self._make_s_plex_on_full_graph(new_x, new_G, forbidden_edges=edges_to_remove) # created_edges is the set of edges newly created, no edges have been removed during the call
        if new_x is None:
            return None, None
        return Solution(self.instance, new_x, components, new_G), created_edges

    def _make_s_plex_on_full_graph(self, new_x, new_G, forbidden_edges=None):
        if forbidden_edges is None:
            forbidden_edges = set()

        components = list(nx.connected_components(new_G))
        new_edges = set()
        for comp in components:
            new_x, _, new_edges_comp = self._make_s_plex_on_component(new_x, comp, forbidden_edges)
            if new_x is None:
                return None, None, None, None
            new_edges = new_edges.union(new_edges_comp)

            new_G.add_edges_from([k for k, v in new_x.items() if v])
            # assert is_s_plex(self.instance.s, nx.subgraph(new_G, comp)), f'Component {comp} is not an s-plex!'
        return new_x, components, new_G, new_edges

    def _make_s_plex_on_component(self, new_x, comp, forbidden_edges):
        """
        make the component an s-plex
        :param new_x: new self._x
        :param comp: component in a nx Graph format ? sur about format ?
        :param forbidden_edges: list of edges that cannot be added
        """
        new_edges = set()
        G_comp = nx.Graph()
        G_comp.add_nodes_from(comp)
        for n in comp:
            for m in comp:
                if n < m and new_x[n, m]:
                    G_comp.add_edge(n, m)
        while not is_s_plex(self.instance.s, G_comp):
            min_degree_node = min(G_comp.nodes(), key=lambda x: G_comp.degree(x))  # < |S|-s
            candidates = [node for node in G_comp.nodes() if node != min_degree_node and
                          (min_degree_node, node) not in G_comp.edges() and  # make sur a,b is not possible if b,a is forbidden
                          (min_degree_node, node) not in forbidden_edges]
            if not candidates: # re allow forbidden edges if they are mandatory
                candidates = [node for node in G_comp.nodes() if node != min_degree_node and
                          (min_degree_node, node) not in G_comp.edges()]
                # return None, None, None
            target_node = min(candidates, key=lambda x: self.instance.weight[min_degree_node, x])
            new_edge = (min(min_degree_node, target_node), max(min_degree_node, target_node))
            new_edges.add(new_edge)
            G_comp.add_edge(new_edge[0], new_edge[1])
            new_x[new_edge] = 1

        return new_x, G_comp, new_edges

    def compute_add_rm_edges(self, new_x):
        add = {e for e in self.instance.edges if new_x[e] and not self.x[e]}
        remove = {e for e in self.instance.edges if not new_x[e] and self.x[e]}
        return add, remove

    def swap_nodes_neighborhood(self, config, n):
        n = int(n)
        # Iterate over all choices of n nodes
        for nodes in combinations(self.instance.nodes, n):
            sol = self.apply_swap_nodes(nodes)
            add, remove = self.compute_add_rm_edges(sol.x)
            delta = self.compute_delta(add, remove)
            sol.obj_val = self.obj_val + delta
            yield sol

    def swap_nodes_random_neighbor(self, config, n):
        n = int(n)
        nodes = random.sample(list(self.instance.nodes), n)
        return self.apply_swap_nodes(nodes)

    def apply_swap_nodes(self, nodes):
        # We will switch n_1 -> n_2, ... n_n -> n_1
        new_x = self.x.copy()
        for a, b in zip(nodes, nodes[1:] + nodes[:1]):
            new_x = self._swap_nodes(new_x, a, b)
        new_G = nx.Graph()
        new_G.add_nodes_from(self.instance.nodes)
        new_G.add_edges_from([e for e in self.instance.edges if new_x[e]])
        new_x, components, new_G, _ = self._make_s_plex_on_full_graph(new_x, new_G)

        return Solution(self.instance, new_x, components, new_G)

    def _swap_nodes(self, new_x, a, b):
        G = nx.Graph()
        G.add_nodes_from(self.instance.nodes)
        G.add_edges_from([k for k, v in new_x.items() if v])
        neighbors_a = list(G.neighbors(a))
        neighbors_b = list(G.neighbors(b))

        n_a = len(neighbors_a)
        n_b = len(neighbors_b)

        G.remove_edges_from([(a, n) for n in neighbors_a])
        G.remove_edges_from([(b, n) for n in neighbors_b])

        G.add_edges_from([(a, n) for n in neighbors_b])
        G.add_edges_from([(b, n) for n in neighbors_a])

        # assert len(list(G.neighbors(a))) == n_b, f'Number of neighbors of {a} is not {n_b}'
        # assert len(list(G.neighbors(b))) == n_a, f'Number of neighbors of {b} is not {n_a}'

        new_x = {e: 1 if e in G.edges() else 0 for e in self.instance.edges}
        return new_x

    def move_nodes_neighborhood(self, config, n):
        n = int(n)
        # Iterate over all pairs of components
        comp_indexes = list(range(len(self.components))) + [-1]
        for A, B in combinations(comp_indexes, 2):
            if A == -1:
                continue

            sol = self.solution_from_move_nodes(A, B, n)

            yield sol

    def move_nodes_random_neighbor(self, config, n):
        n = int(n)
        A = random.choice(list(range(len(self.components))))
        B = random.choice(list(range(len(self.components))) + [-1])
        sol = self.solution_from_move_nodes(A, B, n)
        add, remove = self.compute_add_rm_edges(sol.x)
        delta = self.compute_delta(add, remove)

        sol.obj_val = self.obj_val + delta
        return sol

    def solution_from_move_nodes(self, A, B, n):
        # print(f'Moving {n} nodes in solution from move nodes')
        A = self.components[A]
        B = self.components[B] if B != -1 else set()
        # Randomly choose n nodes from A (or the complete component if n is larger)
        nodes = random.sample(list(A), min(n, len(A)))
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
        new_x, components, new_G, _ = self._make_s_plex_on_full_graph(new_x, new_G)
        sol = Solution(self.instance, new_x, list(nx.connected_components(new_G)), new_G)
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

    def update_solution(self, solution: 'Solution'):
        self.x = solution.x
        self.graph = solution.graph
        self.components = solution.components

    def compute_delta(self, added_edges: Set, removed_edges: Set):
        # remark: cutting or uncutting an edge have the same absolute coste
        delta = sum([self.instance.weight[edge] * -1 * ((2 * self.instance.in_instance[edge]) -1) for edge in added_edges])
        return delta + sum([self.instance.weight[edge] * ((2 * self.instance.in_instance[edge]) -1) for edge in removed_edges])
