from abc import abstractmethod
from typing import Generator

import networkx as nx


class Instance:
    """
    Class that represents an instance of the S-Plex Editing problem
    """

    def __init__(self, s, n, m, edge_info, name):
        self.s = int(s)
        self.n = int(n)
        self.m = int(m)
        self.edge_info = edge_info

        self.name = name

        # Create additional structures needed
        self.nodes = set(range(1, self.n + 1))
        self.edges = set(self.edge_info.keys())
        self.weight = {e: self.edge_info[e][1] for e in self.edges}
        self.weight = {**self.weight, **{(j, i): self.weight[(i, j)] for (i, j) in self.edges}}
        self.in_instance = {e: self.edge_info[e][0] for e in self.edges}  # 1 if edge is in instance, 0 otherwise
        self.in_instance = {**self.in_instance, **{(j, i): self.in_instance[(i, j)] for (i, j) in self.edges}}
        self.edges_in_instance = {e for e in self.edges if self.in_instance[e] == 1}  # List of edges in instance
        self.edges_in_instance = self.edges_in_instance.union({(j, i) for (i, j) in self.edges_in_instance})

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges_in_instance)

    def __repr__(self):
        return f'  {self.name}:\n   - {self.n} nodes\n   - {self.m} edges'


def is_s_plex(s, G: nx.Graph):
    """"
    Check if a connected graph G is a k-plex
    :param s: size of the plex (k-plex)
    :param G: nx graph
    :return: True if G is a k-plex, False otherwise
    """
    return min(dict(G.degree).values()) >= len(G.nodes()) - s


class AbstractSol:
    def __init__(self):
        pass

    @classmethod
    def is_better_obj(cls, obj1: float, obj2: float) -> bool:
        """Return True if the obj1 is a better objective value than obj2."""
        return obj1 < obj2

    @classmethod
    def is_worse_obj(cls, obj1: float, obj2: float) -> bool:
        """Return True if obj1 is a worse objective value than obj2."""
        return obj1 > obj2

    def check(self):
        assert self.is_feasible(), f'Solution is not feasible!'

    def is_better(self, other: 'AbstractSol') -> bool:
        """Return True if the current solution is better in terms of the objective function than the other."""
        return self.obj() < other.obj()

    def is_worse(self, other: 'AbstractSol') -> bool:
        """Return True if the current solution is worse in terms of the objective function than the other."""
        return self.obj() > other.obj()

    @abstractmethod
    def obj(self) -> float:
        """Return the objective value of the solution."""
        pass

    @abstractmethod
    def is_feasible(self) -> bool:
        """Return True if the solution is feasible."""
        pass

    @abstractmethod
    def update_solution(self, other: 'AbstractSol'):
        """Update the current solution with the other solution."""
        pass

    @abstractmethod
    def get_random_neighbor(self, type: str, neighborhood_config: dict) -> 'Solution':
        """
        Generate a random neighbor of the current solution
        :param type: type of neighborhood
        :param neighborhood_config: dictionary with neighborhood parameters
        :return: random neighbor
        """
        pass

    @abstractmethod
    def evaluate(self) -> float:
        """
        Evaluate the current solution
        :return: objective value of the solution
        """
        pass

    @abstractmethod
    def generate_neighborhood(self, type: str, neighborhood_config: dict) -> Generator['AbstractSol', None, None]:
        """
        Generate a neighborhood of the current solution
        :param type: type of neighborhood
        :param neighborhood_config: dictionary with neighborhood parameters
        :return: list of neighbors
        """
        pass