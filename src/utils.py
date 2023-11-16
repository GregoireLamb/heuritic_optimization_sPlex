import networkx as nx
import matplotlib.pyplot as plt


class Instance:
    """
    Class that represents an instance of the S-Plex problem
    """
    def __init__(self, s, n, m, edge_info, name):
        self.s = int(s)
        self.n = int(n)
        self.m = int(m)
        self.edge_info = edge_info

        self.name = name

        # Create additional structures needed
        self.nodes = list(range(1, self.n + 1))
        self.edges = list(self.edge_info.keys())
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


class Solution:
    def __init__(self, instance: Instance, x: dict):
        self.instance = instance
        self.x = x  # Dictionary with edge as key and 1 if edge is in final graph, 0 otherwise

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


def is_s_plex(s, G: nx.Graph):
    """"
    Check if G is a k-plex
    :param s: size of the plex (k-plex)
    :param G: nx graph
    :return: True if G is a k-plex, False otherwise
    """
    return min(dict(G.degree).values()) >= len(G.nodes()) - s
