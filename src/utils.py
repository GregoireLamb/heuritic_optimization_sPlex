import networkx as nx
import matplotlib.pyplot as plt


class Instance:
    """
    Class that represents an instance of the S-Plex problem
    """

    def __init__(self, s, n, m, edge_info):
        self.s = int(s)
        self.n = int(n)
        self.m = int(m)
        self.edge_info = edge_info

        # Create additional structures needed
        self.nodes = list(range(1, self.n + 1))
        self.edges = list(self.edge_info.keys())
        self.weight = {e: self.edge_info[e][1] for e in self.edges}
        self.in_instance = {e: self.edge_info[e][0] for e in self.edges}

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)


class Solution:
    def __init__(self, instance: Instance, x: dict):
        self.instance = instance
        self.x = x  # Dictionary with edge as key and 1 if edge is in final graph, 0 otherwise

    def evaluate(self):
        obj = 0
        for edge in self.instance.edges:
            if self.x[edge] != self.x[edge]:
                obj += self.instance.weight[edge]
        return obj


class Visualisation:
    def __init__(self, instance):
        self.instance = instance

    def plot_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.instance.nodes)
        G.add_edges_from(self.instance.edge_info.keys())
        nx.draw(G, with_labels=True)
        plt.show()
