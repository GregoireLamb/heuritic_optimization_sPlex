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
        self.N = list(range(1, self.n + 1))


class Visualisation:
    def __init__(self, instance):
        self.instance = instance

    def plot_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.instance.N)
        G.add_edges_from(self.instance.edge_info.keys())
        nx.draw(G, with_labels=True)
        plt.show()
