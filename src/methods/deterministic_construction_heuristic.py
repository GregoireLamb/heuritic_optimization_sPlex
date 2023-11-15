import networkx as nx

from src.utils import Solution, is_s_plex


class DeterministicConstructionHeuristic:
    def __init__(self, params=None):
        self._instance = None
        self._x = {}
        self._components = []
        self._s = -1

    def solve(self, instance) -> Solution:
        self._s = instance.s
        self._instance = instance

        self._x = {e: 0 for e in instance.edges}

        to_add = set(instance.nodes)

        while len(to_add) > 0:
            best_cost, best_component, best_node = 1e10, None, None
            for i in to_add:

                # New component
                cost = self.compute_cost_of_new_component(i)
                if cost < best_cost:
                    best_cost = cost
                    best_component = None
                    best_node = i

                # Existing component
                for index, k in enumerate(self._components):
                    cost = self.compute_insertion_cost_to_component(i, k)
                    if cost < best_cost:
                        best_cost = cost
                        best_component = index
                        best_node = i

            if best_component is None:
                self._components.append({best_node})
            else:
                self._components[best_component].add(best_node)
            to_add.remove(best_node)

        for k in self._components:
            new_edges = self.get_edges_of_s_plex(k)
            self._x.update({e: 1 for e in new_edges})

        return Solution(instance, self._x)

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
        G = self.create_component_graph(k)
        G.add_node(i)

        cost = 0
        cost = self.disconnecting_cost(cost, i, k)
        cost = self.make_s_plex(G, cost)

        return cost

    def get_edges_of_s_plex(self, k):
        G = self.create_component_graph(k)
        self.make_s_plex(G, 0)
        return G.edges()

    def make_s_plex(self, G, cost):
        # add edges till S-plex
        while not is_s_plex(self._s, G):
            min_degree_node = min(G.nodes(), key=lambda x: G.degree(x))
            best_node_to_add = -1
            best_sup_cost = 1e10
            for node in G.nodes():
                if node != min_degree_node and (min_degree_node, node) not in G.edges():
                    sup_cost = self._instance.weight[(min_degree_node, node)]
                    if sup_cost < best_sup_cost:
                        best_sup_cost = sup_cost
                        best_node_to_add = node
            G.add_edge(min_degree_node, best_node_to_add)
            cost += best_sup_cost
        return cost

    def create_component_graph(self, k):
        # Create a new graph with the vertex i and the edges of component k
        G = nx.Graph()
        G.add_nodes_from(k)
        for n in G.nodes():
            for m in G.nodes():
                if n < m and (n, m) in self._instance.edges_in_instance:
                    G.add_edge(n, m)
        return G

    def disconnecting_cost(self, cost, i, k):
        # Cost of cutting i from the rest of the existing graph
        for c in self._components:
            if c == k:
                continue
            for j in c:
                cost += self._instance.weight[(i, j)] * self._instance.in_instance[(i, j)]
        return cost
