import math
import random

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from src.config import Config
from src.solution import Solution
from src.utils import Instance

config = Config()


class AntColony:
    """
    Concept:Bipartite graph with every node link to every component
    """
    def __init__(self, config: Config, params=None):
        self._solution = None
        self._instance = None
        self._config = config
        self._params = params

        self._max_components = self._params['max_components']
        self._ants_number = self._params['ants_number']
        self._local_pheromone_update = self._params['local_pheromone_update']
        self._pheromone_initial_value = self._params['pheromone_initial_value']
        self._n_iter = self._params['n_iter']
        self._evaporation_rate = self._params['evaporation_rate']
        self._strategy = self._params['strategy']
        self._wheel_selection_rnd = self._params['wheel_selection_rnd']
        self._informed = self._params['informed']

        self._graph = None
        self._best_solution = None
        self._best_solution_score = None
        self._pheromone_matrix = None
        self._ants_list = []


    def solve(self, instance: Instance, solution: Solution) -> Solution:
        self._instance = instance
        self._solution = solution.copy()
        self._best_solution = solution.copy()
        self._best_solution_score = solution.evaluate()

        if self._max_components == -1:
            self._max_components = len(self._solution.components)

        self.set_ants()

        # assert the number of component in the solution is less than the max number of component
        assert len(self._solution.components) <= self._max_components, 'The number of components in the solution is greater than the max number of components please allow more freedom in max_components'

        self._graph = self.set_graph()
        self._pheromone_matrix = self._initialize_pheromone_matrix()
        self._pheromone_matrix = self._set_first_ph_matrix(self._informed)
        plt.imshow(self._pheromone_matrix, cmap='hot', interpolation='nearest')
        plt.show()

        scores = []
        print(f'Initial cost: {self._best_solution_score}')
        for iteration in range(self._n_iter):
            if iteration%20 == 0 and do_print: print(f'Iteration in Ant colony method solve: {iteration}, self._best_solution_score:{self._best_solution_score}')
            ant: Ant
            for i, ant in enumerate(self._ants_list):
                # force an ant to the construction heuristic if iteration == 0
                if i == 0: # make 1st ant greedy
                    score = ant.create_solution(self._graph, self._pheromone_matrix, greedy=True).evaluate()
                    scores.append(score)
                else:
                    score = ant.create_solution(self._graph, self._pheromone_matrix).evaluate()
                    scores.append(score)
                if score < self._best_solution_score:
                    self._best_solution = ant._solution
                    self._best_solution_score = score

            self._update_pheromone_matrix(strategy=self._strategy)
            if iteration%10 == 0 and do_print:
                print(f'sum ph = {self._pheromone_matrix.sum()}')
                # plot the pheromone matrix as a heat map
                plt.imshow(self._pheromone_matrix, cmap='hot', interpolation='nearest')
                plt.show()
        # plot the scores
        plt.plot(scores)
        # plt.ylim(7000, 10000)
        plt.show()
        scores_sub100 = scores[100:]
        plt.plot(scores_sub100)
        # plt.ylim(7000, 10000)
        plt.show()
        return self._best_solution

    def _update_pheromone_matrix(self, strategy):
        # Update
        if strategy == 'best_ant':
            sorted_ants_list = sorted(self._ants_list, key=lambda ant: ant._solution.evaluate())
            assert sorted_ants_list[0]._solution.evaluate() < sorted_ants_list[-1]._solution.evaluate() #TODO remove
            best_ant = sorted_ants_list[0]
            self._pheromone_matrix = best_ant._pheromone_matrix
        if strategy == 'min_max':
            sorted_ants_list = sorted(self._ants_list, key=lambda ant: ant._solution.evaluate())
            # map the deposit value from 0 to local_pheromone_update regarding the score
            best_score_it = sorted_ants_list[0]._solution.evaluate()
            score_diff = sorted_ants_list[-1]._solution.evaluate() - best_score_it
            if score_diff == 0: score_diff = 1
            for ant in sorted_ants_list:
                self._pheromone_matrix += ant._pheromone_matrix.astype(float) * self._local_pheromone_update * (1-((ant._solution.evaluate()-best_score_it)/score_diff))
        # Evaporation
        self._pheromone_matrix *= (1 - self._evaporation_rate)

    def _initialize_pheromone_matrix(self):
        return np.full((len(self._instance.nodes), self._max_components), self._pheromone_initial_value)

    def set_graph(self):
        """
        Creates a bipartite graph from the instance: 1 node per component and 1 node per node with edges between every node and every component
        """
        graph = nx.Graph()
        graph.add_nodes_from(self._instance.nodes, bipartite=0)
        component_nodes = [f'c{i}' for i in range(self._max_components)]
        graph.add_nodes_from(component_nodes, bipartite=1)
        for node in self._instance.nodes:
            for component in component_nodes:
                graph.add_edge(node, component)
        return graph

    def set_ants(self):
        for i in range(self._ants_number):
            self._ants_list.append(Ant(self._local_pheromone_update, self._instance, wheel_selection_rnd=self._wheel_selection_rnd))

    def _set_first_ph_matrix(self, informed):
        """
        Set the first pheromone matrix with the value of the first solution
        """
        ph_matrix = self._pheromone_matrix.copy()
        for i, component in enumerate(self._solution.components):
            for node in component:
                ph_matrix[node-1][i] += informed
        return ph_matrix.astype(float)


class Ant:
    def __init__(self, local_pheromone_update, instance, wheel_selection_rnd=False):
        self._local_pheromone_update = local_pheromone_update
        self._instance = instance
        self._wheel_selection_rnd = wheel_selection_rnd
        self._pheromone_matrix = None
        self._solution = None

    def create_solution(self, graph,
                        pheromone_matrix, greedy=False):
        """
        Creates a solution for the ant
        """
        self._pheromone_matrix = pheromone_matrix.copy()
        temp_solution = [[] for i in range(self._pheromone_matrix.shape[1])]  # empty list of length max_components
        node_to_visit = list(self._instance.nodes)

        while len(node_to_visit) != 0:
            node = np.random.choice(node_to_visit)
            node_to_visit.remove(node)
            indice = node -1
            choosen_component = self._select_an_edge(graph, pheromone_matrix, node, greedy)
            self._pheromone_matrix[indice][choosen_component] += self._local_pheromone_update
            temp_solution[choosen_component].append(node)

        return self._make_solution(temp_solution)


    def _select_an_edge(self, graph, pheromone_matrix, node, greedy):
        """
        Selects an edge to a component for the ant
        Local information is null
        """
        indice = node -1
        if greedy:
            return np.argmax(pheromone_matrix[indice])
        proba_for_wheel = pheromone_matrix[indice] / pheromone_matrix[indice].sum()
        return self._wheel_selection(proba_for_wheel, randomize_wheel=self._wheel_selection_rnd)

    def _wheel_selection(self, probabilities, randomize_wheel=False):
        # create rand_num randomly between 0 and 1
        if randomize_wheel:
            # map proba to [0.1,1]
            probabilities = np.array([(x - min(probabilities)) * 0.99 / (max(probabilities) - min(probabilities)) + 0.01 for x in probabilities])
        rand_num = random.uniform(0, probabilities.sum())
        cumulative_prob = 0.0
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_num <= cumulative_prob:
                return i  # Return the index of the selected element

    def _make_solution(self, temp_solution):
        # Remove empty component (and push everything to the left)
        while [] in temp_solution: temp_solution.remove([])

        graph_components = [nx.complete_graph(comp) for comp in temp_solution]

        graph = nx.Graph()
        graph.add_nodes_from(self._instance.nodes)
        for i in range(len(graph_components)):
            graph_components[i] = delete_edges(graph_components[i], self._instance)
            graph.add_edges_from(graph_components[i].edges)

        x = {e: 0 for e in self._instance.edge_info.keys()}
        for i in range(len(graph_components)):
            for e in graph_components[i].edges:
                x[min(e), max(e)] = 1
        # make a graph from the grpah components
        self._solution = Solution(self._instance, x, graph=graph)
        assert self._solution.is_feasible() #TODO remove later for efficiency
        return self._solution.copy()

def delete_edges(G: nx.Graph, instance: Instance) -> nx.Graph:
    """
    While is an s-plex, delete the edges with the highest weight.
    We never delete edges that are in the instance.

    :param G: graph
    :param instance: instance
    :return: G with edges deleted
    """
    edges_to_delete = sorted(
        [e for e in G.edges if e not in instance.edges_in_instance],
        key=lambda x: -instance.weight[x]
    )
    for e in edges_to_delete:
        if G.degree(e[0]) > G.number_of_nodes() - instance.s and G.degree(e[1]) > G.number_of_nodes() - instance.s:
            G.remove_edge(*e)

    return G