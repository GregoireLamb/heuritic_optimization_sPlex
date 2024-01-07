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
    # TODO max = f(n)
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
        self.set_ants()
        do_print = False

        # assert the number of component in the solution is less than the max number of component
        assert len(self._solution.components) <= self._max_components, 'The number of components in the solution is greater than the max number of components please allow more freedom in max_components'

        self._graph = self.set_graph()
        self._pheromone_matrix = self._initialize_pheromone_matrix()
        self._pheromone_matrix = self._set_first_ph_matrix() #TODO reinstall ???
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
            if iteration%50 == 0 and do_print:
                print(f'sum ph = {self._pheromone_matrix.sum()}')
                # plot the pheromone matrix as a heat map
                plt.imshow(self._pheromone_matrix, cmap='hot', interpolation='nearest')
                plt.show()
        # plot the scores
        plt.plot(scores)
        plt.ylim(7000, 10000)
        plt.show()
        return self._best_solution

    def _update_pheromone_matrix(self, strategy):
        # Update
        if strategy == 'best_ant':
            sorted_ants_list = sorted(self._ants_list, key=lambda ant: ant._solution.evaluate())
            assert sorted_ants_list[0]._solution.evaluate < sorted_ants_list[-1]._solution.evaluate() #TODO remove
            best_ant = sorted_ants_list[0]
            self._pheromone_matrix = best_ant._pheromone_matrix
        if strategy == 'min_max':
            sorted_ants_list = sorted(self._ants_list, key=lambda ant: ant._solution.evaluate())
            # map the deposit value from 0 to local_pheromone_update regarding the score
            best_score_it = sorted_ants_list[0]._solution.evaluate()
            score_diff = sorted_ants_list[-1]._solution.evaluate() - best_score_it
            for ant in sorted_ants_list:
                self._pheromone_matrix += ant._pheromone_matrix * self._local_pheromone_update * (1-((ant._solution.evaluate()-best_score_it)/score_diff))
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
            self._ants_list.append(Ant(self._local_pheromone_update, self._instance))

    def _set_first_ph_matrix(self):
        """
        Set the first pheromone matrix with the value of the first solution
        """
        ph_matrix = self._pheromone_matrix.copy()
        for i, component in enumerate(self._solution.components):
            for node in component:
                ph_matrix[node-1][i] += self._local_pheromone_update
        return ph_matrix


class Ant:
    def __init__(self, local_pheromone_update, instance):
        self._local_pheromone_update = local_pheromone_update
        self._instance = instance
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
            node = np.random.choice(node_to_visit) #TODO add for diversification
            # node = node_to_visit[0] #TODO remove
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
        return self._wheel_selection(proba_for_wheel)

    def _wheel_selection(self, probabilities):
        # create rand_num randomly between 0 and 1
        rand_num = random.uniform(0, probabilities.sum())
        cumulative_prob = 0.0
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_num <= cumulative_prob:
                return i  # Return the index of the selected element

    def _make_solution(self, temp_solution):
        # Remove empty component (and push everything to the left)
        while [] in temp_solution: temp_solution.remove([])
        solution = Solution(self._instance, {e: 0 for e in self._instance.edge_info.keys()}, temp_solution)

        for component in solution.components:#TODO info instance.edges is fuck up
            if len(component) == 0:
                continue
            sub_graph = nx.Graph()
            sub_graph.add_nodes_from(component)
            new_x, _, new_edges = solution._make_s_plex_on_component(solution.x, sub_graph, [])
            solution.x = new_x
            for new_edge in new_edges:
                solution.graph.add_edge(new_edge[0], new_edge[1])

        self._solution = solution
        assert solution.is_feasible() #TODO remove later for efficiency
        return solution
