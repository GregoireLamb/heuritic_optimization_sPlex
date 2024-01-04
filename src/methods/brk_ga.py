import math
import numpy as np
import random
from collections import defaultdict

import networkx as nx
import logging

from src.config import Config
from src.solution import Solution
from src.utils import Instance, is_s_plex

from pygad import GA


class BRKGA:
    """
    Biased Random Key Genetic Algorithm (BRKGA)
    """

    def __init__(self, config: Config, params: dict):
        self._config = config
        self._params = params

        self._pop_size = self._params['population_size']

        self._instance = None
        self._initial_solution = None

    def solve(self, instance: Instance, initial_solution: Solution) -> Solution:
        """
        Solves the instance using a BRKGA
        """
        self._instance = instance
        self._initial_solution = initial_solution

        logger = self._get_logger()

        ga = GA(
            initial_population=self._build_initial_population(),
            num_generations=self._params['num_generations'],
            num_parents_mating=math.ceil(self._pop_size * self._params['prop_parents_mating']),
            fitness_func=fitness_func,
            sol_per_pop=self._params['population_size'],
            num_genes=self._instance.n + 1,
            gene_type=float,
            init_range_low=0,
            init_range_high=1,
            parent_selection_type=self._params['parent_selection_type'],
            keep_parents=self._params['keep_parents'],
            crossover_type=self._params['crossover_type'],
            mutation_type=self._params['mutation_type'],
            mutation_probability=self._params['mutation_probability'],  # in [0, 1]
            mutation_by_replacement=True,
            random_mutation_min_val=0,
            random_mutation_max_val=1,
            parallel_processing=None if not self._params['parallel_processing'] else self._params[
                'parallel_processing'],
            random_seed=42,
            logger=logger,
            # save_solutions=True,
        )
        ga._instance = self._instance
        ga._params = self._params
        ga.run()
        ga.summary()
        ga.plot_fitness()
        # ga.plot_genes()

        solution_ga, solution_fitness, solution_idx = ga.best_solution()

        x = build_solution(ga, solution_ga, solution_idx)

        return Solution(self._instance, x)

    def _build_initial_population(self):
        if self._initial_solution is None:
            raise ValueError('Initial population can only be built if an initial solution is passed!')
        # The idea is to include the individual that represents the initial solution in the initial population
        ind = self._from_solution_to_individual(self._initial_solution)

        # Rest of solutions are probability random vectors of size n + 1 generated with numpy
        population = np.zeros((self._pop_size, self._instance.n + 1))

        n_random = math.ceil(self._pop_size / 2)

        for i in range(self._pop_size):
            if i < n_random:
                population[i] = np.asarray(np.random.uniform(low=0, high=1, size=self._instance.n + 1))
            elif i == n_random:
                population[i] = ind
            else:
                population[i] = ind + np.asarray(np.random.normal(loc=0, scale=0.1, size=self._instance.n + 1))
                population[i] = np.clip(population[i], a_min=0, a_max=1)

        return population
        # TODO: Additionally, half of the individuals are random and the other half are the initial solution with minor changes

    def _from_solution_to_individual(self, solution: Solution):
        """
        Converts a solution to an individual
        """
        ind = np.zeros(self._instance.n + 1)
        num_s_plexes = len(solution.components)
        ind[0] = ((num_s_plexes - 1) / self._instance.n) ** (1 / self._params['regularization_num_s_plexes'])

        calculated_num_s_plexes = max(1, math.ceil((
                    ind[0] ** self._params['regularization_num_s_plexes']) * self._instance.n))
        assert calculated_num_s_plexes == num_s_plexes, \
            f'Gene 0 is {ind[0]} -> {calculated_num_s_plexes} s-plexes but real num s-plexes is {num_s_plexes}'

        # Calculate the value of the remaining genes
        for i, comp in enumerate(solution.components):
            for node in comp:
                ind[node] = i / num_s_plexes

        return ind

    @staticmethod
    def _get_logger():
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return logger


def build_solution(ga_instance, solution, solution_idx) -> dict:
    """
    Build a solution from a BRKGA individual
    """
    instance = ga_instance._instance
    import numpy as np
    assert np.min(solution) >= 0 and np.max(solution) <= 1, 'Solution must be a vector of probabilities'

    # The first gene represents the number of s-plexes. It is a number from 1 to n
    num_s_plexes = max(1, math.ceil((solution[0] ** ga_instance._params['regularization_num_s_plexes']) * instance.n))

    components = [set() for _ in range(num_s_plexes)]
    # The remaining genes represent the assignment of each node to a s-plex.
    for i in instance.nodes:
        components[min(num_s_plexes - 1, math.floor(solution[i] * num_s_plexes))].add(i)

    components = [c for c in components if c]

    # We finally want to make each component an s-plex
    graph_components = [nx.complete_graph(comp) for comp in components]

    for i in range(len(graph_components)):
        graph_components[i] = delete_edges(graph_components[i], instance)

    x = defaultdict(lambda: 0)
    for i in range(len(graph_components)):
        for e in graph_components[i].edges:
            x[min(e), max(e)] = 1

    return x


def fitness_func(ga_instance, solution, solution_idx) -> float:
    """
    Fitness function for the BRKGA to maximize
    """
    x = build_solution(ga_instance, solution, solution_idx)
    instance = ga_instance._instance
    obj = 0
    for edge in instance.edges:
        if x[edge] != instance.in_instance[edge]:
            obj += instance.weight[edge]
    return - obj


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
