import time

from src.config import Config
from src.solution import Solution
from src.utils import Instance


class LocalSearch:
    def __init__(self, config: Config, params=None):
        self._config = config

        self._step_function = params['step_function']
        assert self._step_function in ['best_improvement', 'first_improvement', 'random', 'best', 'first'], \
            'Step function must be one of [best_improvement, first_improvement, random]'
        self._neighborhood = params['neighborhood']
        if self._step_function in ['best', 'first']:
            self._step_function += '_improvement'

        self._time_limit = params['time_limit']
        self._it_limit = params['it_limit']
        self.silent = params['silent']

        self._instance = None
        self._best_found_solution = None
        self._solution = None

    def solve(self, instance: Instance, solution: Solution) -> Solution:
        """
        Solve the instance using the deterministic construction heuristic
        :param instance: instance to solve
        :param solution: initial solution to improve
        :return: the best solution after local search procedure
        """
        self._instance = instance
        self._solution = solution
        self._best_found_solution = solution

        if not self.silent:
            print(f'\n\n--- Local Search ---')
            print(f'    Step function: {self._step_function}')
            print(f'    Neighborhood: {self._neighborhood}')
            print(f'    Time limit: {self._time_limit}')
            print(f'    Iteration limit: {self._it_limit}')
            print(f'    Initial solution cost: {self._solution.evaluate()}\n\n')

        start_time = time.time()
        it = 0
        while it < self._it_limit:
            it += 1
            # Locql i,prove,nt
            if not self.silent:
                print(f'Iteration {it}. Current incumbent cost: {self._solution.evaluate()}')
            if time.time() - start_time > self._time_limit:
                break

            if self._step_function == 'random':
                self._solution = self._solution.get_random_neighbor(self._neighborhood,
                                                                    self._config.neighborhood_params)
                return self._solution

            best_neighbor = self._solution
            best_cost = self._solution.evaluate()
            explored_neighbors = 0
            for neighbor in self._solution.generate_neighborhood(self._neighborhood, self._config.neighborhood_params):
                explored_neighbors += 1
                if best_neighbor is None:
                    best_neighbor = neighbor
                    best_cost = neighbor.evaluate()
                if neighbor.evaluate() < best_cost:
                    best_neighbor = neighbor
                    best_cost = neighbor.evaluate()
                    self._best_found_solution = best_neighbor
                    if self._step_function == 'first_improvement':
                        break
                if time.time() - start_time > self._time_limit:
                    break
            if not self.silent:
                print(f'Explored {explored_neighbors} neighbors')

        return self._solution if self._best_found_solution.evaluate() > self._solution.evaluate() \
            else self._best_found_solution
