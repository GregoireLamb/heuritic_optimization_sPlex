from src.config import Config
from src.solution import Solution
from src.utils import Instance
from src.methods.local_search import LocalSearch
from src.methods.construction_heuristic import ConstructionHeuristic


class GRASP:
    def __init__(self, config: Config, params=None):
        self._best_cost = None
        self._solution = None
        self._instance = None
        self._config = config
        self._params = params

    def solve(self, instance: Instance) -> Solution:
        """
        Solve the instance using the variable neighborhood descent
        :param instance: instance to solve
        :return: the best solution after local search procedure
        """
        self._instance = instance

        print(f'\n\n--- GRASP ---')
        print(f'    Max iteration: {self._params["n_iter"]}')

        for i in range(self._params['n_iter']):
            if i % 10 == 0:
                print(f'Iteration {i+1} / {self._params["n_iter"]}')
            # Generate new random solution
            assert self._config.method_params['construction_heuristic']['randomized'], \
                'Construction heuristic must be randomized'
            generated_solution = ConstructionHeuristic(
                self._config.method_params['construction_heuristic']).solve(self._instance)
            # Apply local search
            generated_solution = LocalSearch(self._config, params=self._config.method_params['local_search']
                                             ).solve(self._instance, generated_solution)

            if self._best_cost is None or generated_solution.evaluate() < self._best_cost:
                print(f'New best solution found: {generated_solution.evaluate()}')
                self._best_cost = generated_solution.evaluate()
                self._solution = generated_solution

        return self._solution