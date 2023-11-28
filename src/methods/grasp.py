from src.config import Config
from src.solution import Solution
from src.utils import Instance
from src.methods.local_search import LocalSearch
from src.methods.construction_heuristic import ConstructionHeuristic


class GRASP:
    def __init__(self, config: Config, params=None):
        self._solution = None
        self._instance = None
        self._config = config
        self._params = params

    def solve(self, instance: Instance, solution: Solution) -> Solution:
        """
        Solve the instance using the variable neighborhood descent
        :param instance: instance to solve
        :param solution: initial solution to improve
        :return: the best solution after local search procedure
        """
        self._instance = instance
        self._solution = solution
        self._best_cost = solution.evaluate()
        #
        # config.method_params['local_search']['neighborhood'] = _par[0]
        # config.method_params['local_search']['step_function'] = _par[1]

        print(f'\n\n--- GRASP ---')
        print(f'    Max iteration: {self._params["n_iter"]}')

        sol3 = self._solution.copy()
        for i in range(self._params['n_iter']):
            if i%1 == 0:
                print(f'Iteration {i+1} / {self._params["n_iter"]}')
            # Generate new random solution
            sol2 = ConstructionHeuristic({"randomized": True}).solve(self._instance)
            # Apply local search
            sol2 = LocalSearch(self._config, params=self._config.method_params['local_search']).solve(self._instance, sol2)

            if sol2 == sol3:
                print('Same solution as last iteration, skipping')
            else:
                if sol2.evaluate() < self._best_cost:
                    print(f'New best solution found: {sol2.evaluate()}')
                    self._best_cost = sol2.evaluate()
                    self._solution = sol2
            sol3 = sol2.copy()

        return self._solution
