import random

from pymhlib.sa import SA

from src.config import Config
from src.solution import Solution
from src.utils import Instance


class SimulatedAnnealing:
    def __init__(self, config: Config, params=None):
        self._config = config

        self._instance = None

        self._own_settings = params

        def random_move_delta_eval(sol: Solution) -> [Solution, float]:
            neighborhood_structure = random.choice(config.this_method_params['neighborhoods'])
            random_neighbor = sol.get_random_neighbor(type=neighborhood_structure, neighborhood_config=config.neighborhood_params)
            delta = random_neighbor.evaluate() - sol.evaluate()
            return random_neighbor, delta

        def apply_neighborhood_move(sol: Solution, new_sol: Solution):
            sol.update_solution(new_sol)

        def iter_cb(iteration, sol, temperature, acceptance):
            print(f'Iteration {iteration} - Temperature {temperature} - Acceptance {acceptance}')
            print(f' -- Current solution cost: {sol.evaluate()}')

        self._meths_cs = []
        self._random_move_delta_eval = random_move_delta_eval
        self._apply_neighborhood_move = apply_neighborhood_move
        self._iter_cb = iter_cb

        # self._own_settings = None

    def solve(self, instance: Instance, solution: Solution) -> Solution:
        """
        Solve the instance using the deterministic construction heuristic
        :param instance: instance to solve
        :param solution: initial solution to improve
        :return: the best solution after local search procedure
        """
        self._instance = instance
        self._solution = solution

        sa = SA(solution, meths_ch=self._meths_cs, random_move_delta_eval=self._random_move_delta_eval,
                apply_neighborhood_move=self._apply_neighborhood_move, iter_cb=self._iter_cb, own_settings=self._own_settings,
                consider_initial_sol=True)
        sa.sa(self._solution)

        return self._solution
