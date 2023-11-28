from typing import Any

from pymhlib.gvns import GVNS
from pymhlib.scheduler import Method, Result

from src.config import Config
from src.solution import Solution
from src.utils import Instance
from src.methods.local_search import LocalSearch
config = Config()


def local_improve(sol: Solution, _par: Any, _res: Result):
    # Set configuration according to _par
    config.method_params['local_search']['neighborhood'] = _par[0]
    config.method_params['local_search']['step_function'] = _par[1]
    local_search = LocalSearch(config, params=config.method_params['local_search'])
    new_sol = local_search.solve(sol.instance, sol)
    sol.update_solution(new_sol)


class VND:
    def __init__(self, config: Config, params=None):
        self._solution = None
        self._instance = None
        self._config = config
        self._own_settings = params
        self._params = params

        self._meths_cs = []
        self._meths_li = [Method(f'local_improve_{neigh}_{step}', local_improve, (neigh, step)) for
                          neigh, step in self._params['meths_li']]
        self._meths_sh = [Method(f'shaking_{neigh}_{step}', local_improve, (neigh, step)) for
                          neigh, step in self._params['meths_sh']]

    def solve(self, instance: Instance, solution: Solution) -> Solution:
        """
        Solve the instance using the variable neighborhood descent
        :param instance: instance to solve
        :param solution: initial solution to improve
        :return: the best solution after local search procedure
        """
        self._instance = instance
        self._solution = solution

        gvns = GVNS(sol=solution,
                    meths_ch=self._meths_cs,
                    meths_li=self._meths_li,
                    meths_sh=self._meths_sh,
                    own_settings=self._own_settings,
                    consider_initial_sol=True)

        gvns.vnd(self._solution)

        return self._solution
