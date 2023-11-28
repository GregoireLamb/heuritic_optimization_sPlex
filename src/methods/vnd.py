from pymhlib.gvns import GVNS
from pymhlib.scheduler import Method

from src.config import Config
from src.solution import Solution
from src.utils import Instance

config = Config()



def make_method_kflip(sol: Solution, par, res):
    type = "kflips_"+str(par[0])
    return sol.generate_neighborhood(type, config.neighborhood_params.get("kflips", {}))

def make_method_movenodes(sol: Solution, par, res):
    type = "movenodes_"+str(par[0])
    return sol.generate_neighborhood(type, config.neighborhood_params.get("movenodes", {}))

def make_method_nodeswap(sol: Solution, par, res):
    type = "nodeswap_"+str(par[0])
    return sol.generate_neighborhood(type, config.neighborhood_params.get("nodeswap", {}))

class VND:
    def __init__(self, config: Config, params=None):
        self._config = config
        self._instance = None
        self._own_settings = params

        self._meths_cs = []
        self._meths_li = params['meths_li']
        self._meths_sh = []


    def solve(self, instance: Instance, solution: Solution) -> Solution:
        """
        Solve the instance using the variable neighborhood descent
        :param instance: instance to solve
        :param solution: initial solution to improve
        :return: the best solution after local search procedure
        """
        self._instance = instance
        self._solution = solution
        method_li= []

        for meth in self._meths_li:
            type, params = meth.split('_', 1)
            if type == 'kflips':
                method_li.append(Method(type, make_method_kflip, [params]))
            elif type == 'movenodes':
                method_li.append(Method(type, make_method_movenodes, [params]))
            elif type == 'nodeswap':
                method_li.append(Method(type, make_method_nodeswap, [params]))
            else:
                raise ValueError(f'Method {type} not implemented')

        gvns = GVNS(sol=solution,
                    meths_ch=self._meths_cs,
                    meths_li=method_li,
                    meths_sh=self._meths_sh,
                    own_settings=self._own_settings,
                    consider_initial_sol=True)

        gvns.vnd(self._solution)

        return self._solution
