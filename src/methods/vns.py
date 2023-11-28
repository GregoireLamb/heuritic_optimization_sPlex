from pymhlib.gvns import GVNS
from pymhlib.scheduler import Method

from src.config import Config
from src.solution import Solution
from src.utils import Instance, make_method


def make_method_kflip(sol: Solution, par, res):
    # make_method_kflip.__name__ = "kflips"
    type = "kflips_"+str(par[0])
    #TODO wrong dictionary
    return sol.generate_neighborhood(type, {"key":0})

def make_method_movenodes(sol: Solution, par, res):
    type = "movenodes_"+str(par[0])
    #TODO wrong dictionary
    return sol.generate_neighborhood(type, {"key":0})

def make_method_nodeswap(sol: Solution, par, res):
    type = "nodeswap_"+str(par[0])
    #TODO wrong dictionary
    return sol.generate_neighborhood(type, {"key":0})

class Vns:
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

        # li_list = []
        # for meth in self._meths_li:
        #     li_list.append(self._solution.generate_neighborhood(meth, self._config.neighborhood_params))
        gvns = GVNS(sol=solution,
                    meths_ch=self._meths_cs,
                    # meths_li=li_list,
                    meths_li=method_li,
                    meths_sh=self._meths_sh,
                    own_settings=self._own_settings,
                    consider_initial_sol=False)

        gvns.vnd(self._solution)

        return self._solution
