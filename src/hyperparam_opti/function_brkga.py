import os
import tempfile
import subprocess
from typing import Union

import numpy as np
import logging

from ConfigSpace import Configuration, ConfigurationSpace
from hyperparam_config import Config
from src.config import Config as BasicConfig
from instances import Instances
from function_wrapper import TargetFunctionWrapper
from src.methods.brk_ga import BRKGA


class SolverLogParser:
    pass


class FunctionBRKGA(TargetFunctionWrapper):
    def __init__(self, config: Config, instances: Instances):
        super().__init__(config=config)

        self._instances = instances

    def evaluate(self, configuration: Configuration, instance: str = None, seed: float = None) -> float:
        """
        Evaluate the function with the given configuration
        """
        if instance is None:
            instance = self._instances.instances[0]

        instance_object = self._instances.get_instance_object(instance)
        instance_info = self._instances.get_instance_info(instance)
        initial_solution = instance_info['construction_heuristic_sol']
        initial_cost = instance_info['construction_heuristic_cost']

        basic_config = BasicConfig()
        ga_parameters = basic_config.this_method_params

        # Set parameters according to the configuration
        for param, value in configuration.items():
            ga_parameters[param] = value

        logging.info(f'Executing with configuration {ga_parameters}')
        method = BRKGA(basic_config, params=ga_parameters)
        solution = method.solve(instance_object, initial_solution)
        objective = solution.evaluate() / initial_cost
        logging.info(f'Solver finished with target objective {objective}')

        return objective
