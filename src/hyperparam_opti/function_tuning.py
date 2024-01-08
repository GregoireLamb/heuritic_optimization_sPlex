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
from src.methods.ants_colony import AntColony


class SolverLogParser:
    pass


class FunctionTuning(TargetFunctionWrapper):
    def __init__(self, config: Config, instances: Instances, model_to_tune=BRKGA):
        super().__init__(config=config)

        self._model_to_tune = model_to_tune
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
        model_parameters = basic_config.this_method_params

        # Set parameters according to the configuration
        for param, value in configuration.items():
            model_parameters[param] = value

        logging.info(f'Executing instance {instance} with model {self.model_to_tune} configuration {model_parameters}')
        if self.model_to_tune == BRKGA:
            method = BRKGA(basic_config, params=model_parameters)
        elif self.model_to_tune == AntColony:
            method = AntColony(basic_config, params=model_parameters)
        else:
            raise ValueError(f'Model {self.model_to_tune} not implemented')
        solution = method.solve(instance_object, initial_solution)
        objective = solution.evaluate() / initial_cost
        logging.info(f'Solver finished with target objective {objective}')

        return objective
