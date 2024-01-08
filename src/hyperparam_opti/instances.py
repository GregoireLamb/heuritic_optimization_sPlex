import os
import pandas as pd
import logging
from hyperparam_config import Config
from src.instance_loader import InstanceLoader
from src.config import Config as BasicConfig


class Instances:
    """
    Class to read the instances folder and provide access to the instances and their meta data.
    # TODO: Add instance features for possible multi-instance optimization
    """

    def __init__(self, config: Config):
        self._config = config
        basic_config = BasicConfig()
        basic_config.instance_type = config.instance_type
        basic_config.instance_indices = config.instance_indices
        self._instance_loader = InstanceLoader(basic_config)

        self.instances = self._read_instances()
        instance_objects = self._instance_loader.load_instances()
        assert len(self.instances) == len(instance_objects), f'Number of instances does not match.'
        self._instance_objects_map = {instance.name: instance for instance in instance_objects}
        self.instance_features = self._create_instance_features()
        self._default_solutions = {instance.name: self._instance_loader.get_instance_saved_solution(instance)
                                   for instance in instance_objects}

        logging.info(f'Loaded {len(self.instances)} instances')

    def _read_instances(self):
        """
        Read all instances from the instances folder and return a list of all instance names (without extension).
        We just take the unzipped instances (.mps files). Take subset of instances if specified in config.
        Remove infeasible instances if specified in config.
        """
        files = os.listdir(self._instance_loader.instances_dir)

        files = sorted([file.split('.')[0] for file in files if file.endswith('.txt')])

        if self._config.instance_indices:
            files = [file for i, file in enumerate(files) if i in self._config.instance_indices]

        return files

    def _create_instance_features(self):
        """
        Read instance features. Features used are specified in the configuration.
        """
        if not self._config.use_instance_features:
            return None
        feat = {}
        for instance in self.instances:
            instance_obj = self._instance_objects_map[instance]
            feat[instance] = [instance_obj.n, instance_obj.m, instance_obj.s / instance_obj.n]
        return feat

    def get_instance_info(self, instance: str):
        """
        Return the instance info for the given instance
        """
        instance = self.get_instance_object(instance)
        default_solution = self._default_solutions[instance.name]

        return {
            'construction_heuristic_cost': default_solution.evaluate() if default_solution is not None else None,
            'construction_heuristic_sol': default_solution,
            'n': instance.n,
            'm': instance.m,
            's': instance.s,
        }

    def get_instance_object(self, instance: str):
        """
        Return the instance object for the given instance
        """
        return self._instance_objects_map[instance]


