from pathlib import Path
import yaml
from multiprocessing import cpu_count


class Config:
    """
    Class to read the config.yaml, config_multiple.yaml file and provide access to the config parameters.
    The .yaml files should be in the root folder
    """

    def __init__(self, exec_type: str = 'single', worker_id: int = None):
        self.exec_type = exec_type
        self.worker_id = worker_id

        self._root = Path(__file__).parent.parent.parent
        self._config = self.read_config(f'{self._root}/config_hyperparam_tuner.yaml')

        # Sub-configs
        self._execution_config = self._config['execution']
        self._optimization_config = self._config['optimization']
        self._paths = self._config['paths']

        # Parameters
        self.execution_name = self._execution_config['execution_name']
        self.method_to_tune = self._execution_config['method_to_tune']
        self.objective = self._execution_config['objective']
        self.use_instance_features = self._execution_config['use_instance_features']
        self.instance_features = self._execution_config['instance_features']
        self.use_150_selected_instances = self._execution_config['use_150_selected_instances']
        self.instance_type = self._execution_config['instance_type']
        self.instance_indices = self._execution_config['instance_indices']
        self.start_new_run = self._execution_config['start_new_run']
        self.only_get_results = self._execution_config['only_get_results']
        if self.start_new_run and self.only_get_results:
            raise ValueError('start_new_run and only_get_results cannot be both True')

        self.n_trials = self._optimization_config['n_trials']
        self.walltime_limit = self._optimization_config['walltime_limit']
        if type(self.walltime_limit) is str:
            self.walltime_limit = eval(self.walltime_limit)
        self.time_limit_per_trial = self._optimization_config['time_limit_per_trial']
        if type(self.time_limit_per_trial) is str:
            self.time_limit_per_trial = eval(self.time_limit_per_trial)
        self.bb_seed = self._optimization_config['bb_seed']
        self.n_workers = self._optimization_config['n_workers']
        if type(self.n_workers) is str:
            if self.n_workers == 'half_cores':
                self.n_workers = cpu_count() // 2
            else:
                raise ValueError(f'Invalid value for n_workers: {self.n_workers}')
        self.n_iterations_without_improvement = self._optimization_config['n_iterations_without_improvement']

        # Paths
        self.instances_path = f'{self._root}/instances/{self._paths["opt_instances"]}'
        self.instance_info_path = f'{self._root}/instances/instance_information.csv'
        self.smac_output_dir = f'{self._root}/smac3_output' if self.exec_type == 'single' else f'{self._root}/smac3_output_multiple/worker_{self.worker_id}'

        # Hyperparameters information
        hyperparam_dict = self.read_config(f'{self._root}/{self._paths["hyperparameters"]}')
        self.hyperparameters = hyperparam_dict.get(self.method_to_tune, None)
        if not self.hyperparameters:
            raise ValueError(f'Hyperparameters not found for method {self.method_to_tune}. '
                             f'Available: {list(hyperparam_dict.keys())}')

    def read_config(self, path: str):
        """
        Load configuration .yaml file

        Args:
            path (str): Path to config file

        Returns:
            config (dict): Dictionary with configuration file
        """

        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def set_param(self, param: str, value):
        """
        Set a parameter value

        Args:
            param (str): Parameter name
            value (any): Parameter value
        """
        # Check if parameter exists. Is it an attribute of this class?
        if not hasattr(self, param):
            raise ValueError(f'Parameter {param} not found')

        # Check the type of the value is the same as the type of the parameter
        if type(getattr(self, param)) != type(value):
            raise ValueError(
                f'Parameter {param} is of type {type(getattr(self, param))} but value is of type {type(value)}')

        setattr(self, param, value)
