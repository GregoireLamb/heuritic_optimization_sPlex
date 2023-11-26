import yaml
from pathlib import Path


class Config:
    def __init__(self):
        """
        Initialize Config class. Load config.yaml file and define paths.
        """
        # Define root directory, one level up from current directory
        self.root = Path(__file__).parent.parent
        self.config_path = f'{self.root}/config.yaml'

        self.__config = self.load_config()

        # Execution
        self.instance_type = self.__config["execute"]["instance_type"]
        self.instance_indices = self.__config["execute"]["instance_indices"]
        self.method = self.__config["execute"]["method"]
        self.det_or_random_construction = 'randomized' \
            if self.__config["params"]["construction_heuristic"]["randomized"] else 'deterministic'

        # Param dictionary for the particular method
        self.method_params = self.__config["params"][self.method]
        self.neighborhood_params = self.__config["params"]["neighborhood_params"]

        # Paths
        self.instances_dir = f'{self.root}/{self.__config["paths"]["instances"]}'
        self.solutions_dir = f'{self.root}/{self.__config["paths"]["solutions"]}'

    def load_config(self):
        """
        Load config.yaml file
        Returns:
            config (dict): Dictionary with config.yaml file
        """

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
