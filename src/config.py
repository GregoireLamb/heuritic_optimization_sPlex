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

        # Paths
        self.instances_dir = f'{self.root}/{self.__config["paths"]["instances"]}'

    def load_config(self):
        """
        Load config.yaml file
        Returns:
            config (dict): Dictionary with config.yaml file
        """

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
