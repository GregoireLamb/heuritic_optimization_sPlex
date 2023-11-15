import os
from src.config import Config
from src.utils import Instance


class InstanceLoader:
    def __init__(self, config: Config):
        self._config = config

    @staticmethod
    def load_instance(path) -> Instance:
        """
        Load the instance located in the path
        """
        name = path.split('/')[-1].split('.')[0]

        # Open the .txt file
        with open(path, 'r') as f:
            lines = f.readlines()

            # First line is s, n, m, l
            s, n, m, _ = lines[0].split()
            edges = {}

            for line in range(1, len(lines)):
                i, j, x, w = lines[line].split()
                edges[(int(i), int(j))] = (int(x), int(w))

        return Instance(s, n, m, edges, name)

    def load_instances(self) -> list:
        """
        Load instances according to config instance type
        """
        dir_path = f'{self._config.instances_dir}/inst_{self._config.instance_type}'
        assert os.path.isdir(dir_path), f'Instance type {self._config.instance_type} does not exist'

        instances = []
        for i, file in enumerate(os.listdir(dir_path)):
            if self._config.instance_indices is not None and i not in self._config.instance_indices:
                continue
            instances.append(self.load_instance(f'{dir_path}/{file}'))

        return instances

