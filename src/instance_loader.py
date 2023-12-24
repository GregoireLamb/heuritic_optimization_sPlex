import os
from src.config import Config
from src.solution import Solution
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
            if self._config.instance_indices and i not in self._config.instance_indices:
                continue
            instances.append(self.load_instance(f'{dir_path}/{file}'))

        return instances

    def get_instance_saved_solution(self, instance: Instance, method="construction_heuristic"):
        """
        Get the saved solution for the given instance name
        """
        print(f'Looking for saved solution for instance {instance.name}')
        if method == "construction_heuristic":
            path = f'{self._config.solutions_dir}/{method}/' \
                   f'{self._config.det_or_random_construction}/{instance.name}.txt'
        else:
            path = f'{self._config.solutions_dir}/{method}/{instance.name}.txt'
        if not os.path.isfile(path):
            print(f'No saved solution found for {method} for instance {instance.name}')
            return None

        print(f'Loading saved solution for instance {instance.name}')
        with open(path, 'r') as f:
            lines = f.readlines()
            x = {edge: value for edge, value in instance.in_instance.items() if edge[0] < edge[1]}

            for line in lines[1:]:
                i, j = line.split()
                x[(int(i), int(j))] = 1 - x[(int(i), int(j))]

        return Solution(instance, x)
