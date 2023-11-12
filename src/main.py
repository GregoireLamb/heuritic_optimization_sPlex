from src.config import Config
from src.instance_loader import InstanceLoader
from src.methods.deterministic_construction_heuristic import DeterministicConstructionHeuristic


def load_method(method_name):
    if method_name == 'deterministic_construction_heuristic':
        return DeterministicConstructionHeuristic


if __name__ == '__main__':
    config = Config()
    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances()

    for instance in instances:
        method