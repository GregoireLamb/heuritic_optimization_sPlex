from pyinstrument import Profiler

from src.config import Config
from src.instance_loader import InstanceLoader
from src.methods.deterministic_construction_heuristic import DeterministicConstructionHeuristic

def load_method(method_name):
    if method_name == 'deterministic_construction_heuristic':
        return DeterministicConstructionHeuristic


if __name__ == '__main__':
    prof = Profiler()
    prof.start()

    config = Config()
    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances()

    # for instance in instances:
    instance = instances[0]
    method = load_method(config.method)()
    solution = method.solve(instance)
    print(solution)

    prof.stop()
    print(prof.output_text(unicode=True, color=True))