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
    method = load_method(config.method)()
    for i, instance in enumerate(instances):
        print(f'Running instance {i + 1} / {len(instances)}')
        print(f'Instance information:\n{instance}')
        solution = method.solve(instance, Randomize=True)
        print(solution)

    prof.stop()
    print(prof.output_text(unicode=True, color=True))
