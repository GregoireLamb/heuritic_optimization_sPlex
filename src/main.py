from pyinstrument import Profiler

from src.config import Config
from src.instance_loader import InstanceLoader
from src.methods.construction_heuristic import ConstructionHeuristic


def load_method(method_name):
    if method_name == 'construction_heuristic':
        return ConstructionHeuristic

# def load_param(param_list):
#     if param_list == ['deterministic']:
#         return ConstructionHeuristic
#     if param_list == ['randomised']:
#         return ConstructionHeuristic


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
        solution = method.solve(instance, config.param_list)
        print(solution)
        solution.save(config)

    prof.stop()
    # print(prof.output_text(unicode=True, color=True))
