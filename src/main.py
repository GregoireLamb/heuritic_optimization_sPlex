from pyinstrument import Profiler

from src.config import Config
from src.instance_loader import InstanceLoader
from src.methods.construction_heuristic import ConstructionHeuristic
from src.methods.local_search import LocalSearch

if __name__ == '__main__':
    prof = Profiler()
    prof.start()

    config = Config()
    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances()

    for i, instance in enumerate(instances):
        print(f'Running instance {i + 1} / {len(instances)}')
        print(f'Instance information:\n{instance}')

        if config.method == 'construction_heuristic':
            method = ConstructionHeuristic(params=config.method_params)
            solution = method.solve(instance)
        elif config.method == 'local_search':
            solution = instance_loader.get_instance_saved_solution(instance)
            if solution is None:
                method = ConstructionHeuristic(params=config.method_params)
                solution = method.solve(instance)
            method = LocalSearch(config, params=config.method_params)
            solution = method.solve(instance, solution)
        else:
            raise ValueError(f'Method {config.method} not implemented')

        assert solution.is_feasible(), 'Solution is not feasible'
        print(f'Final solution cost: {solution.evaluate()}', end='\n\n')
        print(solution)
        solution.save(config)

    prof.stop()
    print(prof.output_text(unicode=True, color=True))
