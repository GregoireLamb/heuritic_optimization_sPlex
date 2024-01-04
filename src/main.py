from pyinstrument import Profiler

from src.config import Config
from src.instance_loader import InstanceLoader
from src.methods.brk_ga import BRKGA
from src.methods.construction_heuristic import ConstructionHeuristic
from src.methods.grasp import GRASP
from src.methods.local_search import LocalSearch
from src.methods.simulated_annealing import SimulatedAnnealing
from src.methods.vnd import VND
from src.utils import Instance


def run_method(config: Config, instance: Instance, instance_loader: InstanceLoader):
    # These methods improve an initial solution
    initial_solution = instance_loader.get_instance_saved_solution(instance)
    if initial_solution is None:
        print(f'No saved solution for instance {instance.name}, generating one with construction heuristic')
        method = ConstructionHeuristic(params=config.method_params['construction_heuristic'])
        initial_solution = method.solve(instance)
        initial_solution.save(config,
                              path=f'{config.solutions_dir}/construction_heuristic/{config.det_or_random_construction}')
    # Improve solution
    if config.method == 'local_search':
        method = LocalSearch(config, params=config.this_method_params)
        solution = method.solve(instance, initial_solution)
    elif config.method == 'simulated_annealing':
        method = SimulatedAnnealing(config, params=config.this_method_params)
        solution = method.solve(instance, initial_solution)
    elif config.method == 'variable_neighborhood_descent':
        method = VND(config, params=config.this_method_params)
        solution = method.solve(instance, initial_solution)
    elif config.method == 'brk_genetic_algorithm':
        method = BRKGA(config, params=config.this_method_params)
        solution = method.solve(instance, initial_solution)
    else:
        raise ValueError(f'Method {config.method} not implemented')

    return initial_solution, solution


METHODS_THAT_NEED_INITIAL_SOLUTION = ['local_search', 'simulated_annealing', 'variable_neighborhood_descent', 'brk_genetic_algorithm']


if __name__ == '__main__':
    prof = Profiler()
    prof.start()

    config = Config()

    assert config.execution.lower() == 'normal', 'Execution mode must be normal'
    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances()
    for i, instance in enumerate(instances):
        print(f'Running instance {i + 1} / {len(instances)}')
        print(f'Instance information:\n{instance}')
        if config.method in METHODS_THAT_NEED_INITIAL_SOLUTION:
            initial_solution, solution = run_method(config, instance, instance_loader)
            print(f'Initial solution cost: {initial_solution.evaluate()}', end='\n')
        elif config.method == 'construction_heuristic':
            method = ConstructionHeuristic(params=config.this_method_params)
            solution = method.solve(instance)
        elif config.method == 'greedy_randomized_adaptive_search_procedure':
            method = GRASP(config, params=config.this_method_params)
            solution = method.solve(instance)
        elif config.method == 'brk_genetic_algorithm':
            method = BRKGA(config, params=config.this_method_params)
            solution = method.solve(instance)
        else:
            raise ValueError(f'Method {config.method} not implemented')

        assert solution.is_feasible(), 'Solution is not feasible'
        print(f'Final solution cost: {solution.evaluate()}', end='\n\n')
        print(solution)
        solution.save(config)

        new_solution = instance_loader.get_instance_saved_solution(instance, method=config.method)
        assert new_solution.evaluate() == solution.evaluate(), \
            'Saved solution is not the same as the current solution'

    prof.stop()
    print(prof.output_text(unicode=True, color=True))

