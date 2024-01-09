import time

from pyinstrument import Profiler
import pandas as pd

from src.config import Config
from src.instance_loader import InstanceLoader
from src.main import run_method, METHODS_THAT_NEED_INITIAL_SOLUTION
from src.methods.construction_heuristic import ConstructionHeuristic
from src.methods.grasp import GRASP
from src.methods.ants_colony import AntColony
from src.methods.brk_ga import BRKGA

RESULTS_DF_COLUMNS = ['type', 'instance', 'method', 'execution_time', 'solution_cost', 'initial_solution_cost', 'configuration']


if __name__ == '__main__':
    profiler = Profiler()
    profiler.start()

    config = Config()
    assert config.execution.lower() == 'benchmarking', 'Execution mode must be benchmarking'

    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances()

    print(f'Running benchmarking for {config.method} method')

    try:
        df = pd.read_csv(config.results_file, sep=';', decimal=',')
    except FileNotFoundError:
        df = pd.DataFrame(columns=RESULTS_DF_COLUMNS)

    print(f'Running benchmarking for {len(instances)} instances: '
          f'{[instance.name for instance in instances]} ')

    for i, instance in enumerate(instances):
        print(f'\n\nRunning instance {i + 1} / {len(instances)}')
        print(f'Instance information:\n{instance}')
        start = time.time()
        initial_solution = None

        if config.method in METHODS_THAT_NEED_INITIAL_SOLUTION:
            initial_solution, solution = run_method(config, instance, instance_loader)
            print(f'Initial solution cost: {initial_solution.evaluate()}', end='\n')
        elif config.method == 'construction_heuristic':
            method = ConstructionHeuristic(params=config.this_method_params)
            solution = method.solve(instance)
        elif config.method == 'greedy_randomized_adaptive_search_procedure':
            method = GRASP(config, params=config.this_method_params)
            solution = method.solve(instance)
        elif config.method == 'ant_colony_optimization':
            method = AntColony(config, params=config.this_method_params)
            solution = method.solve(instance)
        elif config.method == 'brk_genetic_algorithm':
            method = BRKGA(config, params=config.this_method_params)
            solution = method.solve(instance)
        else:
            raise ValueError(f'Method {config.method} not implemented')

        runtime = time.time() - start
        assert solution.is_feasible(), 'Solution is not feasible'
        print(f'Runtime: {runtime} seconds')
        print(f'Solution cost: {solution.evaluate()}')
        solution.save(config)

        row_df = pd.DataFrame({
            'type': [config.instance_type],
            'instance': [instance.name],
            'method': [config.method],
            'execution_time': [runtime],
            'solution_cost': [solution.evaluate()],
            'initial_solution_cost': [initial_solution.evaluate() if initial_solution is not None else 'N/A'],
            'configuration': [str(config.this_method_params)]
        })
        df = pd.concat([df, row_df], ignore_index=True)
        df.to_csv(config.results_file, sep=';', decimal=',', index=False)

        row_df_test = pd.DataFrame({
            'instance': [instance.name],
            'solution_cost': [solution.evaluate()],
            'initial_solution_cost': [initial_solution.evaluate() if initial_solution is not None else 'N/A'],
        })
        df = pd.concat([df, row_df], ignore_index=True)
        df.to_csv(config.results_test_file, sep=';', decimal=',', index=False)
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
