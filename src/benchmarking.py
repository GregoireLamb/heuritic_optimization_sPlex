import time

from pyinstrument import Profiler
import pandas as pd

from src.config import Config
from src.instance_loader import InstanceLoader
from src.main import run_method
from src.methods.construction_heuristic import ConstructionHeuristic
from src.methods.grasp import GRASP

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

    # Check if there are instances that have not been run yet
    # already_run_instances = df[
    #     (df['type'] == config.instance_type) &
    #     (df.method == config.method)].instance.unique()
    # instances = [instance for instance in instances if instance.name not in already_run_instances]

    print(f'Running benchmarking for {len(instances)} instances: '
          f'{[instance.name for instance in instances]} ')

    for i, instance in enumerate(instances):
        print(f'\n\nRunning instance {i + 1} / {len(instances)}')
        print(f'Instance information:\n{instance}')
        start = time.time()

        if config.method == 'construction_heuristic':
            method = ConstructionHeuristic(params=config.this_method_params)
            solution = method.solve(instance)
            initial_solution = None
        elif config.method == 'greedy_randomized_adaptive_search_procedure':
            method = GRASP(config, params=config.this_method_params)
            solution = method.solve(instance)
            initial_solution = None
        else:
            initial_solution, solution = run_method(config, instance, instance_loader)
        runtime = time.time() - start
        assert solution.is_feasible(), 'Solution is not feasible'
        print(f'Runtime: {runtime} seconds')
        print(f'Initial solution cost: {initial_solution.evaluate() if initial_solution is not None else "N/A"}')
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

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

