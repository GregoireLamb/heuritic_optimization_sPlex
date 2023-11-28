from pyinstrument import Profiler

from src.config import Config
from src.instance_loader import InstanceLoader
from src.methods.construction_heuristic import ConstructionHeuristic
from src.methods.local_search import LocalSearch
from src.methods.simulated_annealing import SimulatedAnnealing
from src.methods.vns import Vns

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
        else:
            # These methods improve an initial solution
            solution = instance_loader.get_instance_saved_solution(instance)
            if solution is None:
                method = ConstructionHeuristic(params=config.method_params)
                solution = method.solve(instance)

            # Improve solution
            if config.method == 'local_search':
                method = LocalSearch(config, params=config.method_params)
                solution = method.solve(instance, solution)
            elif config.method == 'simulated_annealing':
                method = SimulatedAnnealing(config, params=config.method_params)
                solution = method.solve(instance, solution)
            elif config.method == 'variable_neighborhood_descent':
                method = Vns(config, params=config.method_params)
                solution = method.solve(instance, solution)
            else:
                raise ValueError(f'Method {config.method} not implemented')

        assert solution.is_feasible(), 'Solution is not feasible'
        print(f'Final solution cost: {solution.evaluate()}', end='\n\n')
        print(solution)
        solution.save(config)


        new_solution = instance_loader.get_instance_saved_solution(instance)
        assert new_solution.evaluate() == solution.evaluate(), 'Saved solution is not the same as the current solution'


    prof.stop()
    print(prof.output_text(unicode=True, color=True))

