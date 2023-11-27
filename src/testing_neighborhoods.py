import time

from src.config import Config
from src.instance_loader import InstanceLoader
from src.methods.construction_heuristic import ConstructionHeuristic


def evaluate_neighborhood():
    # Generate a solution using a deterministic construction heuristic
    config = Config()
    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances()
    instance = instances[0]

    solution = instance_loader.get_instance_saved_solution(instance)
    if solution is None:
        assert config.method == 'construction_heuristic', \
            f'Method must be construction heuristic because there is no saved solution for instance {instance.name}'
        solution = ConstructionHeuristic(params=config.method_params).solve(instance)

    print(f'Initial solution cost: {solution.evaluate()}')
    print(f'Feasible: {solution.is_feasible()}')
    print(solution)

    to_test = ['kflips_10']
    # to_test = ['kflips_1', 'kflips_2', 'movenodes_2', 'nodeswap_2']

    n_neighbors_test = 300

    print(f'-----------------------------------------------------------------\n'
          f'---- Testing neighborhoods with {n_neighbors_test} neighbors ----\n'
          f'-----------------------------------------------------------------\n')

    for n_type in to_test:
        print(f'Testing neighborhood {n_type}')
        start_time = time.time()
        neighbors = solution.generate_neighborhood(n_type, config.neighborhood_params)
        i = 0
        costs = []
        while i < n_neighbors_test:

            try:
                neighbor = next(neighbors)
                costs.append(neighbor.evaluate())
                i += 1
            except StopIteration:
                break
            if not neighbor.is_feasible():
                print(f'Infeasible neighbor!')

        if not costs:
            print(f'No neighbors could be generated!')
            return
        print(f' - Neighborhood size: {len(costs)}')
        print(f' - Current solution cost: {solution.evaluate()}')
        print(f' - Best neighbor cost: {min(costs)}')
        print(f' - Average neighbor cost: {sum(costs) / len(costs)}')
        improving = [cost for cost in costs if cost < solution.evaluate()]
        print(f' - Percentage of neighbors that improve: {len(improving) / len(costs)}')
        if improving:
            print(f' - Average improvement: {sum(improving) / len(improving)}')
        print(f' - Total time: {time.time() - start_time}')
        print(f' - Time per neighbor: {(time.time() - start_time) / len(costs)}')
        print()


if __name__ == '__main__':
    evaluate_neighborhood()
