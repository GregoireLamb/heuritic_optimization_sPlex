import time

from src.config import Config
from src.instance_loader import InstanceLoader
from src.methods.construction_heuristic import ConstructionHeuristic


def test_neighborhoods():
    # Generate a solution using a deterministic construction heuristic
    config = Config()
    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances()
    instance = instances[0]

    solution = ConstructionHeuristic(params=config.method_params).solve(instance)
    print(f'Initial solution cost: {solution.evaluate()}')
    print(solution)

    to_test = ['kflips_1', 'kflips_2']

    n_neighbors_test = 200

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
    test_neighborhoods()
