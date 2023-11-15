from pymhlib.solution import VectorSolution

from src.utils import Solution


class DeterministicConstructionHeuristic:
    def __init__(self, params=None):
        self._instance = None
        self._x = {}

    def solve(self, instance) -> Solution:
        self._instance = instance
        self._x = {e: 0 for e in instance.edges}
        components = set()

        to_add = set(instance.nodes)

        while len(to_add) > 0:
            best_cost, best_component, best_node = -1e10, -1, -1
            for i in to_add:
                # New component
                cost = self.compute_cost_of_new_component(i)
                if cost < best_cost:
                    best_cost = cost
                    best_component = -1
                    best_node = i
                # Existing component
                for k in components:
                    cost = self.compute_insertion_cost_to_component(i, k)
                    if cost < best_cost:
                        best_cost = cost
                        best_component = k
                        best_node = i
            if best_component == -1:
                components.add({best_node})
            else:
                self.add_to_component(best_node, best_component)
            to_add.remove(best_node)
        return Solution(instance, self._x)


