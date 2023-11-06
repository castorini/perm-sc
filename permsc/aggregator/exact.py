__all__ = ['KemenyOptimalAggregator']

from itertools import combinations, permutations

import numpy as np
from pulp import LpProblem, LpVariable, LpMinimize, lpSum

from .base import RankAggregator
from .utils import cdk_graph_from_preferences, preferences_from_cdk_graph


class KemenyOptimalAggregator(RankAggregator):
    def solve_from_graph(self, cdk_graph: np.ndarray):

        # Create the LP problem
        prob = LpProblem('KemenyOptimalAggregator', LpMinimize)
        vars_dict = {}
        objectives = []
        n = cdk_graph.shape[-1]

        for i, j in combinations(range(n), 2):
            x_ij = LpVariable(f'x{i},{j}', 0, 1, 'Binary')
            x_ji = LpVariable(f'x{j},{i}', 0, 1, 'Binary')
            vars_dict[(i, j)] = x_ij
            vars_dict[(j, i)] = x_ji
            objectives.append(cdk_graph[i, j] * x_ij + cdk_graph[j, i] * x_ji)

        prob += (lpSum(objectives), 'Kemeny loss')

        for i, j in combinations(range(n), 2):
            prob += (vars_dict[(i, j)] + vars_dict[(j, i)] == 1, f'Existence x{i},{j}')

        for i, j, k in permutations(range(n), 3):
            if i == j or j == k or i == k:
                continue

            prob += (vars_dict[(i, j)] + vars_dict[(j, k)] + vars_dict[(k, i)] >= 1, f'Transitivity x{i},{j},{k}')

        prob.solve()

        # Extract the preferences from the solution
        y_graph = np.zeros((n, n), dtype=int)

        for (i, j), var in vars_dict.items():
            y_graph[i, j] = int(var.varValue)

        y_graph = y_graph.T  # LP problem is reversed

        return preferences_from_cdk_graph(y_graph)

    def aggregate(self, preferences: np.ndarray) -> np.ndarray:
        cdk_graph = cdk_graph_from_preferences(preferences)
        return self.solve_from_graph(cdk_graph)
