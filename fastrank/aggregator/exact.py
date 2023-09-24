from itertools import combinations, permutations

import numpy as np
from scipy.optimize import linprog

from .base import RankAggregator
from ..utils import cdk_graph_from_preferences


class KemenyOptimalAggregator(RankAggregator):
    def aggregate(self, preferences: np.ndarray) -> np.ndarray:
        num_voters, num_cands = preferences.shape
        X_graph = cdk_graph_from_preferences(preferences)
        c = -1 * X_graph.ravel()

        idx = lambda i, j: num_cands * i + j

        # constraints for every pair
        pairwise_constraints = np.zeros(((num_cands * (num_cands - 1)) / 2, num_cands ** 2))
        for row, (i, j) in zip(pairwise_constraints, combinations(range(num_cands), 2)):
            row[[idx(i, j), idx(j, i)]] = 1

        # and for every cycle of length 3
        triangle_constraints = np.zeros(((num_cands * (num_cands - 1) *
                                          (num_cands - 2)),
                                         num_cands ** 2))
        for row, (i, j, k) in zip(triangle_constraints, permutations(range(num_cands), 3)):
            row[[idx(i, j), idx(j, k), idx(k, i)]] = 1

        constraint_rhs1 = np.ones(len(pairwise_constraints))  # ==
        constraint_rhs2 = np.ones(len(triangle_constraints))  # >=
        constraint_signs = np.hstack([np.zeros(len(pairwise_constraints)),  # ==
                                      np.ones(len(triangle_constraints))])  # >=

        linprog(c, triangle_constraints, constraint_rhs2, pairwise_constraints, constraint_rhs1, (0, 1))