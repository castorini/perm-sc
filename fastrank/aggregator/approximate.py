__all__ = ['BordaRankAggregator', 'LocalSearchRefiner']

import numba
import numpy as np

from .base import RankAggregator, AggregateRefiner
from ..utils import ranks_from_preferences, sum_kendall_tau, sum_spearmanr


class BordaRankAggregator(RankAggregator):
    def aggregate(self, preferences: np.ndarray) -> np.ndarray:
        def _compute_borda(preferences: np.ndarray) -> np.ndarray:
            positions = ranks_from_preferences(preferences)
            return positions.mean(axis=0)

        borda_counts = _compute_borda(preferences)
        return np.argsort(borda_counts)


@numba.njit
def _ls_refine(preferences: np.ndarray, candidate: np.ndarray, fn, method: str = 'min') -> np.ndarray:
    improved = True
    other = candidate.copy()
    best_dist = fn(preferences, candidate)

    is_max = method == 'max'

    if is_max:
        best_dist = -best_dist

    while improved:
        improved = False
        rand_indices = np.random.permutation(len(candidate))

        for i in rand_indices:
            for j in range(i + 1, len(candidate)):
                other[i], other[j] = other[j], other[i]
                curr_dist = fn(preferences, other)

                if is_max:
                    curr_dist = -curr_dist

                if curr_dist < best_dist:
                    improved = True
                    best_dist = curr_dist
                    candidate, other = other, other.copy()
                else:
                    other[i], other[j] = other[j], other[i]

    return candidate


class LocalSearchRefiner(AggregateRefiner):
    def __init__(self, objective: str = 'kendalltau'):
        self.objective = objective

    def refine(self, preferences: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        match self.objective:
            case 'kendalltau':
                return _ls_refine(preferences, candidate, sum_kendall_tau)
            case 'spearmanr':
                return _ls_refine(preferences, candidate, sum_spearmanr, 'max')
            case _:
                raise ValueError(f'Objective {self.objective} is not supported.')


if __name__ == '__main__':
    real_prefs = np.array([[1, 2, 0], [1, 2, 0], [1, 0, 2], [0, 1, 2], [1, 2, 0]])
    real_proposal = BordaRankAggregator().aggregate(real_prefs)
    refined_proposal = LocalSearchRefiner().refine(real_prefs, real_proposal)
    print(real_proposal, refined_proposal)
    print(sum_kendall_tau(real_prefs, real_proposal), sum_kendall_tau(real_prefs, refined_proposal))
    real_proposal = BordaRankAggregator().aggregate(real_prefs)
    print(real_proposal, LocalSearchRefiner('spearmanr').refine(real_prefs, real_proposal))

    # import time
    # a = time.time()
    # prefs = sample_random_preferences(100, 100)
    # proposal = BordaRankAggregator().aggregate(prefs)
    # print(time.time() - a)
    #
    # a = time.time()
    # print(LocalSearchRefiner().refine(prefs, proposal))
    # print(time.time() - a)
