import cProfile

import numpy as np

from permsc import BordaRankAggregator, RRFRankAggregator, LocalSearchRefiner, sum_kendall_tau, \
    sample_random_preferences, KemenyLocalSearchRefiner, ranks_from_preferences, compute_preferences_kendall_tau
from permsc.aggregator.exact import KemenyOptimalAggregator


def main():
    prefs = np.array([[-1, 2, 0, -1], [1, -1, -1, 3], [-1, 2, -1, 3]])
    proposal = BordaRankAggregator().aggregate(prefs)

    y_optimal = KemenyOptimalAggregator().aggregate(prefs)
    print(y_optimal)
    print('Optimal', compute_preferences_kendall_tau(prefs, y_optimal))
    print('Borda', compute_preferences_kendall_tau(prefs, proposal))
    # print('Borda + LS', compute_preferences_kendall_tau(prefs, ls_proposal))
    # print('RRF', compute_preferences_kendall_tau(prefs, rrf_proposal))


if __name__ == '__main__':
    main()
