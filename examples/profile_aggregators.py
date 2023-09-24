import cProfile

import numpy as np

from fastrank import BordaRankAggregator, RRFRankAggregator, LocalSearchRefiner, sum_kendall_tau, \
    sample_random_preferences, KemenyLocalSearchRefiner, ranks_from_preferences, compute_preferences_kendall_tau
from fastrank.aggregator.exact import KemenyOptimalAggregator


def main():
    real_prefs = np.array([[1, 2, 0], [1, 2, 0], [1, 0, 2], [0, 1, 2], [1, 2, 0]])
    real_proposal = BordaRankAggregator().aggregate(real_prefs)
    rrf_proposal = RRFRankAggregator().aggregate(real_prefs)
    refined_proposal = LocalSearchRefiner().refine(real_prefs, real_proposal)

    print(real_proposal, refined_proposal)
    print(sum_kendall_tau(real_prefs, real_proposal), sum_kendall_tau(real_prefs, refined_proposal))
    real_proposal = BordaRankAggregator().aggregate(real_prefs)

    import time
    a = time.time()
    prefs = sample_random_preferences(10, 50)  # warmup
    proposal = BordaRankAggregator().aggregate(prefs)
    print(time.time() - a)

    print(np.sum(LocalSearchRefiner().refine(prefs, proposal)))
    print(np.sum(KemenyLocalSearchRefiner().refine(prefs, proposal)))

    prefs = sample_random_preferences(30, 30)
    proposal = BordaRankAggregator().aggregate(prefs)
    rrf_proposal = RRFRankAggregator().aggregate(prefs)
    ls_proposal = KemenyLocalSearchRefiner().refine(prefs, proposal)

    y_optimal = KemenyOptimalAggregator().aggregate(prefs)
    print('Optimal', compute_preferences_kendall_tau(prefs, y_optimal))
    print('Borda', compute_preferences_kendall_tau(prefs, proposal))
    print('Borda + LS', compute_preferences_kendall_tau(prefs, ls_proposal))
    print('RRF', compute_preferences_kendall_tau(prefs, rrf_proposal))


if __name__ == '__main__':
    main()
