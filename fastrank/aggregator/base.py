__all__ = ['RankAggregator', 'AggregateRefiner']

import numpy as np


class RankAggregator:
    def aggregate(self, preferences: np.ndarray) -> np.ndarray:
        """
        Takes in a preference matrix and returns a central preference that minimizes some distance to all the
        preferences, either exactly or approximately. The matrix has m rows as observations and n columns as
        preferences, where the index of the element in the list specifies the rank. For example, [[2, 1, 0]] means that
        item 2 is the most preferred.

        Args:
             preferences: the $m$ by $n$ preferences matrix.

        Returns:
            The central preference as a 1D array of length $n$.
        """
        raise NotImplementedError


class AggregateRefiner:
    def refine(self, preferences: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        """
        Takes in a preference matrix and a candidate central preference and returns a refined central preference that
        minimizes some distance to all the preferences, either exactly or approximately. The matrix has m rows as
        observations and n columns as preferences, where the index of the element in the list specifies the rank. For
        example, [[2, 1, 0]] means that item 2 is the most preferred.

        Args:
             preferences: the $m$ by $n$ preferences matrix.
             candidate: the candidate central preference as a 1D array of length $n$.

        Returns:
            The refined central preference as a 1D array of length $n$.
        """
        raise NotImplementedError
