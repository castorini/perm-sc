__all__ = ['ranks_from_preferences', 'sample_random_preferences', 'small_kendall_tau', 'sum_kendall_tau', 'spearmanr',
           'sum_spearmanr']

from typing import Tuple

import numba
import numpy as np


@numba.njit
def ranks_from_preferences(preferences: np.ndarray) -> np.ndarray:
    """
    Takes in a preference matrix (or vector) and returns a rank array of the same shape. The rank matrix has
    m rows as observations and n columns as preferences, where the index of the element in the list specifies the rank.
    For example, the preference list [[2, 1, 0]] has a rank array of [[2, 0, 1]].

    Args:
         preferences: the $m$ by $n$ preferences matrix or a preference vector with $n$ elements.

    Returns:
        The rank matrix as a 1D or 2D array.
    """
    ranks = np.empty_like(preferences)

    if preferences.ndim == 2:  # workaround for numba
        for i in range(preferences.shape[0]):
            for j in range(preferences.shape[1]):
                ranks[i, preferences[i, j]] = j
    elif preferences.ndim == 1:
        for i in range(preferences.shape[0]):
            ranks[preferences[i]] = i

    return ranks


@numba.njit(parallel=True)
def sum_spearmanr(X: np.ndarray, y: np.ndarray, cached_ranks: Tuple[np.ndarray, np.ndarray] = None) -> int:
    """Sums all Spearman's rho from `y` to each row vector in `X`"""
    rho = 0

    if cached_ranks is None:
        X_ranks = ranks_from_preferences(X)
        y_ranks = ranks_from_preferences(y)
    else:
        X_ranks, y_ranks = cached_ranks

    for i in numba.prange(X.shape[0]):
        rho += spearmanr(X[i], y, (X_ranks[i], y_ranks))

    return rho


@numba.njit(parallel=True)
def sum_kendall_tau(X: np.ndarray, y: np.ndarray) -> int:
    """Sums all the Kendall tau distances from `y` to each row vector in `X`"""
    tau = 0

    for i in numba.prange(X.shape[0]):
        tau += small_kendall_tau(X[i], y)

    return tau


@numba.njit
def spearmanr(a: np.ndarray, b: np.ndarray,cached_ranks: Tuple[np.ndarray, np.ndarray] = None) -> float:
    """Computes the Spearman's rho between two preference arrays."""
    if cached_ranks is None:
        a = ranks_from_preferences(a)
        b = ranks_from_preferences(b)
    else:
        a, b = cached_ranks

    return 1 - (6 * ((a - b) ** 2).sum() / (a.shape[0] * (a.shape[0] ** 2 - 1)))


@numba.njit
def small_kendall_tau(a: np.ndarray, b: np.ndarray) -> int:
    """
    Computes the Kendall tau distance between two arrays. This implementation is quadratic-time but incredibly fast for
    small arrays (< 500) due to numba and cache locality. For large arrays, we'll need to implement a MergeSort-based algorithm.
    """
    assert a.shape == b.shape
    n = a.shape[0]
    tau = 0

    for i in range(n):
        for j in range(i + 1, n):
            if (a[i] < a[j] and b[i] > b[j]) or (a[i] > a[j] and b[i] < b[j]):
                tau += 1

    return tau


def sample_random_preferences(m: int, n: int) -> np.ndarray:
    rand_prefs = [np.arange(n) for _ in range(m)]

    for x in rand_prefs:
        np.random.shuffle(x)

    return np.array(rand_prefs)


if __name__ == '__main__':
    rand_prefs = sample_random_preferences(10000, 500)
    print(ranks_from_preferences(np.array(rand_prefs)))
    import time
    a = time.time()
    print(ranks_from_preferences(np.array(rand_prefs)))
    print(time.time() - a)

    print(small_kendall_tau(rand_prefs[0], rand_prefs[1]))
    a = time.time()
    print(small_kendall_tau(rand_prefs[0], rand_prefs[1]))
    print(time.time() - a)
