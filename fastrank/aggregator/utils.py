__all__ = ['ranks_from_preferences', 'sample_random_preferences', 'small_kendall_tau', 'sum_kendall_tau', 'spearmanr',
           'sum_spearmanr', 'cdk_graph_from_preferences', 'cdk_graph_vertex_swap', 'cdk_graph_distance',
           'preferences_from_cdk_graph', 'preferences_from_ranks', 'compute_preferences_kendall_tau']

from typing import Tuple

import numba
import numpy as np


@numba.njit(parallel=True)
def _cdk_graph_from_prefs_2d(preferences: np.ndarray):
    ranks = _ranks_from_preferences_2d(preferences)
    graph = np.zeros((ranks.shape[-1], ranks.shape[-1]))

    for i in numba.prange(ranks.shape[-1]):
        i_mask = ranks[:, i] != -1  # missing values are -1

        for j in range(i + 1, ranks.shape[-1]):
            mask = i_mask & (ranks[:, j] != -1)  # missing values are -1
            i_gt_j = np.sum((ranks[:, i] < ranks[:, j]) & mask)  # times i is preferred over j
            j_gt_i = np.sum((ranks[:, i] > ranks[:, j]) & mask)  # times j is preferred over i

            if i_gt_j > j_gt_i:  # i is preferred, so point from i to j
                graph[i, j] = i_gt_j - j_gt_i
            else:
                graph[j, i] = j_gt_i - i_gt_j

    return graph


@numba.njit
def _cdk_graph_from_prefs_1d(preferences: np.ndarray):
    ranks = _ranks_from_preferences_1d(preferences)
    graph = np.zeros((ranks.shape[-1], ranks.shape[-1]))

    for i in numba.prange(ranks.shape[-1]):
        for j in range(i + 1, ranks.shape[-1]):
            if ranks[i] > ranks[j]:  # j is preferred, so point from j to i
                graph[j, i] = 1
            else:
                graph[i, j] = 1

    return graph


def cdk_graph_from_preferences(preferences: np.ndarray) -> np.ndarray:
    """
    Takes in a preference matrix (or vector) and returns an adjacency matrix whose elements (i, j) have the weight
    |#{i preferred to j} - #{j preferred to i}|, with edges pointing from the more to the less preferred candidate. We
    call this the Conitzer-Davenport-Kalagnanam (CDK) graph.

    The preference matrix can be _partial_, meaning that some preferences are missing. We denote missing elements with
    -1, e.g., [[2, 0, -1, -1]] means that item 2 is the most preferred, item 0 is the second, and the rest are unknown.
    However, the CDK graph resulting from this input is noninvertible for obvious reasons. Attempting to convert it back
    to a preference matrix will result in a matrix with random values in the unknown positions.

    See Also:
        - https://vene.ro/blog/kemeny-young-optimal-rank-aggregation-in-python.html
        - https://cdn.aaai.org/AAAI/2006/AAAI06-099.pdf
    """
    if preferences.ndim == 2:  # ndim breaks in numba
        return _cdk_graph_from_prefs_2d(preferences)
    elif preferences.ndim == 1:
        return _cdk_graph_from_prefs_1d(preferences)

    raise ValueError(f'Preferences must be a 1D or 2D array, got {preferences.ndim}D.')


@numba.njit
def cdk_graph_vertex_swap(graph: np.ndarray, i: int, j: int):
    tmp_i = graph[:, i].copy()
    graph[:, i] = graph[:, j]
    graph[:, j] = tmp_i

    tmp_i = graph[i, :].copy()
    graph[i, :] = graph[j, :]
    graph[j, :] = tmp_i


@numba.njit
def cdk_graph_distance(X_graph: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Computes the Conitzer-Davenport-Kalagnanam (CDK) distance (our coined term) between two graphs, defined as the
    weights of disagreed edges in https://cdn.aaai.org/AAAI/2006/AAAI06-099.pdf. `y_pred` should be a binary graph
    associated with the candidate aggregate ranking. `X_graph` should be a weighted directed CDK graph computed from
    :py:meth:`.graph_from_preferences`.
    """
    return np.sum(y_pred.T * X_graph)


def preferences_from_cdk_graph(graph: np.ndarray) -> np.ndarray:
    i_pref_counts = graph.sum(axis=1)  # sum of all edges pointing to i (i is preferred)
    return np.argsort(i_pref_counts)[::-1]  # sort by most preferred


@numba.njit
def _ranks_from_preferences_2d(preferences: np.ndarray) -> np.ndarray:
    ranks = np.full_like(preferences, -1)

    for i in range(preferences.shape[0]):
        for j in range(preferences.shape[1]):
            pref = preferences[i, j]

            if pref != -1:  # -1 means missing
                ranks[i, pref] = j

    return ranks


@numba.njit
def _ranks_from_preferences_1d(preferences: np.ndarray) -> np.ndarray:
    ranks = np.full_like(preferences, -1)

    for i in range(preferences.shape[0]):
        pref = preferences[i]

        if pref != -1:
            ranks[pref] = i

    return ranks


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
    if preferences.ndim == 2:  # workaround for numba
        return _ranks_from_preferences_2d(preferences)
    elif preferences.ndim == 1:
        return _ranks_from_preferences_1d(preferences)

    raise ValueError(f'Preferences must be a 1D or 2D array, got {preferences.ndim}D.')


def preferences_from_ranks(preferences: np.ndarray) -> np.ndarray:
    return np.argsort(preferences)


@numba.njit(parallel=True)
def sum_spearmanr(X: np.ndarray, y: np.ndarray, cached_ranks: Tuple[np.ndarray, np.ndarray] = None) -> int:
    """Sums all Spearman's rho from `y` to each row vector in `X`"""
    rhos = np.empty(X.shape[0])

    if cached_ranks is None:
        X_ranks = _ranks_from_preferences_2d(X)
        y_ranks = _ranks_from_preferences_1d(y)
    else:
        X_ranks, y_ranks = cached_ranks

    for i in numba.prange(X.shape[0]):
        rhos[i] = spearmanr(X[i], y, (X_ranks[i], y_ranks))

    return rhos.sum()


@numba.njit(parallel=True)
def sum_kendall_tau(X: np.ndarray, y: np.ndarray) -> int:
    """Sums all the Kendall tau distances from `y` to each row vector in `X`"""
    taus = np.empty(X.shape[0])

    for i in numba.prange(X.shape[0]):
        taus[i] = small_kendall_tau(X[i], y)

    return taus.sum()


def compute_preferences_kendall_tau(X_prefs: np.ndarray, y_prefs: np.ndarray) -> int:
    return sum_kendall_tau(ranks_from_preferences(X_prefs), ranks_from_preferences(y_prefs))


@numba.njit
def spearmanr(a: np.ndarray, b: np.ndarray,cached_ranks: Tuple[np.ndarray, np.ndarray] = None) -> float:
    """Computes the Spearman's rho between two preference arrays."""
    if cached_ranks is None:
        a = _ranks_from_preferences_1d(a)
        b = _ranks_from_preferences_1d(b)
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
            if ((a[i] < a[j] and b[i] > b[j]) or (a[i] > a[j] and b[i] < b[j])) and \
                    a[i] != -1 and b[i] != -1 and a[j] != -1 and b[j] != -1:  # check for missing values
                tau += 1

    return tau


def sample_random_preferences(m: int, n: int) -> np.ndarray:
    rand_prefs = [np.arange(n) for _ in range(m)]

    for x in rand_prefs:
        np.random.shuffle(x)

    return np.array(rand_prefs)
