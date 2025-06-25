from typing import List
import numpy as np

"""
Source code for these functions was taken from the recmetrics library. The library is no longer maintained with many dependency errors,
so I just copied the functions I need. Source link to the file these functions were taken from: 
https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
"""

def _precision(predicted, actual):
    prec = [value for value in predicted if value in actual]
    prec = float(len(prec)) / float(len(predicted))
    return prec

def _ark(actual: list, predicted: list, k=10) -> float:
    """
    Computes the average recall at k.
    Parameters
    ----------
    actual : list
        A list of actual items to be predicted
    predicted : list
        An ordered list of predicted items
    k : int, default = 10
        Number of predictions to consider
    Returns:
    -------
    score : float
        The average recall at k.
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / len(actual)


def mark(actual: List[list], predicted: List[list], k=10) -> float:
    """
    Computes the mean average recall at k.
    Parameters
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        mark: float
            The mean average recall at k (mar@k)
    """
    if len(actual) != len(predicted):
        raise AssertionError("Length mismatched")

    return np.mean([_ark(a, p, k) for a, p in zip(actual, predicted)])