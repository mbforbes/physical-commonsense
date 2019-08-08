"""Metrics (confusion matrices, F1 calcuation, etc.)."""

import code
import logging
from typing import Tuple, List, Dict, Union, Optional, Callable, Any

import numpy as np
from scipy import stats


def cms(
    y_hat: np.ndarray, y: np.ndarray, y_labels: List[str]
) -> Dict[int, Dict[str, Any]]:
    """Computes confusion matrices (CMs).

    Returns for each label subgroup of index i, {i: {'overall': cm, 'per-item':
        {item: cm}
    }.

    y_labels are of form "a/b/..." where "/" separates subgroups. E.g., if they are all
    of the form "a/b", then there would be 2 label subgroups.

    Each subgroup has its own dict of results. 'overall' is the 2x2 confusion matrix of
    overall performance in that subgroup (e.g., overall "a" performance. 'per-item is a
    mapping from each item in the subgroup to the confusion matrix for all of its
    instances (e.g., overall "a = banana" performance).

    Remember that cm[i][j] is number truly in group i but predicted to be in j.
    """
    # in the models, we work with an extra dimension even though we predict 0/1.
    y_hat = y_hat.squeeze()
    y = y.squeeze()

    # greedy init w/ first y_label
    res: Dict[int, Dict[str, Any]] = {}
    for i in range(len(y_labels[0].split("/"))):
        res[i] = {"overall": np.zeros((2, 2)), "per-item": {}}

    # step through each pred vs actual, incrementing the overall res and the item.
    for i, y_label in enumerate(y_labels):
        want = y[i]
        got = y_hat[i]
        subgroups = y_label.split("/")
        for j, item in enumerate(subgroups):
            # code.interact(local=dict(globals(), **locals()))
            res[j]["overall"][want][got] += 1
            if item not in res[j]["per-item"]:
                res[j]["per-item"][item] = np.zeros((2, 2))
            res[j]["per-item"][item][want][got] += 1

    return res


def prf1(cm: np.ndarray) -> Tuple[float, float, float]:
    """Returns (precision, recall, f1) from a provided 2x2 confusion matrix.

    We special case a few F1 situations where the F1 score is technically undefined or
    pathological. For example, if there are no 1s to predict, 1.0 is returned for
    p/r/f1.
    """
    # cm: cm[i][j] is number truly in group i but predicted to be in j
    tp = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]
    total_1s = tp + fn

    # precision undefined if tp + fp == 0, i.e. no 1s were predicted.
    if tp + fp == 0:
        # precision should not be penalized for not predicting anything.
        precision = 1.0
    else:
        # normal precision
        precision = tp / (tp + fp)

    # recall undefined if tp + fn == total_1s == 0
    if total_1s == 0:
        # couldn't have predicted any 1s because there weren't any. should not
        # penalize recall.
        recall = 1.0
    else:
        # normal recall
        recall = tp / (tp + fn)

    # f1 undefined if precision + recall == 0.
    if precision + recall == 0:
        # if precision and recall are both 0, f1 should just be 0.
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def mc_nemar(results_1: np.ndarray, results_2: np.ndarray) -> float:
    """
    Does McNemar's test on two sets of results, and returns the p value.

    Both inputs should be y-length int arrays with binary values (1 for correct, 0 for
    incorrect).
    """
    assert results_1.shape == results_2.shape

    # slow version for now. contingency table = [[a, b], [c, d]]; we're using incorrect
    # as first row and column, and correct as second row and column. (this is flipped
    # from wikipedia, but the mcnemar formula (below) is agnostic to this). results_1
    # indexes row, results_2 indexes column.
    contingency = np.zeros((2, 2))
    for i in range(len(results_1)):
        contingency[results_1[i], results_2[i]] += 1

    # mcnemar
    b = contingency[0][1]
    c = contingency[1][0]
    chi2 = (b - c) ** 2 / (b + c)
    return stats.chi2.sf(chi2, 1)


def report(
    y_hat: np.ndarray, y: np.ndarray, y_labels: List[str], task_labels: List[str]
) -> Tuple[float, float, Dict[str, float], Dict[int, Dict[str, Any]], np.ndarray]:
    """Shorthand function for computing metrics and printing summary.

    Returns 5-tuple: (
        accuracy,
        micro F1,
        dict of category -> macro f1 score,
        category cms (see return of cms()),
        1-D int vector of len y with binary values: 1 for correct, 0 for incorrect
    )
    """
    # accuracy
    acc = (y_hat == y).sum() / len(y)
    txt = ["Acc: {:.3f}".format(acc)]

    # get cms
    category_cms = cms(y_hat, y, y_labels)

    # micro f1 is the same for any category, because it's the sum of the confusion
    # matrices. we pick category 0 arbitrarily.
    _, _, micro_f1 = prf1(category_cms[0]["overall"])
    txt.append("Micro F1: {:.3f}".format(micro_f1))

    # macro f1 is a bit more involved.
    macro_f1s: Dict[str, float] = {}
    for i, results in category_cms.items():
        sum_p, sum_r = 0.0, 0.0
        n = 0
        for cm in results["per-item"].values():
            # don't count "all-0" items towards the cateogry macro. (i.e., skip if total
            # 1s = tp + fn = 0)
            if cm[1][1] + cm[1][0] == 0:
                continue
            precision, recall, _ = prf1(cm)
            sum_p += precision
            sum_r += recall
            n += 1

        macro_precision = sum_p / n
        macro_recall = sum_r / n
        macro_f1 = (
            0
            if macro_precision == 0 and macro_recall == 0
            else (
                2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
            )
        )
        macro_f1s[task_labels[i]] = macro_f1
        txt.append("{} macro F1: {:.3f}".format(task_labels[i], macro_f1))

    logging.info("\t" + ", ".join(txt))
    return acc, micro_f1, macro_f1s, category_cms, (y_hat == y).astype(int).squeeze()
