"""Baseline models.

It turns out there are a few ways of doing the "simplest" thing, like majority.
"""

import argparse
import code
from collections import Counter
import logging
import os
import typing
from typing import List, Tuple, Set, Dict, Any, Optional, Callable, Iterator

import numpy as np

from pc import data
from pc import metrics
from pc import util
from pc.data import Task, TASK_MEDIUMHAND


# logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def _random(
    y_labels_train: List[str],
    y_train: np.ndarray,
    y_labels_test: List[str],
    y_test_shape: Tuple[int, ...],
) -> np.ndarray:
    return np.random.uniform(size=y_test_shape).round().astype("int")


def _all_0(
    y_labels_train: List[str],
    y_train: np.ndarray,
    y_labels_test: List[str],
    y_test_shape: Tuple[int, ...],
) -> np.ndarray:
    return np.zeros(y_test_shape).round().astype("int")


def _all_1(
    y_labels_train: List[str],
    y_train: np.ndarray,
    y_labels_test: List[str],
    y_test_shape: Tuple[int, ...],
) -> np.ndarray:
    return np.ones(y_test_shape).round().astype("int")


def _maj_naive(
    y_labels_train: List[str],
    y_train: np.ndarray,
    y_labels_test: List[str],
    y_test_shape: Tuple[int, ...],
) -> np.ndarray:
    if y_train.sum() / len(y_train) >= 0.5:
        return _all_1(y_labels_train, y_train, y_labels_test, y_test_shape)
    return _all_0(y_labels_train, y_train, y_labels_test, y_test_shape)


def _maj_cat(
    y_labels_train: List[str],
    y_train: np.ndarray,
    y_labels_test: List[str],
    y_test_shape: Tuple[int, ...],
) -> np.ndarray:
    # we use the right hand side of a label (e.g., "b" for "a/b") to aggregate votes for
    # that category, picking the majority.
    cats: Dict[str, typing.Counter[int]] = {}
    for i in range(len(y_labels_train)):
        cat = y_labels_train[i].split("/")[-1]
        val = y_train[i][0]
        if cat not in cats:
            cats[cat] = Counter()
        cats[cat][val] += 1
    majs = {cat: c.most_common()[0][0] for cat, c in cats.items()}

    # use majority label. backup to 1 if not seen.
    assert y_test_shape == (len(y_labels_test), 1)
    res = np.zeros(y_test_shape, dtype=int)
    for i in range(len(y_labels_test)):
        cat = y_labels_test[i].split("/")[-1]
        res[i][0] = majs.get(cat, 1)
    return res


def baseline(
    func: Callable[[List[str], np.ndarray, List[str], Tuple[int, ...]], np.ndarray],
    name: str,
    shortname: str,
) -> str:
    # settings
    tasks = [
        (Task.Abstract_ObjectsProperties, ["object", "property"]),
        (Task.Situated_ObjectsProperties, ["object", "property"]),
        (Task.Situated_ObjectsAffordances, ["object", "affordance"]),
        (Task.Situated_AffordancesProperties, ["affordance", "property"]),
    ]
    nums = []
    for task, mf1_labs in tasks:
        logging.info("Running {} baseline for {}".format(name, task.name))
        train_data, test_data = data.get(task)
        labels_train, y_train = train_data
        labels_test, y_test = test_data
        y_test_hat = func(labels_train, y_train, labels_test, y_test.shape)
        _, _, macro_f1s, _, per_datum = metrics.report(
            y_test_hat, y_test, labels_test, data.TASK_LABELS[task]
        )
        for mf1_lab in mf1_labs:
            nums.append(macro_f1s[mf1_lab])

        # write full results to file
        path = os.path.join(
            "data",
            "results",
            "{}-{}-perdatum.txt".format(shortname, TASK_MEDIUMHAND[task]),
        )
        with open(path, "w") as f:
            f.write(util.np2str(per_datum) + "\n")

    logging.info("")
    return name + "," + ",".join(["{:.2f}".format(num) for num in nums])


def main() -> None:
    # header
    lines = [
        ",Abstract,,Situated,,,,,",
        ",OP,,OP,,OA,,AP,",
        "method,obj,prop,obj,prop,obj,aff,aff,prop",
    ]

    # baselines to report
    lines.append(baseline(_random, "random", "Random"))
    # baseline(_all_0, "all-0")
    # baseline(_all_1, "all-1")
    # baseline(_maj_naive, "majority: naive")
    lines.append(baseline(_maj_cat, "majority: by category", "Majority"))

    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
