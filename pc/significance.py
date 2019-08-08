"""Run statistical significance tests on results.

Specifically, we run a McNemar's test on the per-datum output.
"""

import argparse
import code
import logging
import os
from typing import Tuple, List, Set, Dict, Any, Callable

import numpy as np

from pc import metrics
from pc import util

# logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def get_symbol(p: float) -> str:
    """Returns statistically significant symbol to use for p value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def get_data(task: str, method: str) -> np.ndarray:
    path = os.path.join("data", "results", "{}-{}-perdatum.txt".format(method, task))
    with open(path, "r") as f:
        contents = f.read().strip()
    return util.str2np(contents)


def main() -> None:
    tasks = ["abstract-OP", "situated-OP", "situated-OA", "situated-AP"]
    methods = ["Random", "Majority", "Glove", "DepEmbs", "Bert", "Elmo"]

    # best is the best model per task. we use to compare vs others. must be in same
    # order as tasks.
    best = ["Bert", "Bert", "Bert", "Bert"]
    assert len(tasks) == len(best)

    results: Dict[str, Dict[str, str]] = {}
    for i, task in enumerate(tasks):
        print(task + ":")
        results[task] = {}
        base = best[i]
        base_data = get_data(task, base)
        results[task][base] = "(base)"
        for method in methods:
            # only run test if not self
            if method != base:
                other_data = get_data(task, method)
                p = metrics.mc_nemar(base_data, other_data)
                # results[task][method] = str(p)
                results[task][method] = get_symbol(p)

            # always display
            print("- {}: {}".format(method, results[task][method]))
        print()


if __name__ == "__main__":
    main()
