"""Turns per-datum results into per-category results.

Example
    python -m scripts.perdatum_to_category

"""

import code  # code.interact(local=dict(globals(), **locals()))
from typing import List, Tuple, Dict, Set, Any, Optional, Callable

import numpy as np

from pc import data
from pc import metrics
from pc import util


def main() -> None:
    # settings. (onetime use so no flags.)
    perdatum_path = "data/results/Bert-situated-AP-perdatum.txt"
    task = data.Task.Situated_AffordancesProperties

    # load per-datum output
    with open(perdatum_path, "r") as f:
        perdatum = util.str2np(f.read())

    # get test data: labels and groundtruth y-values
    _, test_data = data.get(task)
    labels, y = test_data
    y = y.squeeze()

    # the per-datum is y_hat == y. we want to recover y_hat so we can pass it back into
    # metrics to easily re-compute everything we need. probably a vectorized op that can
    # do this but oh well.
    y_hat = np.zeros_like(y)
    for i in range(len(y)):
        y_hat[i] = y[i] if perdatum[i] else 1 - y[i]

    # sanity check
    assert len(labels) == len(y_hat)
    assert len(y) == len(y_hat)

    _, _, _, category_cms, _ = metrics.report(y_hat, y, labels, data.TASK_LABELS[task])

    # write out
    task_short = data.TASK_SHORTHAND[task]
    for i in [0, 1]:
        # e.g., "O" for objects, "P" for properties
        cat_short = task_short[i]
        out_path = "data/results/{}-{}-{}.txt".format("Bert", task_short, cat_short)
        print("Writing {} results to {}".format(cat_short, out_path))
        with open(out_path, "w") as f:
            for item, cm in category_cms[i]["per-item"].items():
                f.write("{} {}\n".format(item, util.np2str(cm)))


if __name__ == "__main__":
    main()
