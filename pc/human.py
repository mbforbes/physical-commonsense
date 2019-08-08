"""Human evaluation!"""

import argparse
import code
import logging
import os
import random
from typing import Tuple, List, Set, Dict, Any, Callable

import numpy as np
import pandas as pd

from pc import data
from pc import metrics
from pc.data import Task, TASK_MEDIUMHAND


# logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def create() -> None:
    """This was run once to build the set that was annotated by humans."""
    phase = "round1"
    for task in Task:
        _, _, test = data.get(task)
        labels, _, gold_2d = test
        gold = gold_2d.squeeze()
        index = random.sample(range(len(labels)), 100)
        selected_labels = np.array(labels)[index].tolist()
        selected_gold = gold[index].tolist()

        label_path = os.path.join(
            "data", "human", "{}-{}-labels.txt".format(TASK_MEDIUMHAND[task], phase)
        )
        with open(label_path, "w") as f:
            for label in selected_labels:
                f.write(label + "\n")

        gold_path = os.path.join(
            "data", "human", "{}-{}-gold.txt".format(TASK_MEDIUMHAND[task], phase)
        )
        with open(gold_path, "w") as f:
            for gold in selected_gold:
                f.write(str(gold) + "\n")

    # code.interact(local=dict(globals(), **locals()))


def get_gold(path: str, lim: int = None) -> np.ndarray:
    with open(path, "r") as f:
        arr = np.array([int(line.strip()) for line in f.readlines()], dtype=int)
        if lim is not None:
            return arr[:lim]
        return arr


def get_labels(path: str, lim: int = None) -> List[str]:
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        if lim is not None:
            return lines[:lim]
        return lines


def get_anns(path: str, lim: int = None) -> np.ndarray:
    df = pd.read_csv(path)
    if lim is not None:
        df = df[:lim]
    return df["label (0 = false, 1 = true)"].values.astype(int)


def evaluate_round1() -> None:
    """This is run to score human annotations."""
    phase = "round1"
    lim = 50
    for task in Task:
        logging.info(task)
        gold_path = os.path.join(
            "data", "human", "{}-{}-gold.txt".format(TASK_MEDIUMHAND[task], phase)
        )
        gold_data = get_gold(gold_path, lim=lim)

        label_path = os.path.join(
            "data", "human", "{}-{}-labels.txt".format(TASK_MEDIUMHAND[task], phase)
        )
        labels = get_labels(label_path, lim=lim)

        ann_path = os.path.join(
            "data",
            "human",
            "{}-{}-annotations-first50.csv".format(TASK_MEDIUMHAND[task], phase),
        )
        ann_data = get_anns(ann_path, lim=lim)

        task_labels = data.TASK_LABELS[task]

        acc, micro_f1, macro_f1s, category_cms, per_datum = metrics.report(
            ann_data, gold_data, labels, task_labels
        )


def main() -> None:
    # create()
    evaluate_round1()


if __name__ == "__main__":
    main()
