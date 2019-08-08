"""Run experiments (GloVe, Dep-Embs, ELMo).

This actually does the final train-and-test runs of the above models. (Note that Bert is
run separately in bert.py as it is fine-tuned end-to-end.)

Cross validation is performed in models.py; there are extensive comments there about the
hyperparameters.
"""

import logging
import os
from torch import nn

from pc.data import Task, Variant, TASK_MEDIUMHAND
from pc import models
from pc import util


# logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def main() -> None:
    config = {
        "epochs": 300,
        "report_every": 20,
        "batch_size": 100000,
        "lr": 1.00,
        "l2": 0.01,
        "decay_epochs": 30,
        "decay_scale": 0.8,
        "center": True,
    }
    variants = [(600, Variant.Glove), (600, Variant.DepEmbs), (2048, Variant.Elmo)]
    # task, which macro f1s to extract
    tasks = [
        (Task.Abstract_ObjectsProperties, ["object", "property"]),
        (Task.Situated_ObjectsProperties, ["object", "property"]),
        (Task.Situated_ObjectsAffordances, ["object", "affordance"]),
        (Task.Situated_AffordancesProperties, ["affordance", "property"]),
    ]
    lines = []
    for d, variant in variants:
        model = models.mlp(d, 0.0, 128, nn.ReLU, 0.0, 1)
        nums = []
        for task, mf1_labs in tasks:
            # run
            _, _, macro_f1s, _, per_datum = models.train_and_test(
                task, variant, model, config
            )

            # get metrics
            for mf1_lab in mf1_labs:
                nums.append(macro_f1s[mf1_lab])

            # save overall confusion matrix for computing statistical significance. both
            # categories (0 and 1) have the same overall confusion matrix, so we pick 0
            # arbitrarily.
            path = os.path.join(
                "data",
                "results",
                "{}-{}-perdatum.txt".format(variant.name, TASK_MEDIUMHAND[task]),
            )
            with open(path, "w") as f:
                f.write(util.np2str(per_datum) + "\n")

        # save for report
        lines.append(
            variant.name + "," + ",".join(["{:.2f}".format(num) for num in nums])
        )
        logging.info("")  # newline between variants

    # print report
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
