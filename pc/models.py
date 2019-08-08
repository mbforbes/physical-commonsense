"""Machine learning models running on our tasks & data.

Note that these are the lightweight models (MLPs) that run *on top* of the real
star-of-the-show "models:" GloVe, DepEmbs, ELMo. (Note that Bert lives all on its own in
a separate file: bert.py.)

This file implements:
- the model itself (mlp()) and utils (init_weights())
- batching (make_batcher())
- training (train(), test())
- training wrappers: cross-validation (cv()), train then test (train_and_test())

The main function contains some demo code for plyaing around with hyperparamters and
performing cross-validation.
"""

import argparse
import code
import logging
from typing import List, Tuple, Any, Callable, Iterable, Optional, Dict

import numpy as np
from sklearn import model_selection
import torch
from torch import nn
from tqdm import tqdm

from pc import data
from pc.data import Task, Variant
from pc import metrics

# logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#
# models (+ closely related utils)
#


def mlp(
    d_in: int,
    input_dropout: float,
    h: int,
    activation: Any,
    inner_dropout: float,
    d_out: int,
) -> nn.Module:
    return nn.Sequential(
        nn.Dropout(input_dropout),
        nn.Linear(d_in, h),
        activation(),
        nn.Dropout(inner_dropout),
        nn.Linear(h, d_out),
        nn.Sigmoid(),
    )


def init_weights(m: nn.Module):
    if type(m) in [nn.Linear, nn.Conv1d, nn.Conv2d]:
        nn.init.normal_(m.weight)
        m.bias.data.fill_(0.01)


#
# functions
#


def make_batcher(
    x: torch.Tensor, y: torch.Tensor, batch_size: int
) -> Callable[[], Iterable[Tuple[torch.Tensor, torch.Tensor]]]:
    """Returns a function that can be called to yield batches over x and y."""
    assert x.shape[0] == y.shape[0]

    idxes = range(0, x.shape[0], batch_size)

    def batcher() -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        for start in idxes:
            end = start + batch_size
            yield x[start:end], y[start:end]

    return batcher


def train(
    model: nn.Module, x_np: np.ndarray, y_np: np.ndarray, config: Dict[str, Any]
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Trains model."""
    # model + harness init
    model = model.float().apply(init_weights).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=config["l2"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, config["decay_epochs"], gamma=config["decay_scale"]
    )

    # move data to GPU
    x = torch.from_numpy(x_np).float().to(DEVICE)
    y = torch.from_numpy(y_np).float().to(DEVICE)
    assert x.shape[0] == y.shape[0]

    # center data
    centering = None
    if config["center"]:
        means = x.mean(0)
        stds = x.std(0)
        x = (x - means) / stds
        centering = (means, stds)

    # train
    model.train()
    batcher = make_batcher(x, y, config["batch_size"])
    for epoch in range(config["epochs"]):
        scheduler.step()  # type: ignore
        for batch_x, batch_y in batcher():
            batch_y_hat = model(batch_x)
            loss = loss_fn(batch_y_hat, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % config["report_every"] == 0:
            acc = 100 * (
                (batch_y_hat.round().int() == batch_y.int()).sum().item()
                / len(batch_y)  # type: ignore
            )
            logging.info(
                "Epoch {}. Train acc: {:4.2f}. Train loss: {:.4f}".format(
                    epoch, acc, loss.item()
                )
            )

    # we return centering for eval time
    return centering


def test(
    model: nn.Module,
    x_np: np.ndarray,
    y_np: np.ndarray,
    centering: Optional[Tuple[torch.Tensor, torch.Tensor]],
    config: Dict[str, Any],
) -> np.ndarray:
    """Tests model. Returns y_hat 0/1 predictions as 2D int vector of shape (n, 1)."""
    # move data to gpu
    x = torch.from_numpy(x_np).float().to(DEVICE)
    y = torch.from_numpy(y_np).float().to(DEVICE)
    assert x.shape[0] == y.shape[0]

    # apply centering
    if centering is not None:
        means, stds = centering
        x = (x - means) / stds

    # eval
    model.eval()
    y_hat = model(x).round().int().cpu().numpy()
    return y_hat


def cv(task: Task, variant: Variant, model: nn.Module, config: Dict[str, Any]) -> None:
    """Run cross validation (for hyperparameter selection)."""
    logging.info("Running cross validation for {}, {}".format(task.name, variant.name))
    # get data. cv uses train data only.
    train_data, _ = data.get(task)
    labels, y_np = train_data
    labels_np = np.array(labels)
    x_np = data.features(task, variant, labels)

    # run k-fold cross validation
    folder = model_selection.KFold(n_splits=5, shuffle=True)
    overall_y_hat = np.zeros_like(y_np)
    for i, (train_index, test_index) in enumerate(folder.split(x_np)):
        # logging.info("Fold {}".format(i))
        x_train, x_test = x_np[train_index], x_np[test_index]
        y_train, y_test = y_np[train_index], y_np[test_index]
        labels_train, labels_test = (labels_np[train_index], labels_np[test_index])

        centering = train(model, x_train, y_train, config)
        y_test_hat = test(model, x_test, y_test, centering, config)

        # Uncomment the next line to report on individual folds.
        # metrics.report(y_test_hat, y_test, labels_test, data.TASK_LABELS[task])

        # Save results into overall aggregate.
        overall_y_hat[test_index] = y_test_hat

    # Report on overall results.
    metrics.report(overall_y_hat, y_np, labels, data.TASK_LABELS[task])


def train_and_test(
    task: Task, variant: Variant, model: nn.Module, config: Dict[str, Any]
) -> Tuple[float, float, Dict[str, float], Dict[int, Dict[str, Any]]]:
    """Run a final train + test run over a task."""
    logging.info("Running train+test for {}, {}".format(task.name, variant.name))
    # get data. cv uses train data only.
    train_data, test_data = data.get(task)

    # train
    labels_train, y_train_np = train_data
    x_train_np = data.features(task, variant, labels_train)
    centering = train(model, x_train_np, y_train_np, config)

    # test
    labels_test, y_test_np = test_data
    x_test_np = data.features(task, variant, labels_test)
    y_test_hat = test(model, x_test_np, y_test_np, centering, config)
    return metrics.report(y_test_hat, y_test_np, labels_test, data.TASK_LABELS[task])


def main() -> None:
    # model hyperparameters tuned here! some notes:
    # - input size (e.g., 600) fixed (just input features size)
    # - input dropout (e.g., 0.0) saw slight degredation with > 0 values (e.g., 0.2, 0.5)
    # - hidden size (e.g., 128) saw slight degredation with others (e.g., 32, 64, 256)
    # - activation (e.g., nn.ReLU) saw slight degredation with others (e.g., tanh, sigmoid)
    # - inner dropout (e.g., 0.0) saw slight degredation with > 0 values (e.g., 0.2, 0.5)
    # - output features (e.g., 1) fixed based on task (predicting compatability)
    model = mlp(600, 0.0, 128, nn.ReLU, 0.0, 1)
    logging.info("Model:")
    logging.info(model)
    # optimization hyperparameters tuned here! some notes:
    # - epochs (e.g., 300) tuned where models consistently plateau. with huge batch size
    #          (set below) note that this is few updates. This, however, saw by far the
    #          fastest convergence (in wall = human time). more details in batch size.
    # - report_every (e.g., 301) just is how often the model prints accuracy and loss
    #                (on the training set) during training. Set > epochs to disable.
    # - batch_size (e.g., 100k) is batch size for SGD. We can do huge batches here
    #              because of the nature of our task. Setting to smaller (e.g., values
    #              you're probably used to, like 128) yields to much slower convergence
    #              (wall time) and empirically worse results. (This is, of course,
    #              because you have to do way more batches = backprops and model updates
    #              per epoch.) Since our data are small enough, and our GPU can fit the
    #              data, it turns out (or at least seems to be) that if you can do plain
    #              old GD and use your entire dataset to calculate the gradient before
    #              updating, and that's just as fast as using smaller batches, this
    #              works pretty dang well.
    # - lr (e.g., 1.00) works pretty well
    # - l2 (e.g., 0.01), AKA "weight decay" for you deep learning enthusiasts, is
    #      absurdly vital to get right. Try 0.1 or 1.0? Your F1 score is zero. Try
    #      0.001? F1 score of zero again.
    # - decay_epochs (e.g., 30) is how often to decay the learning rate. This works
    #                pretty well given the other choices.
    # - decay_scale (e.g., 0.8) is how much to decay the learning rate by
    #               (multiplicatively). Again, this strikes a good balance.
    # - center (e.g., True) is whether to make data zero mean and unit variance, then
    #          save this tranformation (calculated on the training data) and apply it to
    #          the test data.
    config = {
        "epochs": 300,
        "report_every": 301,
        "batch_size": 100000,
        "lr": 1.00,
        "l2": 0.01,
        "decay_epochs": 30,
        "decay_scale": 0.8,
        "center": True,
    }
    logging.info("Config:")
    logging.info(config)
    cv(Task.Abstract_ObjectsProperties, Variant.Glove, model, config)
    # train_and_test(Task.Situated_AffordancesProperties, Variant.Glove, model, config)


if __name__ == "__main__":
    main()
