"""
Load data for GloVe, DepEmbs, and ELMo.

(BERT does its own thing because it is fine-tuned end-to-end.)

This file breaks the notion of "data" into two main groups: Task and Variants.

Each Task has its associated set of labels and y values (which are "yes or no" answers).

Given a Task, a particular variant (like GloVe) will load its X matrix --- i.e., its
embedding representation of the input. We build these in advance to save time.
"""

import argparse
import code
from enum import Enum, auto
import logging
import random
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

#
# task
#


class Task(Enum):
    Abstract_ObjectsProperties = auto()
    Situated_ObjectsProperties = auto()
    Situated_ObjectsAffordances = auto()
    Situated_AffordancesProperties = auto()


# TaskData = 2-tuple of (labels, y data)
TaskData = Tuple[List[str], np.ndarray]


# These are the categories used for the labels in each task. E.g., the X and Y labels
# for the Abstract Objects Properties task is "object/property" for each datum.
TASK_LABELS = {
    Task.Abstract_ObjectsProperties: ["object", "property"],
    Task.Situated_ObjectsProperties: ["object", "property"],
    Task.Situated_ObjectsAffordances: ["object", "affordance"],
    Task.Situated_AffordancesProperties: ["affordance", "property"],
}

TASK_SHORTHAND = {
    Task.Abstract_ObjectsProperties: "OP",
    Task.Situated_ObjectsProperties: "OP",
    Task.Situated_ObjectsAffordances: "OA",
    Task.Situated_AffordancesProperties: "AP",
}

TASK_MEDIUMHAND = {
    Task.Abstract_ObjectsProperties: "abstract-OP",
    Task.Situated_ObjectsProperties: "situated-OP",
    Task.Situated_ObjectsAffordances: "situated-OA",
    Task.Situated_AffordancesProperties: "situated-AP",
}

TASK_REV_MEDIUMHAND = {v: k for k, v in TASK_MEDIUMHAND.items()}


def _read(path: str) -> List[str]:
    with open(path, "r") as f:
        return [l.strip() for l in f.readlines() if len(l.strip()) > 0]


def _expand(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """Expands r1: c1 c2 c3 into r1/c1, r1/c2, r1/c3. Returns labels, data.

    Takes df shape (n, d) and returns the combination of every row and column.
    This is n*d entries. The labels are "row/col" and the data is a 2D array of shape
    (n*d, 1) with each value.
    """
    rows = [row for row, _ in df.iterrows()]
    cols = df.columns.to_list()
    labels = []
    for row in rows:
        for col in cols:
            labels.append("{}/{}".format(row, col))

    return labels, np.expand_dims(df.to_numpy().reshape(-1), 1)


def _train_test_df_split(
    df: pd.DataFrame, train_uid_path: str, test_uid_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Helper for task data getters who split a df by index."""
    train_obj_uids = set(_read(train_uid_path))
    test_obj_uids = set(_read(test_uid_path))

    train_df = df[df.index.isin(train_obj_uids)]
    test_df = df[df.index.isin(test_obj_uids)]

    return train_df, test_df


def _train_test_df_expand(
    df: pd.DataFrame, train_uid_path: str, test_uid_path: str
) -> Tuple[TaskData, TaskData]:
    """Helper for task data getters who split a df by index, then expand the features."""
    # get our object UID splits, and split df
    train_df, test_df = _train_test_df_split(df, train_uid_path, test_uid_path)

    train_labels, train_y = _expand(train_df)
    test_labels, test_y = _expand(test_df)
    return ((train_labels, train_y), (test_labels, test_y))


def _get_abstract_objects_properties() -> Tuple[TaskData, TaskData]:
    # read all abstract data. map {-2, -1, 0} -> 0, {1} -> 1.
    df = pd.read_csv("data/pc/abstract.csv", index_col="objectUID")
    for prop in df.columns:
        df[prop] = df[prop].apply(lambda x: 0 if x <= 0 else 1)

    return _train_test_df_expand(
        df,
        "data/pc/abstract-train-object-uids.txt",
        "data/pc/abstract-test-object-uids.txt",
    )


def _get_situated_objects_properties() -> Tuple[TaskData, TaskData]:
    # read in situated properties data, index by object. remove extra data cols.
    df = pd.read_csv("data/pc/situated-properties.csv", index_col="objectUID").drop(
        columns=["cocoImgID", "cocoAnnID"]
    )

    return _train_test_df_expand(
        df,
        "data/pc/situated-train-object-uids.txt",
        "data/pc/situated-test-object-uids.txt",
    )


def _get_situated_objects_affordances() -> Tuple[TaskData, TaskData]:
    # read in situated affordances data, index by object. remove extra data cols.
    df = pd.read_csv(
        "data/pc/situated-affordances-sampled.csv", index_col="objectUID"
    ).drop(columns=["cocoImgID", "cocoAnnID", "objectHuman"])

    # grab the 3 points that are on, and sample 3 that are off
    train_df, test_df = _train_test_df_split(
        df,
        "data/pc/situated-train-object-uids.txt",
        "data/pc/situated-test-object-uids.txt",
    )

    results: List[TaskData] = []
    for sub_df in [train_df, test_df]:
        data: List[int] = []
        labels: List[str] = []
        for _, row in sub_df.iterrows():
            obj = row.name

            # record positive examples
            for aff_yes in row["affordancesYes"].split(","):
                labels.append("{}/{}".format(obj, aff_yes))
                data.append(1)

            # record negative examples
            for aff_no in row["affordancesNo"].split(","):
                labels.append("{}/{}".format(obj, aff_no))
                data.append(0)

        results.append((labels, np.expand_dims(np.array(data), 1)))

    return (results[0], results[1])


def _get_situated_affordances_properties() -> Tuple[TaskData, TaskData]:
    # read in BOTH the situated affordances and properties data frames. Conceptually
    # join based on the coco annotation id (uniqe + matching across both).
    aff_df_full = pd.read_csv(
        "data/pc/situated-affordances-sampled.csv", index_col="objectUID"
    ).drop(columns=["cocoImgID", "objectHuman"])
    prop_df_full = pd.read_csv(
        "data/pc/situated-properties.csv", index_col="objectUID"
    ).drop(columns=["cocoImgID"])

    aff_train_df, aff_test_df = _train_test_df_split(
        aff_df_full,
        "data/pc/situated-train-object-uids.txt",
        "data/pc/situated-test-object-uids.txt",
    )
    prop_train_df, prop_test_df = _train_test_df_split(
        prop_df_full,
        "data/pc/situated-train-object-uids.txt",
        "data/pc/situated-test-object-uids.txt",
    )

    results: List[TaskData] = []
    for aff_df, prop_df in [(aff_train_df, prop_train_df), (aff_test_df, prop_test_df)]:
        props = list(prop_df.drop(columns=["cocoAnnID"]).columns)
        data = np.zeros((3 * len(aff_df), len(props)), dtype=int)
        labels: List[str] = []
        # for each object, pick its 3 affordances. set the same property vector for each
        # of those 3 affordances.
        for i, (_, aff_row) in enumerate(aff_df.iterrows()):
            prop_data = (
                prop_df[prop_df["cocoAnnID"] == aff_row["cocoAnnID"]]
                .drop(columns=["cocoAnnID"])
                .to_numpy()
                .squeeze()
            )
            for j, aff in enumerate(aff_row["affordancesYes"].split(",")):
                data[i * 3 + j] = prop_data
                for prop in props:
                    labels.append("{}/{}".format(aff, prop))

        results.append((labels, np.expand_dims(data.reshape(-1), 1)))

    return (results[0], results[1])


def get(task: Task) -> Tuple[TaskData, TaskData]:
    """Returns 2-tuple (train, test).

    Each of them can be None if that split isn't defined for that task.
    """
    if task is Task.Abstract_ObjectsProperties:
        return _get_abstract_objects_properties()
    elif task is Task.Situated_ObjectsProperties:
        return _get_situated_objects_properties()
    elif task is Task.Situated_ObjectsAffordances:
        return _get_situated_objects_affordances()
    elif task is Task.Situated_AffordancesProperties:
        return _get_situated_affordances_properties()
    else:
        raise ValueError("Unknown task: {}".format(task))


#
# features
#


class Variant(Enum):
    Glove = auto()
    DepEmbs = auto()
    Elmo = auto()


def _get_wordembedding_name_map(path: str) -> Dict[str, str]:
    """Reads tsv from path returns mapping from 'uid' col to 'word-embedding' col."""
    obj_df = pd.read_csv(path, delimiter="\t")
    return {row["uid"]: row["word-embedding"] for _, row in obj_df.iterrows()}


def _wordembedding_words_objects_properties(labels: List[str]) -> List[Tuple[str, str]]:
    """Returns list of (obj, prop) words from labels.

    For the abstract or situated objects properties task.
    """
    obj_map = _get_wordembedding_name_map("data/pc/objects.tsv")
    prop_map = _get_wordembedding_name_map("data/pc/properties.tsv")

    res: List[Tuple[str, str]] = []
    for label in labels:
        obj_uid, prop_uid = label.split("/")
        res.append((obj_map[obj_uid], prop_map[prop_uid]))
    return res


def _wordembedding_words_objects_affordances(
    labels: List[str]
) -> List[Tuple[str, str]]:
    """Returns list of (obj, affordance) words from labels.

    For the situated objects affordances task.
    """
    # we map objects, but use affordances as they are.
    obj_map = _get_wordembedding_name_map("data/pc/objects.tsv")
    res: List[Tuple[str, str]] = []
    for label in labels:
        obj_uid, affordance = label.split("/")
        res.append((obj_map[obj_uid], affordance))
    return res


def _wordembedding_words_affordances_properties(
    labels: List[str]
) -> List[Tuple[str, str]]:
    """Returns list of (affordance, property) words from labels.

    For the situated affordances properties task.
    """
    # we map properties, but use affordances as they are.
    prop_map = _get_wordembedding_name_map("data/pc/properties.tsv")
    res: List[Tuple[str, str]] = []
    for label in labels:
        affordance, prop_uid = label.split("/")
        res.append((affordance, prop_map[prop_uid]))
    return res


def _wordembedding(
    task: Task, labels: List[str], archive_path: str, d: int
) -> np.ndarray:
    """Returns 2D feature matrix shaped (n, d*num_words) for list of "a/b/..." tuples."""
    # labels are "A_uid/B_uid/...". have to convert each from uid to glove name.
    # how to do this depends on the task.
    if task is Task.Abstract_ObjectsProperties:
        word_tuples = _wordembedding_words_objects_properties(labels)
    elif task is Task.Situated_ObjectsProperties:
        word_tuples = _wordembedding_words_objects_properties(labels)
    elif task is Task.Situated_ObjectsAffordances:
        word_tuples = _wordembedding_words_objects_affordances(labels)
    elif task is Task.Situated_AffordancesProperties:
        word_tuples = _wordembedding_words_affordances_properties(labels)
    else:
        raise ValueError("Unknown task: {}".format(task))

    # now, we load our archive and build the rep for each word.
    with np.load(archive_path) as archive:
        index = archive["index"]
        word2idx = {word: i for i, word in enumerate(index)}
        matrix = archive["matrix"]
        result = np.zeros((len(labels), d * len(word_tuples[0])))
        for i, words in enumerate(word_tuples):
            for j, word in enumerate(words):
                # if OOV, just leave as 0s
                if word in word2idx:
                    result[i][j * d : (j + 1) * d] = matrix[word2idx[word]]
    return result


def glove(task: Task, labels: List[str]) -> np.ndarray:
    return _wordembedding(
        task, labels, "data/glove/vocab-pc.glove.840B.300d.txt.npz", 300
    )


def dep_embs(task: Task, labels: List[str]) -> np.ndarray:
    return _wordembedding(task, labels, "data/dep-embs/vocab-pc.dep-embs.npz", 300)


def _uids2sentidx(task: Task, labels: List[str]) -> np.ndarray:
    """Given a set of uid1/uid2 labels, return a numerical index for where these labels
    occur as sentences in our sentence corpus.
    """
    df = pd.read_csv("data/sentences/index.csv")
    task_code = TASK_SHORTHAND[task]

    # build up the reverse map only for the task at hand
    uids2idx = {}
    for i, row in df.iterrows():
        if row["task"] == task_code:
            uids2idx[row["uids"]] = i

    # map each label to the row index
    idx = np.array([uids2idx[label] for label in labels])
    assert len(idx) == len(labels)
    return idx


def _ctx_emb(task: Task, labels: List[str], archive_path: str) -> np.ndarray:
    # big files, but only 2.7 seconds on my machine. so we just load the whole thing.
    with np.load(archive_path) as archive:
        matrix = archive["matrix"]
        idx = _uids2sentidx(task, labels)
        return matrix[idx]


def elmo(task: Task, labels: List[str]) -> np.ndarray:
    return _ctx_emb(task, labels, "data/elmo/sentences.elmo.npz")


def features(task: Task, variant: Variant, x_labels: List[str]) -> np.ndarray:
    """Returns the (n, d) feature matrix for x_labels on task using variant."""
    if variant is Variant.Glove:
        return glove(task, x_labels)
    elif variant is Variant.DepEmbs:
        return dep_embs(task, x_labels)
    elif variant is Variant.Elmo:
        return elmo(task, x_labels)
    else:
        raise ValueError("Unknown task: {}".format(task))


#
# testing
#


def main() -> None:
    """Just for testing out functionality."""
    train, test = get(Task.Abstract_ObjectsProperties)
    assert train is not None

    # train_labels, train_y = train
    # train_features = elmo(Task.Abstract_ObjectsProperties, train_labels)

    code.interact(local=dict(globals(), **locals()))


if __name__ == "__main__":
    main()
