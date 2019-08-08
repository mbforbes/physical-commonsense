"""Gotta make graphs."""

import code
import os
from typing import List, Tuple, Dict, Any, Optional, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pc import metrics
from pc import util


def get_cms(path: str) -> Dict[str, np.ndarray]:
    item2cm: Dict[str, np.ndarray] = {}
    with open(path, "r") as f:
        for line in f.readlines():
            pieces = line.split(" ")
            item = pieces[0]
            cm = util.str2np(" ".join(pieces[1:]))
            item2cm[item] = cm
    return item2cm


def _setup(figsize=((6.4, 4.8))) -> None:
    plt.clf()

    sns.set()
    # sns.set_context("paper")
    sns.set_palette("husl")
    # sns.set_palette("bright")
    # sns.set_style("whitegrid")
    # sns.set_style("darkgrid")

    plt.figure(figsize=figsize)


def build_prop_cat_graph(cat2props, prop2cm, prop2human, metric) -> None:
    """Renders props only, breakdown by prop category."""
    _setup()

    # build graph data
    rows: List[Dict[str, Any]] = []
    for cat, props in cat2props.items():
        for prop in props:
            cm = prop2cm[prop]
            precision, recall, f1 = metrics.prf1(cm)
            acc = (cm[0][0] + cm[1][1]) / cm.sum()
            all_metrics = {
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
            rows.append(
                {
                    "Category": cat,
                    "Property": prop2human[prop],
                    metric: all_metrics[metric],
                }
            )
    df = pd.DataFrame(rows)

    # render graph
    sns.violinplot(x="Category", y=metric, data=df, inner=None, color=".88")
    sns.stripplot(x="Category", y=metric, data=df, jitter=True)

    # format and display
    plt.tight_layout()
    plt.savefig("data/results/graphs/bert-AP-props-{}.pdf".format(metric))


def build_prop_aff_graph(prop2cm, aff2cm) -> None:
    """Renders affs and props, overall view."""
    _setup()

    rows: List[Dict[str, Any]] = []
    # print("Properties:")
    for prop, cm in prop2cm.items():
        precision, recall, f1 = metrics.prf1(cm)
        rows.append({"Data": "Property", "Property": prop, "F1": f1})
        # print("- {}: {:.3f}".format(prop, f1))
    # print("Affordances:")
    for aff, cm in aff2cm.items():
        precision, recall, f1 = metrics.prf1(cm)
        rows.append({"Data": "Affordance", "Affordance": aff, "F1": f1})
        # print("- {}: {:.3f}".format(aff, f1))
    df = pd.DataFrame(rows)

    # render graph
    sns.violinplot(x="Data", y="F1", data=df, inner=None, color=".88")
    sns.stripplot(x="Data", y="F1", data=df, jitter=True)

    # format and display
    plt.tight_layout()
    plt.savefig("data/results/graphs/bert-AP-props-affs.pdf")


def build_item_freqs_vs_f1(
    item_freqs,
    item2cm,
    freq_lab,
    out_path,
    title: Optional[str] = None,
    f1_lab: str = "F1",
) -> None:
    """Renders item freq in our data vs item f1 score."""
    _setup((3.2 * 1.5, 1.9 * 1.5))

    rows: List[Dict[str, Any]] = []
    for item, cm in item2cm.items():
        freq = item_freqs[item]
        precision, recall, f1 = metrics.prf1(cm)
        rows.append({freq_lab: freq, f1_lab: f1})
    df = pd.DataFrame(rows)

    sns.regplot(x=freq_lab, y=f1_lab, data=df)  # , marker="+")
    # plt.ylim((-0.0002, 0.0006))
    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path)


def build_prop_freqs_vs_acc(prop_freqs, prop2cm, freq_lab, out_path) -> None:
    """Renders prop freq in our data vs prop f1 score."""
    _setup()

    rows: List[Dict[str, Any]] = []
    y_lab = "Accuracy"
    for prop, freq in prop_freqs.items():
        cm = prop2cm[prop]
        acc = (cm[0][0] + cm[1][1]) / cm.sum()
        rows.append({freq_lab: freq, y_lab: acc})
    df = pd.DataFrame(rows)

    sns.scatterplot(x=freq_lab, y=y_lab, data=df)

    plt.tight_layout()
    plt.savefig(out_path)


def build_freq_vs_freq(
    freq1: Dict[str, float], freq2: Dict[str, float], freq1_lab: str, freq2_lab: str
) -> None:
    """Plot freq vs freq from two data sources"""
    _setup()

    rows: List[Dict[str, Any]] = []
    for prop, item_freq1 in freq1.items():
        item_freq2 = freq2[prop]
        rows.append({freq1_lab: item_freq1, freq2_lab: item_freq2})
    df = pd.DataFrame(rows)

    sns.scatterplot(x=freq1_lab, y=freq2_lab, data=df)

    plt.tight_layout()
    plt.savefig("data/results/graphs/bert-AP-props-freq-vs-freq.pdf")


def main() -> None:
    # setup
    os.makedirs("data/results/graphs/", exist_ok=True)

    # read in prop mapping
    prop_def_df = pd.read_csv("data/pc/properties.tsv", delimiter="\t")

    # read in property classification for mapping
    cat2props: Dict[str, List[str]] = {}
    prop2human: Dict[str, str] = {}
    prop_oneword2uid: Dict[str, str] = {}
    for i, row in prop_def_df.iterrows():
        cat = row["categorization"].capitalize()
        prop = row["uid"]
        if cat not in cat2props:
            cat2props[cat] = []
        cat2props[cat].append(prop)
        prop2human[prop] = row["sentence"]
        prop_oneword2uid[row["word-embedding"]] = row["uid"]

    # get frequency counts for properties in our data.
    prop_pc_freqs: Dict[str, float] = {}
    prop_ext_df = pd.read_csv("data/pc/situated-properties.csv").drop(
        columns=["cocoImgID", "cocoAnnID", "objectUID"]
    )
    tot = len(prop_ext_df)
    for prop in prop_ext_df.columns:
        prop_pc_freqs[prop] = prop_ext_df[prop].sum() / tot

    # get frequency counts from natural language
    prop_nl_freqs: Dict[str, float] = {}
    with open("data/nl/prop-freqs-nl.csv", "r") as f:
        lines = [l.strip() for l in f.readlines()[1:]]
        for line in lines:
            prop_oneword, freq_str = line.split(",")
            prop_uid = prop_oneword2uid[prop_oneword]
            prop_nl_freqs[prop_uid] = float(freq_str)

    # get frequency counts from natural language
    aff_nl_freqs: Dict[str, float] = {}
    with open("data/nl/aff-freqs-nl.csv", "r") as f:
        lines = [l.strip() for l in f.readlines()[1:]]
        for line in lines:
            aff, freq_str = line.split(",")
            aff_nl_freqs[aff] = float(freq_str)

    # read in our output dta
    prop2cm = get_cms("data/results/Bert-AP-P.txt")
    aff2cm = get_cms("data/results/Bert-AP-A.txt")

    # build graphs
    build_item_freqs_vs_f1(
        prop_pc_freqs,
        prop2cm,
        "Frequency (our data)",
        "data/results/graphs/bert-AP-props-pcdata-vs-f1.pdf",
    )
    build_prop_freqs_vs_acc(
        prop_pc_freqs,
        prop2cm,
        "Frequency (our data)",
        "data/results/graphs/bert-AP-props-pcdata-vs-acc.pdf",
    )
    build_item_freqs_vs_f1(
        prop_nl_freqs,
        prop2cm,
        "Frequency (natural language)",
        "data/results/graphs/bert-AP-props-nl-vs-f1.pdf",
        "Property F1 Scores vs Frequency in Text",
        "F1 Scores\nby Property",
    )
    build_item_freqs_vs_f1(
        aff_nl_freqs,
        aff2cm,
        "Frequency (natural language)",
        "data/results/graphs/bert-AP-aff-nl-vs-f1.pdf",
        "Affordance F1 Scores vs Frequency in Text",
        "F1 Scores\nby Affordance",
    )
    build_prop_cat_graph(cat2props, prop2cm, prop2human, "Accuracy")
    build_prop_cat_graph(cat2props, prop2cm, prop2human, "F1")
    build_prop_aff_graph(prop2cm, aff2cm)
    build_freq_vs_freq(
        prop_pc_freqs,
        prop_nl_freqs,
        "Property Frequency (Our Data)",
        "Property Frequency (Natural Language)",
    )


if __name__ == "__main__":
    main()
