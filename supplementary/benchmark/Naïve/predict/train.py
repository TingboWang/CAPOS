# -*- coding: utf-8 -*-
"""Naïve (majority-class) baseline.

Every test protein is assigned the most frequent class in the training split;
the probability vector is the training class prior. This baseline measures how
much a classifier gains over always predicting the majority state.

Inputs are CSV tables with ``ID``, ``Sequence`` and ``Label`` columns. If a
structure-selection CSV with an ``ID,pass`` column is supplied, only rows whose
``pass`` value is true are evaluated. Outputs are written to ``--out_dir``:
``prediction_results.csv``, ``classification_report.csv``,
``confusion_matrix.pdf`` and ``PR_curves_per_class.pdf``.
"""
import argparse
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve)

import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from metrics_report import write_metrics_report

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
RELEASE_ROOT = os.path.dirname(BENCH_DIR)
DATASETS_DIR = os.path.join(RELEASE_ROOT, "datasets")


def parse_args():
    parser = argparse.ArgumentParser(description="Naïve majority-class baseline")
    parser.add_argument("--train_csv",
                        default=os.path.join(DATASETS_DIR,
                                             "dataset_5_train.csv"),
                        help="training split CSV")
    parser.add_argument("--test_csv",
                        default=os.path.join(DATASETS_DIR, "test.csv"),
                        help="test split CSV")
    parser.add_argument("--selection_csv",
                        default=os.path.join(DATASETS_DIR,
                                             "structure_selection.csv"),
                        help="optional ID,pass structure filter; skipped if the "
                             "file does not exist")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed")
    parser.add_argument("--out_dir", default=os.path.join(CURRENT_DIR,
                                                          "naive_result"),
                        help="output directory")
    return parser.parse_args()


def load_selection(path):
    """Return an ID -> pass map, or None when no selection file is available."""
    if not path or not os.path.exists(path):
        print("structure selection: not found; using all rows")
        return None
    selection = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            selection[row["ID"].strip()] = (
                (row.get("pass") or "").strip().lower() == "true")
    print("structure selection: %d rows" % len(selection))
    return selection


def apply_quality_filter(frame, selection, name):
    if selection is None:
        return frame
    before = len(frame)
    frame = frame[
        frame["ID"].astype(str).str.strip().map(
            lambda pid: selection.get(pid, False))].copy()
    print("  %s: %d -> %d rows after the pass filter"
          % (name, before, len(frame)))
    return frame


def process_labels(frame, mapping):
    frame = frame[frame["Label"].isin(mapping)].copy()
    frame["mapped_label"] = frame["Label"].map(mapping)
    return frame


def main():
    args = parse_args()
    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    selection = load_selection(args.selection_csv)
    df_train = apply_quality_filter(pd.read_csv(args.train_csv), selection,
                                    "train")
    df_test = apply_quality_filter(pd.read_csv(args.test_csv), selection, "test")

    label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 10: 7, 12: 8}
    label_names = ["1", "2", "3", "4", "5", "6", "8", "10", "12"]
    num_classes = len(label_map)
    reverse_label_map = {index: label for label, index in label_map.items()}

    df_train = process_labels(df_train, label_map)
    df_test = process_labels(df_test, label_map)

    # Training class prior; classes absent from the training split get zero.
    train_counts = df_train["mapped_label"].value_counts(
        normalize=True).sort_index()
    full_freqs = np.zeros(num_classes)
    for index, frequency in train_counts.items():
        full_freqs[index] = frequency

    majority_class = int(np.argmax(full_freqs))
    print("majority class: %s"
          % reverse_label_map[majority_class])

    test_ids = df_test["ID"].values
    true_labels = df_test["mapped_label"].values
    predicted_labels = np.full(len(true_labels), majority_class)
    val_predictions = np.tile(full_freqs, (len(true_labels), 1))

    # Confusion matrix.
    cm = confusion_matrix(true_labels, predicted_labels,
                          labels=list(range(num_classes)), normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted Label (Naïve)")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Naïve")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.pdf"),
                bbox_inches="tight")
    plt.close()

    # Per-sequence predictions.
    df_results = pd.DataFrame(
        val_predictions,
        columns=["Prob_Class_%s" % label for label in label_names])
    df_results.insert(0, "ID", test_ids)
    df_results["True_Label"] = [reverse_label_map[label]
                                for label in true_labels]
    df_results["Pred_Label"] = [reverse_label_map[label]
                                for label in predicted_labels]
    df_results.to_csv(os.path.join(output_dir, "prediction_results.csv"),
                      index=False)

    # PR curves; numeric metrics are written by write_metrics_report below.
    y_true_onehot = np.eye(num_classes)[true_labels]
    plt.figure(figsize=(10, 8))
    for index in range(num_classes):
        if y_true_onehot[:, index].sum() > 0:
            precision, recall, _ = precision_recall_curve(
                y_true_onehot[:, index], val_predictions[:, index])
            auc = average_precision_score(y_true_onehot[:, index],
                                          val_predictions[:, index])
            plt.plot(recall, precision,
                     label="Class %s (AUC=%.2f)"
                           % (label_names[index], auc))
    plt.title("PR Curves - Naïve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, "PR_curves_per_class.pdf"))
    plt.close()

    write_metrics_report(
        y_true=true_labels,
        y_pred=predicted_labels,
        probs=val_predictions,
        classes=list(range(num_classes)),
        label_names=label_names,
        out_dir=output_dir,
    )
    print("Naïve baseline finished: %s" % output_dir)


if __name__ == "__main__":
    main()
