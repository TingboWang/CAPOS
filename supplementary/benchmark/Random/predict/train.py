# -*- coding: utf-8 -*-
"""Random baseline: sample predictions from the training class prior.

For every test protein a label is drawn from the training class distribution.
The probability vector written to the report is that same prior. The analytic
accuracy and macro-F1 of this sampler are computed alongside the sampled run
to verify that the sampling matches the intended distribution.

Inputs are the training and test CSV splits. If a structure-selection CSV with
an ``ID,pass`` column is supplied, only passing rows are evaluated.
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
    parser = argparse.ArgumentParser(description="Random prior-sampling baseline")
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
                        help="RNG seed for label sampling")
    parser.add_argument("--out_dir", default=os.path.join(CURRENT_DIR,
                                                          "random_result"),
                        help="output directory")
    return parser.parse_args()


def load_selection(path):
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


def expectation(prior, observed, class_indices):
    """Analytic accuracy and macro-F1 when predictions follow ``prior``."""
    accuracy = sum(prior[i] * observed[i] for i in class_indices)
    f1 = {
        i: (2 * prior[i] * observed[i] / (prior[i] + observed[i])
            if prior[i] + observed[i] > 0 else 0.0)
        for i in class_indices
    }
    return accuracy, sum(f1.values()) / len(class_indices)


def main():
    args = parse_args()
    output_root = args.out_dir
    os.makedirs(output_root, exist_ok=True)

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

    train_counts = df_train["mapped_label"].value_counts(
        normalize=True).sort_index()
    full_freqs = np.zeros(num_classes)
    for index, frequency in train_counts.items():
        full_freqs[index] = frequency

    class_indices = list(range(num_classes))
    names = [label_names[index] for index in class_indices]
    prior = full_freqs[class_indices] / full_freqs[class_indices].sum()

    mask = df_test["mapped_label"].isin(class_indices).values
    test_ids = df_test["ID"].values[mask]
    true_labels = df_test["mapped_label"].values[mask]
    num_test = len(true_labels)

    # Draw one label per test protein from the training prior.
    predicted_labels = np.random.choice(class_indices, size=num_test, p=prior)

    # The reported probability vector is the prior, identical for every row.
    val_predictions = np.tile(prior, (num_test, 1))

    cm = confusion_matrix(true_labels, predicted_labels, labels=class_indices,
                          normalize="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Oranges",
                xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted Label (Random)")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Random")
    plt.savefig(os.path.join(output_root, "confusion_matrix.pdf"),
                bbox_inches="tight")
    plt.close()

    df_results = pd.DataFrame(
        val_predictions,
        columns=["Prob_Class_%s" % name for name in names])
    df_results.insert(0, "ID", test_ids)
    df_results["True_Label"] = [reverse_label_map[label]
                                for label in true_labels]
    df_results["Pred_Label"] = [reverse_label_map[label]
                                for label in predicted_labels]
    df_results.to_csv(os.path.join(output_root, "prediction_results.csv"),
                      index=False)

    # PR curves; numeric metrics are written by write_metrics_report below.
    y_onehot = np.zeros((num_test, num_classes))
    for index in class_indices:
        y_onehot[:, index] = (true_labels == index).astype(float)
    plt.figure(figsize=(8, 6))
    for index in class_indices:
        if y_onehot[:, index].sum() > 0:
            precision, recall, _ = precision_recall_curve(
                y_onehot[:, index], val_predictions[:, index])
            auc = average_precision_score(y_onehot[:, index],
                                          val_predictions[:, index])
            plt.plot(recall, precision,
                     label="Class %s (AUC=%.2f)"
                           % (label_names[index], auc))
    plt.title("PR Curves - Random")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_root, "PR_curves_per_class.pdf"))
    plt.close()

    df_report = write_metrics_report(
        y_true=true_labels,
        y_pred=predicted_labels,
        probs=val_predictions,
        classes=class_indices,
        label_names=names,
        out_dir=output_root,
    )

    observed = np.array([np.mean(true_labels == index)
                         for index in class_indices])
    expected_accuracy, expected_f1 = expectation(
        {index: prior[position] for position, index in enumerate(class_indices)},
        {index: observed[position] for position, index in enumerate(class_indices)},
        class_indices)
    sampled_accuracy = float(df_report.loc["Accuracy", "Precision"])
    sampled_f1 = float(df_report.loc["Macro avg", "F1-score"])
    print("sampled accuracy %.4f  macro-F1 %.4f" % (sampled_accuracy, sampled_f1))
    print("analytic accuracy %.4f  macro-F1 %.4f"
          % (expected_accuracy, expected_f1))
    print("Random baseline finished: %s" % output_root)


if __name__ == "__main__":
    main()
