# -*- coding: utf-8 -*-
"""ESM2-NN-Trans: ESM-2 embedding nearest-neighbour transfer.

Each test protein receives the label of its cosine-nearest training protein in
a precomputed ESM-2 embedding space. Embeddings are read from an ``.npz`` file
with ``ids`` and ``features`` arrays.

Inputs are CSV splits and an optional structure-selection CSV. The embedding
file is supplied with ``--features``. Outputs are ``prediction_results.csv``,
``classification_report.csv``, a confusion matrix and PR curves.
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from metrics_report import write_metrics_report

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
RELEASE_ROOT = os.path.dirname(BENCH_DIR)
DATASETS_DIR = os.path.join(RELEASE_ROOT, "datasets")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ESM2-NN-Trans baseline")
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
    parser.add_argument("--features",
                        default=os.path.join(BENCH_DIR, "feature_prep",
                                             "esm2_features_all.npz"),
                        help="NPZ file with embedding ids and features")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--out_dir",
                        default=os.path.join(CURRENT_DIR,
                                             "embedding_transfer_result"),
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


def load_split(path, selection, name):
    frame = pd.read_csv(path)
    before = len(frame)
    frame["ID"] = frame["ID"].astype(str).str.strip()
    if selection is not None:
        frame = frame[frame["ID"].map(
            lambda pid: selection.get(pid, False))].copy()
        print("  %s: %d -> %d rows after the pass filter"
              % (name, before, len(frame)))
    return frame


def main():
    args = parse_args()
    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    selection = load_selection(args.selection_csv)
    df_train = load_split(args.train_csv, selection, "train")
    df_test = load_split(args.test_csv, selection, "test")

    if not os.path.exists(args.features):
        raise SystemExit("embedding file not found: %s (pass --features)"
                         % args.features)
    # The embedding archive must contain plain arrays (Unicode IDs and numeric
    # features); allow_pickle is deliberately disabled for safety.
    data = np.load(args.features, allow_pickle=False)
    feature_ids = [str(value) for value in data["ids"]]
    features = data["features"]
    positions = {}
    for index, identifier in enumerate(feature_ids):
        positions.setdefault(identifier, index)

    def features_for(frame, name):
        identifiers = frame["ID"].tolist()
        missing = [pid for pid in identifiers if pid not in positions]
        if missing:
            raise SystemExit("%d %s IDs have no embedding (first: %s)"
                             % (len(missing), name, missing[0]))
        return features[[positions[pid] for pid in identifiers]]

    x_train = features_for(df_train, "train")
    x_test = features_for(df_test, "test")
    print("train embeddings %s   test embeddings %s"
          % (x_train.shape, x_test.shape))

    label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 10: 7, 12: 8}
    label_names = ["1", "2", "3", "4", "5", "6", "8", "10", "12"]
    num_classes = len(label_map)
    reverse_label_map = {index: label for label, index in label_map.items()}

    y_train = df_train["Label"].map(label_map).values
    y_test = df_test["Label"].map(label_map).values
    test_ids = df_test["ID"].values

    similarity = cosine_similarity(x_test, x_train)
    nearest = np.argmax(similarity, axis=1)
    predicted_labels = y_train[nearest]

    val_predictions = np.zeros((len(x_test), num_classes))
    val_predictions[np.arange(len(predicted_labels)), predicted_labels] = 1.0

    cm = confusion_matrix(y_test, predicted_labels,
                          labels=list(range(num_classes)), normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted Label (ESM2-NN-Trans)")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - ESM2-NN-Trans")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.pdf"),
                bbox_inches="tight")
    plt.close()

    df_results = pd.DataFrame(
        val_predictions,
        columns=["Prob_Class_%s" % label for label in label_names])
    df_results.insert(0, "ID", test_ids)
    df_results["True_Label"] = [reverse_label_map[label] for label in y_test]
    df_results["Pred_Label"] = [reverse_label_map[label]
                                for label in predicted_labels]
    df_results.to_csv(os.path.join(output_dir, "prediction_results.csv"),
                      index=False)

    write_metrics_report(
        y_true=y_test,
        y_pred=predicted_labels,
        probs=val_predictions,
        classes=list(range(num_classes)),
        label_names=label_names,
        out_dir=output_dir,
    )
    print("ESM2-NN-Trans finished: %s" % output_dir)


if __name__ == "__main__":
    main()
