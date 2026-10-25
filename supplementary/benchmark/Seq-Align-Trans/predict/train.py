# -*- coding: utf-8 -*-
"""Seq-Align-Trans: MMseqs2 sequence-alignment label transfer.

Each test sequence is searched against the training sequences with MMseqs2.
The best hit transfers its training label; queries without a hit receive a
label sampled from the training class prior. Hits are ranked by sequence
identity, then alignment length, then target ID.

The search enforces at least 80% coverage of both query and target
(``-c 0.8 --cov-mode 0``) and no minimum identity.

Inputs are CSV splits and an optional structure-selection CSV (``ID,pass``).
Outputs include the generated FASTA files, MMseqs2 search tables,
``prediction_results.csv``, ``classification_report.csv`` and figures.
"""
import argparse
import csv
import glob
import os
import random
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
        description="Seq-Align-Trans baseline")
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
    parser.add_argument("--out_dir",
                        default=os.path.join(CURRENT_DIR,
                                             "alignment_transfer_result"),
                        help="output directory")
    parser.add_argument("--mmseqs", default="mmseqs",
                        help="MMseqs2 executable")
    parser.add_argument("--threads", type=int, default=32,
                        help="MMseqs2 search threads")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for no-hit fallback sampling")
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


def write_fasta(frame, path):
    with open(path, "w") as handle:
        for _, row in frame.iterrows():
            handle.write(">%s\n%s\n" % (row["ID"], row["Sequence"]))


def clear_search_artifacts(output_dir):
    for pattern in (os.path.join(output_dir, "train_db*"),
                    os.path.join(output_dir, "test_db*"),
                    os.path.join(output_dir, "search_res*"),
                    os.path.join(output_dir, "alignment_results.tsv"),
                    os.path.join(output_dir, "tmp")):
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)


def run_mmseqs(mmseqs, train_fasta, test_fasta, output_dir, threads):
    train_db = os.path.join(output_dir, "train_db")
    test_db = os.path.join(output_dir, "test_db")
    result_db = os.path.join(output_dir, "search_res")
    result_tsv = os.path.join(output_dir, "alignment_results.tsv")
    tmp_dir = os.path.join(output_dir, "tmp")

    # MMseqs2 cannot overwrite an existing result database; clear old files.
    clear_search_artifacts(output_dir)

    subprocess.run([mmseqs, "createdb", train_fasta, train_db,
                    "--dbtype", "0"], check=True)
    subprocess.run([mmseqs, "createdb", test_fasta, test_db,
                    "--dbtype", "0"], check=True)
    subprocess.run([mmseqs, "search", test_db, train_db, result_db, tmp_dir,
                    "--min-seq-id", "0.0", "-c", "0.8", "--cov-mode", "0",
                    "-s", "7.5", "--threads", str(threads)], check=True)
    subprocess.run([mmseqs, "convertalis", test_db, train_db, result_db,
                    result_tsv, "--format-output",
                    "query,target,pident,alnlen,qcov,evalue"], check=True)
    return result_tsv


def main():
    args = parse_args()
    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    if shutil.which(args.mmseqs) is None and not os.path.exists(args.mmseqs):
        raise SystemExit("MMseqs2 executable not found: %s" % args.mmseqs)

    selection = load_selection(args.selection_csv)
    df_train = apply_quality_filter(pd.read_csv(args.train_csv), selection,
                                    "train")
    df_test = apply_quality_filter(pd.read_csv(args.test_csv), selection, "test")

    train_fasta = os.path.join(output_dir, "train_set.fasta")
    test_fasta = os.path.join(output_dir, "test_set.fasta")
    write_fasta(df_train, train_fasta)
    write_fasta(df_test, test_fasta)

    label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 10: 7, 12: 8}
    label_names = ["1", "2", "3", "4", "5", "6", "8", "10", "12"]
    num_classes = len(label_map)
    reverse_label_map = {index: label for label, index in label_map.items()}

    result_tsv = run_mmseqs(args.mmseqs, train_fasta, test_fasta, output_dir,
                            args.threads)

    train_id_to_label = dict(zip(df_train["ID"],
                                 df_train["Label"].map(label_map).values))
    train_labels = df_train["Label"].map(label_map).values
    class_indices, class_counts = np.unique(train_labels, return_counts=True)
    prior = class_counts / class_counts.sum()

    if os.path.exists(result_tsv) and os.path.getsize(result_tsv) > 0:
        search_results = pd.read_csv(
            result_tsv, sep="\t",
            names=["query", "target", "pident", "alnlen", "qcov", "evalue"])
        best_hits = (search_results
                     .sort_values(["pident", "alnlen", "target"],
                                  ascending=[False, False, True],
                                  kind="stable")
                     .groupby("query", sort=False)
                     .head(1)
                     .set_index("query"))
    else:
        print("no alignment hits; every query uses the training prior")
        best_hits = pd.DataFrame()

    true_labels = []
    predicted_labels = []
    test_ids = []
    n_hit = n_miss = 0
    for _, row in df_test.iterrows():
        query_id = row["ID"]
        raw_label = row["Label"]
        if raw_label not in label_map:
            continue

        true_label = label_map[raw_label]
        if query_id in best_hits.index:
            predicted_label = train_id_to_label.get(
                best_hits.loc[query_id, "target"])
        else:
            predicted_label = None

        if predicted_label is None:
            predicted_label = int(np.random.choice(class_indices, p=prior))
            n_miss += 1
        else:
            n_hit += 1

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
        test_ids.append(query_id)

    print("  hits: %d   no hit -> prior sample: %d" % (n_hit, n_miss))
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Hard-label predictions are represented as one-hot scores.
    val_predictions = np.zeros((len(true_labels), num_classes))
    val_predictions[np.arange(len(predicted_labels)), predicted_labels] = 1.0

    cm = confusion_matrix(true_labels, predicted_labels,
                          labels=list(range(num_classes)), normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Reds",
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted Label (Seq-Align-Trans)")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Seq-Align-Trans")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.pdf"),
                bbox_inches="tight")
    plt.close()

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

    write_metrics_report(
        y_true=true_labels,
        y_pred=predicted_labels,
        probs=val_predictions,
        classes=list(range(num_classes)),
        label_names=label_names,
        out_dir=output_dir,
    )
    print("Seq-Align-Trans finished: %s" % output_dir)


if __name__ == "__main__":
    main()
