#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Foldseek-Trans: structural-transfer baseline.

Each test structure is searched against a database of training structures.
The best structural hit transfers its training label; queries without a hit
receive a label sampled from the training class prior. This is the structural
counterpart of the sequence-alignment transfer baseline.

Hits are ranked by:
    tm_mean   descending   (qtmscore + ttmscore) / 2
    alnlen    descending
    target    ascending
over a stable sort, so the choice is deterministic. ``tm_mean`` normalises by
the full chains rather than only the aligned region, which prevents very short
matches from outranking genuine homologues. No coverage floor is applied:
a hit is any training structure returned by Foldseek.

The test split is scored; the validation split is not read. Inputs are CSV
splits, an optional structure-selection CSV, and two flat directories of
structures prepared by ``prepare_structures.py``.
"""
import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from metrics_report import write_metrics_report

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.dirname(CURRENT_DIR)
BENCH_DIR = os.path.dirname(BASELINE_DIR)
RELEASE_ROOT = os.path.dirname(BENCH_DIR)
DATASETS_DIR = os.path.join(RELEASE_ROOT, "datasets")

COVERAGE = 0.0
RANK_BY = "tm_mean"
FOLDSEEK_FORMAT = (
    "query,target,alntmscore,qtmscore,ttmscore,lddt,prob,rmsd,"
    "alnlen,qcov,tcov,evalue,bits,fident")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Foldseek-Trans baseline")
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
    parser.add_argument("--train_structures",
                        default=os.path.join(BASELINE_DIR, "structures",
                                             "train"),
                        help="flat directory of training CIF files")
    parser.add_argument("--test_structures",
                        default=os.path.join(BASELINE_DIR, "structures",
                                             "test"),
                        help="flat directory of test CIF files")
    parser.add_argument("--foldseek", default="foldseek",
                        help="Foldseek executable")
    parser.add_argument("--out_dir",
                        default=os.path.join(CURRENT_DIR,
                                             "foldseek_trans_result"),
                        help="output directory")
    parser.add_argument("--search", action="store_true",
                        help="run Foldseek even if alignment_results.tsv exists")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for no-hit fallback sampling")
    return parser.parse_args()


def load_selection(path):
    if not path or not os.path.exists(path):
        print("structure selection: not found; using all rows")
        return {}
    selection = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            selection[row["ID"].strip()] = (
                (row.get("pass") or "").strip().lower() == "true")
    print("structure selection: %d rows" % len(selection))
    return selection


def apply_quality_filter(frame, selection, name):
    if not selection:
        return frame
    before = len(frame)
    frame = frame[
        frame["ID"].astype(str).str.strip().map(
            lambda pid: selection.get(pid, False))].copy()
    print("  %s: %d -> %d rows after the pass filter"
          % (name, before, len(frame)))
    return frame


def run_search(foldseek, train_structures, test_structures, output_dir):
    result_tsv = os.path.join(output_dir, "alignment_results.tsv")
    tmp_dir = os.path.join(output_dir, "tmp")
    for pattern in (result_tsv, os.path.join(output_dir, "train_db*")):
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    train_db = os.path.join(output_dir, "train_db")
    subprocess.run([foldseek, "createdb", train_structures, train_db,
                    "-v", "1"], check=True)
    subprocess.run(
        [foldseek, "easy-search", test_structures, train_db, result_tsv,
         tmp_dir, "--format-output", FOLDSEEK_FORMAT,
         "-c", str(COVERAGE), "--cov-mode", "0", "-e", "10000",
         "--threads", "32", "-v", "1"],
        check=True)
    return result_tsv


def main():
    args = parse_args()
    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    if shutil.which(args.foldseek) is None and not os.path.exists(args.foldseek):
        raise SystemExit("Foldseek executable not found: %s" % args.foldseek)
    for directory in (args.train_structures, args.test_structures):
        if not os.path.isdir(directory):
            raise SystemExit("missing structure directory %s; run "
                             "prepare_structures.py first" % directory)

    selection = load_selection(args.selection_csv)
    df_train = apply_quality_filter(pd.read_csv(args.train_csv), selection,
                                    "train")
    df_test = apply_quality_filter(pd.read_csv(args.test_csv), selection, "test")

    label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 10: 7, 12: 8}
    label_names = ["1", "2", "3", "4", "5", "6", "8", "10", "12"]
    num_classes = len(label_map)
    reverse_label_map = {index: label for label, index in label_map.items()}

    result_tsv = os.path.join(output_dir, "alignment_results.tsv")
    if args.search or not os.path.exists(result_tsv) \
            or os.path.getsize(result_tsv) == 0:
        result_tsv = run_search(args.foldseek, args.train_structures,
                                args.test_structures, output_dir)
    else:
        print("reusing existing search results: %s" % result_tsv)

    if not os.path.exists(result_tsv) or os.path.getsize(result_tsv) == 0:
        raise SystemExit("no Foldseek hits were produced")

    train_id_to_label = dict(zip(
        df_train["ID"], df_train["Label"].map(label_map).values))
    train_labels = df_train["Label"].map(label_map).values
    class_indices, class_counts = np.unique(train_labels, return_counts=True)
    prior = class_counts / class_counts.sum()

    search_results = pd.read_csv(
        result_tsv, sep="\t", header=None,
        names=["query", "target", "alntmscore", "qtmscore", "ttmscore",
               "lddt", "prob", "rmsd", "alnlen", "qcov", "tcov", "evalue",
               "bits", "fident"])
    print("raw alignments: %d" % len(search_results))

    search_results["tm_mean"] = (
        search_results["qtmscore"] + search_results["ttmscore"]) / 2.0
    search_results["tm_min"] = np.minimum(search_results["qtmscore"],
                                          search_results["ttmscore"])

    best_hits = (search_results
                 .sort_values([RANK_BY, "alnlen", "target"],
                              ascending=[False, False, True], kind="stable")
                 .groupby("query", sort=False)
                 .head(1)
                 .set_index("query"))
    print("queries with a hit: %d" % len(best_hits))

    true_labels = []
    predicted_labels = []
    test_ids = []
    hit_tmscore = {}
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
            if predicted_label is not None:
                hit_tmscore[query_id] = float(best_hits.loc[query_id, RANK_BY])
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

    print("hits: %d   no hit -> prior sample: %d" % (n_hit, n_miss))
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Hard-label predictions are represented as one-hot scores.
    val_predictions = np.zeros((len(true_labels), num_classes))
    val_predictions[np.arange(len(predicted_labels)), predicted_labels] = 1.0

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

    # Record the winning hit and both chain-normalised TM scores for analysis.
    with open(os.path.join(output_dir, "best_hit_tmscore.csv"), "w",
              newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "target", "rank_by", "rank_key", "alntmscore",
                         "qtmscore", "ttmscore", "tm_mean", "tm_min", "alnlen",
                         "qcov", "tcov", "prob"])
        for query_id in test_ids:
            if query_id in hit_tmscore:
                row = best_hits.loc[query_id]
                writer.writerow([
                    query_id, row["target"], RANK_BY, row[RANK_BY],
                    row["alntmscore"], row["qtmscore"], row["ttmscore"],
                    row["tm_mean"], row["tm_min"], row["alnlen"],
                    row["qcov"], row["tcov"], row["prob"]])
            else:
                writer.writerow([query_id] + [""] * 12)

    write_metrics_report(
        y_true=true_labels,
        y_pred=predicted_labels,
        probs=val_predictions,
        classes=list(range(num_classes)),
        label_names=label_names,
        out_dir=output_dir,
    )
    print("Foldseek-Trans finished: %s" % output_dir)


if __name__ == "__main__":
    main()
