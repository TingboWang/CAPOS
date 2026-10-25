#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bootstrap confidence intervals and paired permutation tests for CAPOS.

Methods
-------
CAPOS, CAPOS-seq, and the five custom baselines shipped under ``benchmark/``:
Naïve, Random, Seq-Align-Trans, ESM2-NN-Trans, and Foldseek-Trans.

Inputs
------
CAPOS and CAPOS-seq test-set predictions are passed explicitly:

    --capos-pred      PATH   prediction_results.csv for CAPOS
    --capos-seq-pred  PATH   prediction_results.csv for CAPOS-seq

The baseline predictions are read from the method directories below
``--bench-dir`` (default: this script's directory). Every CSV must contain
``ID``, ``True_Label``, ``Pred_Label`` and ``Prob_Class_<label>`` columns.

Outputs
-------
``bootstrap_out/ci_by_method.csv``
    Point estimates and percentile bootstrap confidence intervals per method
    and metric.
``bootstrap_out/paired_vs_capos.csv``
    Paired CAPOS-minus-baseline differences, intervals, and two-sided Monte
    Carlo permutation p-values.
``bootstrap_out/summary.txt``
    Copy of the console log.

Statistics
----------
Macro-F1, Macro-MCC, AUC-PR and accuracy use the same macro-averaged
definitions as the benchmark reports. AUC-PR is computed per class and then
averaged. Confidence intervals use percentile bootstrapping over sequences.
Paired tests permute whole per-sequence observations between the two methods;
for AUC-PR the complete probability rows are swapped and macro AP is
recomputed, so the uncertainty of the probability vectors is preserved.

Usage
-----
    python benchmark/bootstrap_analysis.py \\
        --capos-pred /path/to/capos_prediction_results.csv \\
        --capos-seq-pred /path/to/capos_seq_prediction_results.csv

    python benchmark/bootstrap_analysis.py ... --n_boot 5000 --n_perm 5000
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))

CLASSES = ["1", "2", "3", "4", "5", "6", "8", "10", "12"]
LABEL = {int(c): i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

METRICS = [("f1", "Macro-F1"), ("mcc", "Macro-MCC"),
           ("ap", "AUC-PR"), ("acc", "Accuracy")]

# Prediction files produced by the custom baselines in this directory.
# Paths are relative to --bench-dir; they are data layout, not machine paths.
BASELINE_FILES = [
    ("Naïve", "Naïve/predict/naive_result/prediction_results.csv"),
    ("Random", "Random/predict/seed42/prediction_results.csv"),
    ("Seq-Align-Trans",
     "Seq-Align-Trans/predict/alignment_transfer_result/"
     "prediction_results.csv"),
    ("ESM2-NN-Trans",
     "ESM2-NN-Trans/predict/embedding_transfer_result/"
     "prediction_results.csv"),
    ("Foldseek-Trans",
     "Foldseek-Trans/predict/foldseek_trans_result/"
     "prediction_results.csv"),
]

REFERENCE = "CAPOS"


def load_standard(path):
    """Load a standard prediction CSV into arrays.

    Rows whose label is outside the nine benchmark classes are mapped to index
    ``NUM_CLASSES`` so they still count as errors rather than being dropped.
    """
    y_true, y_pred, probs, ids = [], [], [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"ID", "True_Label", "Pred_Label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit("%s is missing columns: %s"
                             % (path, ", ".join(sorted(missing))))
        for row in reader:
            ids.append(row["ID"].strip())
            true = int(row["True_Label"])
            pred = int(row["Pred_Label"])
            y_true.append(LABEL.get(true, NUM_CLASSES))
            y_pred.append(LABEL.get(pred, NUM_CLASSES))
            probs.append([float(row.get("Prob_Class_%s" % c, 0.0) or 0.0)
                          for c in CLASSES])
    if not ids:
        raise SystemExit("empty prediction CSV: %s" % path)
    return (np.array(y_true), np.array(y_pred),
            np.array(probs, dtype=np.float64), ids)


def _confusion(y_true, y_pred, k):
    """Confusion counts for in-range predictions; out-of-range counts separately."""
    ok = (y_pred >= 0) & (y_pred < k)
    cm = np.bincount(y_true[ok] * k + y_pred[ok],
                     minlength=k * k).reshape(k, k).astype(np.float64)
    stray = (np.bincount(y_true[~ok], minlength=k).astype(np.float64)
             if (~ok).any() else np.zeros(k))
    return cm, stray


def macro_f1(y_true, y_pred, k):
    cm, stray = _confusion(y_true, y_pred, k)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp + stray
    return float((2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)).mean())


def macro_mcc(y_true, y_pred, k):
    cm, stray = _confusion(y_true, y_pred, k)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp + stray
    total = cm.sum() + stray.sum()
    tn = total - (tp + fp + fn)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-12), 0.0).mean())


def macro_ap(y_true, probs, k):
    """Average precision per class, averaged over classes present in y_true."""
    values = []
    for c in range(k):
        y = (y_true == c).astype(int)
        if y.sum() == 0:
            values.append(0.0)
            continue
        order = np.argsort(-probs[:, c], kind="mergesort")
        ys, scores = y[order], probs[order, c]
        tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
        idx = np.r_[np.where(np.diff(scores))[0], len(scores) - 1]
        precision = tp[idx] / np.maximum(tp[idx] + fp[idx], 1e-12)
        recall = tp[idx] / max(tp[-1], 1e-12)
        values.append(float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision)))
    return float(np.mean(values))


def score(y_true, y_pred, probs, k):
    return {
        "f1": macro_f1(y_true, y_pred, k),
        "mcc": macro_mcc(y_true, y_pred, k),
        "ap": macro_ap(y_true, probs, k),
        "acc": float((y_true == y_pred).mean()),
    }


def _metric_fn(key, y_true, k):
    if key == "f1":
        return lambda y_pred: macro_f1(y_true, y_pred, k)
    if key == "mcc":
        return lambda y_pred: macro_mcc(y_true, y_pred, k)
    return lambda y_pred: float((y_true == y_pred).mean())


def permutation_p(y_true, y_pred_ref, y_pred_base, key, k, n_perm, rng,
                  probs_ref=None, probs_base=None):
    """Two-sided permutation p-value for metric(reference) - metric(baseline).

    Hard-label metrics swap labels between methods. AUC-PR swaps complete
    per-sequence probability rows and recomputes macro AP. The Monte Carlo
    convention is p = (count + 1) / (n_perm + 1).
    """
    if key == "ap":
        if probs_ref is None or probs_base is None:
            raise ValueError("AUC-PR permutation requires probability matrices")
        return _permutation_p_full_vector(
            lambda a, b: macro_ap(y_true, a, k) - macro_ap(y_true, b, k),
            probs_ref, probs_base, n_perm, rng)

    metric = _metric_fn(key, y_true, k)

    def statistic(a, b):
        return metric(a) - metric(b)

    result = stats.permutation_test(
        data=(y_pred_ref, y_pred_base),
        statistic=statistic,
        permutation_type="samples",
        n_resamples=n_perm,
        alternative="two-sided",
        random_state=int(rng.integers(0, 2**31 - 1)),
        vectorized=False,
    )
    return float(result.pvalue)


def _permutation_p_full_vector(delta_fn, probs_a, probs_b, n_perm, rng):
    """Two-sided permutation p when each observation is a probability row."""
    observed = delta_fn(probs_a, probs_b)
    n = probs_a.shape[0]
    count_less = 0
    count_greater = 0
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5
        a = np.where(swap[:, None], probs_b, probs_a)
        b = np.where(swap[:, None], probs_a, probs_b)
        delta = delta_fn(a, b)
        if delta <= observed:
            count_less += 1
        if delta >= observed:
            count_greater += 1
    p_less = (count_less + 1) / (n_perm + 1)
    p_greater = (count_greater + 1) / (n_perm + 1)
    return float(min(1.0, 2.0 * min(p_less, p_greater)))


def bootstrap_ci(n, statistic, n_boot, alpha, rng):
    """Percentile bootstrap CI of statistic(indices) over resampled sequences.

    Paired statistics must receive the same indices for both methods; the
    caller supplies a statistic that indexes all arrays consistently.
    """
    indices = np.arange(n)
    result = stats.bootstrap(
        (indices,),
        statistic,
        n_resamples=n_boot,
        paired=True,
        confidence_level=1.0 - alpha,
        method="percentile",
        random_state=int(rng.integers(0, 2**31 - 1)),
        vectorized=False,
    )
    return float(result.confidence_interval.low), float(
        result.confidence_interval.high)


class Tee:
    """Duplicate stdout to a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capos-pred", required=True,
                        help="CAPOS test-set prediction_results.csv")
    parser.add_argument("--capos-seq-pred", required=True,
                        help="CAPOS-seq test-set prediction_results.csv")
    parser.add_argument("--bench-dir", default=HERE,
                        help="directory containing the baseline result folders "
                             "(default: this script's directory)")
    parser.add_argument("--out_dir", default=os.path.join(HERE, "bootstrap_out"),
                        help="output directory")
    parser.add_argument("--n_boot", type=int, default=2000,
                        help="bootstrap resamples")
    parser.add_argument("--n_perm", type=int, default=2000,
                        help="permutation resamples")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="significance level (95%% CI by default)")
    return parser.parse_args()


def main():
    args = parse_args()
    bench = os.path.abspath(args.bench_dir)
    methods = [
        (REFERENCE, os.path.abspath(args.capos_pred)),
        ("CAPOS-seq", os.path.abspath(args.capos_seq_pred)),
    ]
    methods.extend((name, os.path.join(bench, rel))
                   for name, rel in BASELINE_FILES)

    for name, path in methods:
        if not os.path.exists(path):
            raise SystemExit("missing %s predictions: %s" % (name, path))

    os.makedirs(args.out_dir, exist_ok=True)
    tee_fh = open(os.path.join(args.out_dir, "summary.txt"), "w",
                  encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, tee_fh)
    rng = np.random.default_rng(args.seed)

    try:
        raw = {name: load_standard(path) for name, path in methods}

        ref_ids = raw[REFERENCE][3]
        if len(set(ref_ids)) != len(ref_ids):
            raise SystemExit("CAPOS predictions contain duplicate IDs")
        ref_set = set(ref_ids)

        data = {}
        for name, (y_true, y_pred, probs, ids) in raw.items():
            if len(set(ids)) != len(ids):
                raise SystemExit("%s predictions contain duplicate IDs" % name)
            if set(ids) != ref_set:
                raise SystemExit(
                    "%s covers a different ID set than CAPOS "
                    "(%d missing, %d extra)"
                    % (name, len(ref_set - set(ids)), len(set(ids) - ref_set)))
            position = {pid: i for i, pid in enumerate(ids)}
            order = np.array([position[pid] for pid in ref_ids])
            data[name] = (y_true[order], y_pred[order], probs[order])

        n = len(ref_ids)
        k = NUM_CLASSES
        print("seed %d   bootstrap %d   permutations %d   alpha %.2f"
              % (args.seed, args.n_boot, args.n_perm, args.alpha))
        print("scope full   n=%d   classes=%d" % (n, k))
        print()

        point = {name: score(*values, k) for name, values in data.items()}

        print("=" * 80)
        print("  %-18s %10s %11s %10s %10s"
              % ("method", *(label for _, label in METRICS)))
        print("  " + "-" * 72)
        for name in data:
            print("  %-18s %10.4f %11.4f %10.4f %10.4f"
                  % (name, *(point[name][key] for key, _ in METRICS)))
        print()

        def make_stat(name, key):
            y_true, y_pred, probs = data[name]

            def statistic(indices):
                if key == "f1":
                    return macro_f1(y_true[indices], y_pred[indices], k)
                if key == "mcc":
                    return macro_mcc(y_true[indices], y_pred[indices], k)
                if key == "ap":
                    return macro_ap(y_true[indices], probs[indices], k)
                return float((y_true[indices] == y_pred[indices]).mean())

            return statistic

        def make_delta_stat(key, ref_name, base_name):
            y_true = data[ref_name][0]
            pred_a, pred_b = data[ref_name][1], data[base_name][1]

            def statistic(indices):
                if key == "f1":
                    return (macro_f1(y_true[indices], pred_a[indices], k)
                            - macro_f1(y_true[indices], pred_b[indices], k))
                if key == "mcc":
                    return (macro_mcc(y_true[indices], pred_a[indices], k)
                            - macro_mcc(y_true[indices], pred_b[indices], k))
                return float((y_true[indices] == pred_a[indices]).mean()
                             - (y_true[indices] == pred_b[indices]).mean())

            return statistic

        def make_ap_delta_stat(ref_name, base_name):
            y_true = data[ref_name][0]
            probs_a, probs_b = data[ref_name][2], data[base_name][2]

            def statistic(indices):
                return (macro_ap(y_true[indices], probs_a[indices], k)
                        - macro_ap(y_true[indices], probs_b[indices], k))

            return statistic

        # Per-method confidence intervals.
        jobs = [(name, key) for name in data for key, _ in METRICS]
        seeds = rng.integers(0, 2**31 - 1, size=len(jobs))

        def ci_job(name, key, seed):
            local_rng = np.random.default_rng(int(seed))
            return (name, key), bootstrap_ci(
                n, make_stat(name, key), args.n_boot, args.alpha, local_rng)

        ci_cache = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(ci_job, name, key, seed)
                       for (name, key), seed in zip(jobs, seeds)]
            for future in futures:
                key, interval = future.result()
                ci_cache[key] = interval

        ci_rows = []
        for name in data:
            for key, label in METRICS:
                low, high = ci_cache[(name, key)]
                ci_rows.append([
                    name, label, "%.4f" % point[name][key],
                    "%.4f" % low, "%.4f" % high,
                ])

        # Paired comparisons against CAPOS.
        pair_rows = []
        level = int(round(100 * (1 - args.alpha)))
        for name in data:
            if name == REFERENCE:
                continue
            print("  %s minus %s   (positive = %s better)"
                  % (REFERENCE, name, REFERENCE))
            for key, label in METRICS:
                if key == "ap":
                    low, high = bootstrap_ci(
                        n, make_ap_delta_stat(REFERENCE, name),
                        args.n_boot, args.alpha, rng)
                else:
                    low, high = bootstrap_ci(
                        n, make_delta_stat(key, REFERENCE, name),
                        args.n_boot, args.alpha, rng)
                verdict = "significant" if (low > 0 or high < 0) else "not resolved"
                p_value = permutation_p(
                    data[REFERENCE][0], data[REFERENCE][1], data[name][1],
                    key, k, args.n_perm, rng,
                    probs_ref=data[REFERENCE][2], probs_base=data[name][2])
                diff = point[REFERENCE][key] - point[name][key]
                pair_rows.append([
                    REFERENCE, name, label, "%+.4f" % diff,
                    "%.4f" % low, "%.4f" % high,
                    verdict, "%.4g" % p_value,
                ])
                print("    %-11s %+9.4f  [%+.4f, %+.4f]  %-13s p=%s"
                      % (label, diff, low, high, verdict, "%.4g" % p_value))
            print()

        print("  'significant' = the %d%% paired-difference interval excludes 0."
              % level)
        print("  'not resolved' = this test set cannot distinguish the methods; "
              "it does not mean they are equal.")
        print()

        def write_csv(path, header, rows):
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

        ci_path = os.path.join(args.out_dir, "ci_by_method.csv")
        pair_path = os.path.join(args.out_dir, "paired_vs_capos.csv")
        write_csv(ci_path, ["method", "metric", "point", "ci_lo", "ci_hi"],
                  ci_rows)
        write_csv(pair_path,
                  ["reference", "baseline", "metric", "point_diff",
                   "ci_lo", "ci_hi", "verdict", "p_permutation"],
                  pair_rows)
        print("  wrote %s   (%d rows)" % (ci_path, len(ci_rows)))
        print("  wrote %s   (%d rows)" % (pair_path, len(pair_rows)))
        print("  wrote %s" % os.path.join(args.out_dir, "summary.txt"))
    finally:
        sys.stdout = original_stdout
        tee_fh.close()


if __name__ == "__main__":
    main()
