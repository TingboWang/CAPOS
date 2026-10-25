#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write one metrics CSV per evaluation run.

The report contains Precision, Recall, F1-score, AUC-PR, MCC and Support for
every class, plus Macro-average, Accuracy and a global multiclass MCC row.

Schema
------
::

    ,Precision,Recall,F1-score,AUC-PR,MCC,Support
    1,...,...,...,...,...,...
    ...
    Macro avg,...
    Accuracy,<accuracy in the Precision column>,,,,
    Global-Multiclass,,,,,<global MCC>,

``Macro-MCC`` is the mean of the per-class one-vs-rest MCCs. The global
multiclass MCC is a different quantity and is written to the
``Global-Multiclass`` row for completeness. Accuracy is stored in the
``Precision`` column of the ``Accuracy`` row because that is the established
report layout used by the benchmark readers.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             precision_recall_fscore_support)

MACRO_ROW = "Macro avg"
ACC_ROW = "Accuracy"
GLOBAL_MCC_ROW = "Global-Multiclass"

REPORT_COLUMNS = ["Precision", "Recall", "F1-score", "AUC-PR", "MCC", "Support"]


def write_metrics_report(y_true, y_pred, probs, classes, out_dir,
                         label_names=None,
                         filename="classification_report.csv"):
    """Write one report CSV and return it as a DataFrame.

    Parameters
    ----------
    y_true, y_pred : array-like
        Class values. Any hashable labels work (mapped indices ``0..K-1`` or
        original numbers) as long as they match ``classes``.
    probs : array-like of shape (n, K) or None
        Per-class scores for AUC-PR; column ``j`` belongs to ``classes[j]``.
        If ``None``, the AUC-PR column is left blank.
    classes : sequence
        Class values to report, in row order. A subset is allowed.
    out_dir : str
        Output directory; created if missing.
    label_names : sequence of str, optional
        Row labels in the same order as ``classes``. Defaults to ``str(c)``.
    filename : str
        Output file name.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = list(classes)
    k = len(classes)
    if label_names is None:
        label_names = [str(c) for c in classes]
    label_names = [str(name) for name in label_names]
    if len(label_names) != k:
        raise ValueError("label_names and classes must have the same length")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if probs is not None:
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[1] != k:
            raise ValueError("probs must have shape (n, %d)" % k)

    # Only rows whose true label is in the requested class set contribute to
    # the reported metrics.
    in_scope = np.isin(y_true, classes)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)

    auc = np.full(k, np.nan)
    if probs is not None:
        for index, class_value in enumerate(classes):
            binary_true = (y_true == class_value).astype(np.float64)
            if binary_true.sum() > 0:
                auc[index] = average_precision_score(binary_true,
                                                     probs[:, index])

    mcc = np.array([
        matthews_corrcoef((y_true == value).astype(int),
                          (y_pred == value).astype(int))
        for value in classes
    ], dtype=np.float64)
    macro_mcc = float(np.mean(mcc))
    global_mcc = float(matthews_corrcoef(y_true[in_scope], y_pred[in_scope]))

    accuracy = (float((y_true[in_scope] == y_pred[in_scope]).mean())
                if in_scope.any() else float("nan"))

    rows = {}
    for index, name in enumerate(label_names):
        rows[name] = {
            "Precision": precision[index],
            "Recall": recall[index],
            "F1-score": f1[index],
            "AUC-PR": auc[index],
            "MCC": mcc[index],
            "Support": support[index],
        }
    rows[MACRO_ROW] = {
        "Precision": float(np.mean(precision)),
        "Recall": float(np.mean(recall)),
        "F1-score": float(np.mean(f1)),
        "AUC-PR": (float(np.nanmean(auc))
                   if not np.all(np.isnan(auc)) else np.nan),
        "MCC": macro_mcc,
        "Support": np.nan,
    }
    rows[ACC_ROW] = {
        "Precision": accuracy,
        "Recall": np.nan,
        "F1-score": np.nan,
        "AUC-PR": np.nan,
        "MCC": np.nan,
        "Support": np.nan,
    }
    rows[GLOBAL_MCC_ROW] = {
        "Precision": np.nan,
        "Recall": np.nan,
        "F1-score": np.nan,
        "AUC-PR": np.nan,
        "MCC": global_mcc,
        "Support": np.nan,
    }

    order = label_names + [MACRO_ROW, ACC_ROW, GLOBAL_MCC_ROW]
    frame = pd.DataFrame.from_dict(rows, orient="index").loc[
        order, REPORT_COLUMNS]
    frame = frame.apply(pd.to_numeric, errors="coerce").round(4)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    frame.to_csv(path, encoding="utf-8-sig", na_rep="")
    return frame
