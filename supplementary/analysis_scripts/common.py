"""Shared helpers for CAPOS result summarization.

All summary scripts use the same class order, label mapping, and metric
definitions as the CAPOS benchmark. Macro MCC, Macro F1, AUC-PR, and accuracy
are computed over all nine classes; a degenerate class contributes zero.
"""
import csv
import os

import numpy as np

CLASSES = [1, 2, 3, 4, 5, 6, 8, 10, 12]
NAMES = [str(c) for c in CLASSES]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
K = len(CLASSES)


def read_report(path):
    """Read a classification_report.csv written by the CAPOS evaluation code."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = {r[""]: r for r in csv.DictReader(f)}
    macro = rows.get("Macro avg", {})
    acc = rows.get("Accuracy", {})
    return {
        "Macro-MCC": float(macro.get("MCC") or 0.0),
        "Macro-F1": float(macro.get("F1-score") or 0.0),
        "AUC-PR": float(macro.get("AUC-PR") or 0.0),
        "Accuracy": float(acc.get("Precision") or 0.0),
        "Precision": float(macro.get("Precision") or 0.0),
        "Recall": float(macro.get("Recall") or 0.0),
        "MCC": float(macro.get("MCC") or 0.0),
        "F1-score": float(macro.get("F1-score") or 0.0),
        "n": int(sum(float(rows.get(name, {}).get("Support") or 0.0)
                     for name in NAMES)),
    }


def load_predictions(path):
    """Load a prediction_results.csv file.

    Returns true/predicted integer labels (raw and mapped), class
    probabilities, and IDs. Labels outside the nine-class set map to -1 so
    that metric helpers can count them as misses.
    """
    ids, true_raw, pred_raw, probs = [], [], [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ids.append(r["ID"].strip())
            true_raw.append(int(r["True_Label"]))
            pred_raw.append(int(r["Pred_Label"]))
            probs.append([float(r.get("Prob_Class_%s" % c, 0.0) or 0.0)
                          for c in NAMES])
    yt = np.array([LABEL_MAP.get(v, -1) for v in true_raw], dtype=np.int64)
    yp = np.array([LABEL_MAP.get(v, -1) for v in pred_raw], dtype=np.int64)
    return {
        "ids": ids,
        "y_true": yt,
        "y_pred": yp,
        "probs": np.array(probs, dtype=np.float64),
        "true_raw": true_raw,
        "pred_raw": pred_raw,
    }


def _confusion(yt, yp, k):
    ok = (yp >= 0) & (yp < k)
    cm = np.bincount(yt[ok] * k + yp[ok], minlength=k * k).reshape(k, k).astype(float)
    stray = (np.bincount(yt[~ok], minlength=k).astype(float)
             if (~ok).any() else np.zeros(k))
    return cm, stray


def macro_f1(yt, yp, k=K):
    cm, stray = _confusion(yt, yp, k)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp + stray
    return float((2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)).mean())


def macro_mcc(yt, yp, k=K):
    cm, stray = _confusion(yt, yp, k)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp + stray
    total = cm.sum() + stray.sum()
    tn = total - (tp + fp + fn)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-12),
                          0.0).mean())


def macro_ap(yt, probs, k=K):
    vals = []
    for c in range(k):
        y = (yt == c).astype(int)
        if y.sum() == 0:
            vals.append(0.0)
            continue
        order = np.argsort(-probs[:, c], kind="mergesort")
        ys, ss = y[order], probs[order, c]
        tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
        idx = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
        prec = tp[idx] / np.maximum(tp[idx] + fp[idx], 1e-12)
        rec = tp[idx] / max(tp[-1], 1e-12)
        vals.append(float(np.sum((rec - np.r_[0.0, rec[:-1]]) * prec)))
    return float(np.mean(vals))


def metrics(yt, yp, probs=None):
    out = {
        "Macro-MCC": macro_mcc(yt, yp),
        "Macro-F1": macro_f1(yt, yp),
        "Accuracy": float((yt == yp).mean()) if len(yt) else float("nan"),
    }
    if probs is not None:
        out["AUC-PR"] = macro_ap(yt, probs)
    return out


def mean_std(values):
    values = [float(v) for v in values]
    if not values:
        return "", ""
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else ""
    return mean, std


def write_csv(path, header, rows):
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote %s (%d rows)" % (path, len(rows)))
