#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run CAPOS fusion-operator and SaProt-channel ablation variants.

Each variant is trained and evaluated on the validation split with the
specified random seed. Per-run classification reports and prediction files are
written under the output directory.

Usage:
    python ablation.py train --ablation_mode <variant> --train_seq <h5> --train_struct <h5> --val_seq <h5> --val_struct <h5> --out_dir <dir> --seed <seed>
    python ablation.py summarize --runs <variant> [<variant> ...] --out_csv <csv>
    python ablation.py bootstrap --left <variant> --right <variant> --out_csv <csv>
"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "model")

# torch, the model stack and main.py are loaded on demand by the commands that
# need them. summarize and bootstrap only read prediction CSVs, and making them
# import torch first would mean needing the GPU environment to fold a table.

_torch = None
_nn = None
_DataLoader = None
_align_data = None
_OligomerDataset = None
_train_one_epoch = None
_evaluate = None
_save_metrics = None
_CAPOSModel = None
set_seed = None
macro_metrics = None
np = None


def _load_stack():
    """Import torch and the model stack once, for train and gate.

    Also defines ABLATIONModel and _BoundMode, which need nn.Module as a base
    class and so cannot be built before torch is in hand.
    """
    global _torch, _nn, _DataLoader
    global _align_data, _OligomerDataset, _train_one_epoch, _evaluate
    global _save_metrics, _CAPOSModel, set_seed, macro_metrics, np
    global ABLATIONModel, _BoundMode

    if _torch is not None:
        return

    import numpy
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    sys.path.insert(0, MODEL_DIR)
    from src.data_utils import align_data, OligomerDataset
    from src.engine import train_one_epoch, evaluate, save_metrics
    from src.models.capos_model import CAPOSModel

    # main.py holds set_seed and macro_metrics but is a script, not a module,
    # so it is loaded under a private name rather than as `main`.
    spec = importlib.util.spec_from_file_location(
        "capos_main", os.path.join(MODEL_DIR, "main.py"))
    capos_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capos_main)

    class ABLATIONModel(nn.Module):
        """CAPOS variant with a replaced fusion operator or a single active branch.
        """

        def __init__(self, seq_input_dim, struct_input_dim, num_classes,
                     vector_dim=256):
            super(ABLATIONModel, self).__init__()
            self.vector_dim = vector_dim

            # same order and shapes as CAPOSModel: seq_mlp, struct_mlp,
            # gate_mlp, layer_norm, classifier
            self.esms_mlp = nn.Sequential(
                nn.Linear(seq_input_dim, vector_dim),
                nn.GELU(),
                nn.Dropout(0.3)
            )
            self.saprot_mlp = nn.Sequential(
                nn.Linear(struct_input_dim, vector_dim),
                nn.GELU(),
                nn.Dropout(0.3)
            )
            self.gate_mlp = nn.Sequential(
                nn.Linear(vector_dim * 2, vector_dim),
                nn.Sigmoid()
            )
            self.layer_norm = nn.LayerNorm(vector_dim)
            self.classifier = nn.Sequential(
                nn.Linear(vector_dim, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
            # only `concat` uses it; built last on purpose
            self.concat_proj = nn.Linear(vector_dim * 2, vector_dim)

        def forward(self, input_esms, input_saprot, mode="esms_only"):
            input_esms = input_esms.view(input_esms.size(0), -1)
            input_saprot = input_saprot.view(input_saprot.size(0), -1)

            esms_feat = self.esms_mlp(input_esms)
            saprot_feat = self.saprot_mlp(input_saprot)

            if mode == "fixed_gate":
                x_fused = 0.5 * esms_feat + 0.5 * saprot_feat
            elif mode == "concat":
                x_fused = self.concat_proj(
                    torch.cat([esms_feat, saprot_feat], dim=1))
            elif mode == "esms_only":
                x_fused = esms_feat
            elif mode == "saprot_only":
                x_fused = saprot_feat
            else:
                raise ValueError("ABLATIONModel does not serve mode %r -- "
                                 "'full' is CAPOSModel" % mode)

            x_fused = self.layer_norm(x_fused)
            return self.classifier(x_fused)

    class _BoundMode(nn.Module):
        """Binds a mode into forward so the training loop can call the model
        with CAPOSModel's (seq, struct) signature and stay unaware of which
        branches are in play."""

        def __init__(self, inner, mode):
            super(_BoundMode, self).__init__()
            self.inner = inner
            self.mode = mode

        def forward(self, input_esms, input_saprot):
            return self.inner(input_esms, input_saprot, mode=self.mode)

    np = numpy
    _torch, _nn, _DataLoader = torch, nn, DataLoader
    _align_data, _OligomerDataset = align_data, OligomerDataset
    _train_one_epoch, _evaluate = train_one_epoch, evaluate
    _save_metrics, _CAPOSModel = save_metrics, CAPOSModel
    set_seed, macro_metrics = capos_main.set_seed, capos_main.macro_metrics


LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 10: 7, 12: 8}
DISPLAY = ["1", "2", "3", "4", "5", "6", "8", "10", "12"]
NUM_CLASSES = len(LABEL_MAP)

# Every candidate is run with each seed. The seed set is fixed by the
# experiment design.
SEEDS = (40, 41, 42, 43, 44)

MODES = ("full", "fixed_gate", "concat", "esms_only", "saprot_only")


def build_model(mode, seq_dim, struct_dim, num_classes=NUM_CLASSES):
    """Return the CAPOS model for `full` or an ablation variant for other modes.
    """
    if mode not in MODES:
        raise SystemExit("ablation_mode must be one of %s, got %r"
                         % (", ".join(MODES), mode))
    _load_stack()
    if mode == "full":
        return _CAPOSModel(seq_input_dim=seq_dim,
                           struct_input_dim=struct_dim,
                           num_classes=num_classes)
    return _BoundMode(
        ABLATIONModel(seq_dim, struct_dim, num_classes), mode)


def _load_split(seq_path, struct_path, tag):
    """Both feature files are always read, in every mode.

    Even a seq_only run reads the struct file: align_data keeps the ids the two
    files share, so reading both is what pins every run to one identical
    sequence list. Dropping the unused branch's file would let a run score a
    slightly different validation set and break the comparison.
    """
    _load_stack()
    X_seq, X_struct, y, ids = _align_data(seq_path, struct_path, LABEL_MAP,
                                          mode=tag)
    X_seq = X_seq.reshape(X_seq.shape[0], 1, -1)
    if X_struct is None:
        raise SystemExit("both --*_seq and --*_struct are required in every "
                         "mode so that all runs share one id list")
    X_struct = X_struct.reshape(X_struct.shape[0], 1, -1)
    return X_seq, X_struct, y, ids


def _read_report(path):
    """One classification_report.csv as {metric: float}."""
    import csv

    with open(path, encoding="utf-8-sig") as f:
        rows = {r[""]: r for r in csv.DictReader(f)}
    macro, acc = rows.get("Macro avg", {}), rows.get("Accuracy", {})
    # save_metrics leaves Support blank on the macro row and stores the
    # accuracy score under "Precision", where sklearn's report leaves it.
    return {
        "Precision": float(macro.get("Precision", "nan") or "nan"),
        "Recall": float(macro.get("Recall", "nan") or "nan"),
        "F1-score": float(macro.get("F1-score", "nan") or "nan"),
        "AUC-PR": float(macro.get("AUC-PR", "nan") or "nan"),
        "MCC": float(macro.get("MCC", "nan") or "nan"),
        "Accuracy": float(acc.get("Precision", "nan") or "nan"),
    }


METRIC_COLS = ["Precision", "Recall", "F1-score", "AUC-PR", "MCC", "Accuracy"]


def _seed_dirs(root, run):
    """[(seed, dir)] for one run, for every seed that has been scored."""
    import re
    out = []
    base = os.path.join(root, run)
    if not os.path.isdir(base):
        return out
    for entry in sorted(os.listdir(base)):
        m = re.fullmatch(r"seed(\d+)", entry)
        if not m:
            continue
        d = os.path.join(base, entry)
        if os.path.exists(os.path.join(d, "classification_report.csv")):
            out.append((int(m.group(1)), d))
    return sorted(out)


def cmd_train(args):
    _load_stack()
    if args.seed not in SEEDS:
        raise SystemExit("--seed must be one of %s" % (", ".join(map(str, SEEDS))))
    set_seed(args.seed)
    print("Seed: %d" % args.seed)
    print("Ablation mode: %s" % args.ablation_mode)
    print("Scored on: validation. The test split is never read.")
    os.makedirs(args.out_dir, exist_ok=True)

    X_tr_s, X_tr_t, y_tr, _ = _load_split(args.train_seq, args.train_struct,
                                          "Train")
    X_va_s, X_va_t, y_va, va_ids = _load_split(args.val_seq, args.val_struct,
                                               "Val")

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    print("Using Device: %s" % device)

    train_loader = _DataLoader(
        _OligomerDataset(X_tr_s, X_tr_t, y_tr),
        batch_size=args.batch_size, shuffle=True)
    val_loader = _DataLoader(
        _OligomerDataset(X_va_s, X_va_t, y_va),
        batch_size=args.batch_size, shuffle=False)

    model = build_model(args.ablation_mode,
                        X_tr_s.shape[-1], X_tr_t.shape[-1]).to(device)

    criterion = _nn.CrossEntropyLoss()
    optimizer = _torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training for %d epochs..." % args.epochs)
    for epoch in range(args.epochs):
        tr_loss, tr_acc = _train_one_epoch(model, train_loader, criterion,
                                           optimizer, device)
        va_loss, va_acc, va_probs, va_preds, va_labels = _evaluate(
            model, val_loader, criterion, device)

        va_f1, va_mcc = macro_metrics(va_labels, va_preds, NUM_CLASSES)
        print("Epoch [%d/%d] Train Loss: %.4f Acc: %.4f | Val Loss: %.4f "
              "Acc: %.4f F1: %.4f MCC: %.4f"
              % (epoch + 1, args.epochs, tr_loss, tr_acc, va_loss, va_acc,
                 va_f1, va_mcc))

    ckpt = os.path.join(args.out_dir, "ablation_model.pth")
    _torch.save(model.state_dict(), ckpt)
    print("Training completed. Model saved to %s" % ckpt)

    _save_metrics(va_probs, va_preds, va_labels, va_ids, NUM_CLASSES, DISPLAY,
                  LABEL_MAP, args.out_dir)
    print("Validation metrics written to %s" % args.out_dir)
    return 0


def cmd_summarize(args):
    """Mean and spread across seeds for each run, plus paired differences.

    A single seed cannot separate a real difference between two candidates from
    the luck of one draw. What does is the paired difference: every seed is run
    under every candidate, so subtracting within a seed cancels the difficulty
    of that particular draw, and the spread of those differences says whether
    the sign is stable.
    """
    import csv

    import numpy as np

    all_metrics = METRIC_COLS
    scores = {}          # scores[run][seed][metric]
    for run in args.runs:
        scores[run] = {}
        for seed, d in _seed_dirs(args.root, run):
            scores[run][seed] = _read_report(
                os.path.join(d, "classification_report.csv"))

    seeds = [s for s in SEEDS if all(s in scores[r] for r in args.runs)]
    if not seeds:
        raise SystemExit("no seed has every run -- nothing to summarise")

    print("=" * 96)
    print("  VALIDATION  over %d seeds (%s)   mean +/- std"
          % (len(seeds), ",".join(map(str, seeds))))
    print("=" * 96)
    hdr = ["run"] + all_metrics
    rows = []
    for run in args.runs:
        row = [run]
        for met in all_metrics:
            vals = [scores[run][s][met] for s in seeds]
            row.append("%.4f +-%.4f" % (float(np.mean(vals)),
                                        float(np.std(vals, ddof=1))
                                        if len(vals) > 1 else 0.0))
        rows.append(row)
    widths = [max(len(str(r[i])) for r in [hdr] + rows) for i in range(len(hdr))]
    fmt = "  " + "  ".join("%%-%ds" % w for w in widths)
    print(fmt % tuple(hdr))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt % tuple(r))

    out = args.out_csv
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "metric", "mean", "std"] +
                   ["seed%d" % s for s in seeds])
        for run in args.runs:
            for met in all_metrics:
                vals = [scores[run][s][met] for s in seeds]
                w.writerow([run, met, "%.4f" % float(np.mean(vals)),
                            "%.4f" % (float(np.std(vals, ddof=1))
                                      if len(vals) > 1 else 0.0)] +
                           ["%.4f" % v for v in vals])
        if args.compare:
            w.writerow([])
            w.writerow(["left", "right", "metric"] +
                       ["seed%d" % s for s in seeds] + ["mean", "std", "wins"])
            print()
            print("=" * 96)
            print("  PAIRED DIFFERENCES   (left minus right, within the same seed)")
            print("=" * 96)
            for left, right in args.compare:
                if left not in scores or right not in scores:
                    print("  !! %s vs %s skipped -- not collected" % (left, right))
                    continue
                print()
                print("  %s minus %s" % (left, right))
                print("  %-11s %s" % ("metric", "  ".join("seed%-6d" % s
                                                          for s in seeds)
                                      + "   mean      std     wins"))
                for met in all_metrics:
                    diffs = [scores[left][s][met] - scores[right][s][met]
                             for s in seeds]
                    mean = float(np.mean(diffs))
                    std = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
                    wins = sum(1 for d in diffs if d > 0)
                    print("  %-11s %s   %+8.4f  %.4f   %d/%d"
                          % (met, "  ".join("%+7.4f" % d for d in diffs),
                             mean, std, wins, len(diffs)))
                    w.writerow([left, right, met] +
                               ["%+.4f" % d for d in diffs] +
                               ["%+.4f" % mean, "%.4f" % std, wins])
    print()
    print("  'wins' counts seeds where the left run is ahead. A sign that")
    print("  holds in every seed is much stronger evidence than a mean alone.")
    print()
    print("wrote %s" % out)
    return 0


def cmd_gate(args):
    """Read the gate CAPOS actually learned, and say whether it is doing
    anything. Scored on validation, like everything else here."""
    import csv

    _load_stack()
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    X_s, X_t, y, ids = _load_split(args.seq, args.struct, "Val")

    model = build_model("full", X_s.shape[-1], X_t.shape[-1]).to(device)
    model.load_state_dict(_torch.load(args.checkpoint, map_location=device))
    model.eval()

    captured = []

    def hook(_module, _inp, out):
        captured.append(out.detach().cpu().numpy())

    handle = model.gate_mlp.register_forward_hook(hook)
    loader = _DataLoader(_OligomerDataset(X_s, X_t, y),
                         batch_size=args.batch_size, shuffle=False)
    with _torch.no_grad():
        for seq, struct, _ in loader:
            model(seq.to(device), struct.to(device))
    handle.remove()

    gate = np.concatenate(captured, axis=0)
    n, d = gate.shape
    print("checkpoint : %s" % args.checkpoint)
    print("split      : %s  (%d sequences, %d gate dims)"
          % (args.seq, n, d))
    print()
    print("  mean %.4f   sd %.4f   min %.4f   max %.4f"
          % (gate.mean(), gate.std(), gate.min(), gate.max()))
    print("  Near 0 means 'use the structure branch', near 1 'use the")
    print("  sequence branch'.")
    print()
    per_dim_mean = gate.mean(axis=0)
    per_smp_mean = gate.mean(axis=1)
    var_tot = float(np.var(gate))
    print("  across dimensions : sd %.4f   (variance share %.1f%%)"
          % (np.sqrt(float(np.var(per_dim_mean))),
             100 * float(np.var(per_dim_mean)) / var_tot if var_tot else 0))
    print("  across samples    : sd %.4f   (variance share %.1f%%)"
          % (np.sqrt(float(np.var(per_smp_mean))),
             100 * float(np.var(per_smp_mean)) / var_tot if var_tot else 0))

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "gate_values.npy"), gate)
    print("wrote %s" % out)
    return 0


def cmd_bootstrap(args):
    """Compute bootstrap confidence intervals, resampling seeds and proteins.
    """
    import csv

    import numpy as np

    sys.path.insert(0, os.path.join(ROOT, "benchmark"))
    import bootstrap_analysis as S
    # Shared bootstrap helpers used by the summarize and bootstrap commands.
    def _sig_load(path):
        return S.load_standard(path, S.LABEL9)

    def _sig_score(yt, yp, pr):
        return S.score(yt, yp, pr, len(S.CLASSES9))

    runs = [args.left] + ([args.right] if args.right else [])
    preds, ids = {}, None
    seeds = [s for s in SEEDS]
    for run in runs:
        present = dict(_seed_dirs(args.root, run))
        missing = [s for s in seeds if s not in present]
        if missing:
            raise SystemExit("%s has no results for seeds %s"
                             % (run, missing))
        preds[run] = []
        for sd in seeds:
            yt, yp, pr, these_ids = _sig_load(os.path.join(
                present[sd], "prediction_results.csv"))
            if ids is None:
                ids = these_ids
            elif these_ids != ids:
                raise SystemExit("%s seed %d is on a different id list -- "
                                 "results cannot be paired" % (run, sd))
            preds[run].append((yt, yp, pr))
    n, n_seed = len(ids), len(seeds)

    metrics = [k for k, _ in S.METRICS
               if k in ("f1", "mcc", "acc") or args.with_ap]
    labels = dict(S.METRICS)

    point = {}
    for run in runs:
        per = [_sig_score(*preds[run][i]) for i in range(n_seed)]
        point[run] = {k: float(np.mean([p[k] for p in per])) for k in metrics}

    rng = np.random.default_rng(args.seed)
    B = args.n_boot
    draws = {run: {k: np.empty(B) for k in metrics} for run in runs}
    diff = ({k: np.empty(B) for k in metrics} if args.right else None)

    for b in range(B):
        if args.axes == "seq":
            si = np.arange(n_seed)
        else:
            si = rng.integers(0, n_seed, n_seed)
        if args.axes == "seed":
            qi = np.arange(n)
        else:
            qi = rng.integers(0, n, n)

        acc = {run: {k: [] for k in metrics} for run in runs}
        for j in si:
            for run in runs:
                yt, yp, pr = preds[run][j]
                s = _sig_score(yt[qi], yp[qi], pr[qi])
                for k in metrics:
                    acc[run][k].append(s[k])
        for run in runs:
            for k in metrics:
                draws[run][k][b] = float(np.mean(acc[run][k]))
        if args.right:
            for k in metrics:
                diff[k][b] = draws[args.left][k][b] - draws[args.right][k][b]

    lo_q, hi_q = 100 * args.alpha / 2, 100 * (1 - args.alpha / 2)
    level = int(round(100 * (1 - args.alpha)))
    axis_note = {"both": "seeds then sequences (hierarchical)",
                 "seed": "seeds only, validation set whole",
                 "seq": "sequences only, averaged over seeds"}[args.axes]

    print("=" * 78)
    print("  BOOTSTRAP %d%% CI   %s" % (level, axis_note))
    print("=" * 78)
    print("  n_seeds=%d  n_val=%d  draws=%d  bootstrap_seed=%d"
          % (n_seed, n, B, args.seed))
    print()
    print("  %-28s %-22s %s" % ("run", "metric", "point [CI]"))
    print("  " + "-" * 74)
    for run in runs:
        for k in metrics:
            lo, hi = np.percentile(draws[run][k], [lo_q, hi_q])
            print("  %-28s %-22s %.4f [%.4f, %.4f]"
                  % (run, labels[k], point[run][k], lo, hi))
        print()

    rows = []
    if args.right:
        print("=" * 78)
        print("  PAIRED DIFFERENCE   %s minus %s" % (args.left, args.right))
        print("=" * 78)
        print("  %-12s %10s  %-26s %s" % ("metric", "point",
                                          "%d%% CI of the difference" % level,
                                          "verdict"))
        print("  " + "-" * 74)
        for k in metrics:
            lo, hi = np.percentile(diff[k], [lo_q, hi_q])
            verdict = "significant" if (lo > 0 or hi < 0) else "not resolved"
            d = point[args.left][k] - point[args.right][k]
            print("  %-12s %+10.4f  [%+.4f, %+.4f]        %s"
                  % (labels[k], d, lo, hi, verdict))
            rows.append([args.left, args.right, labels[k], "%.4f" % d,
                         "%.4f" % lo, "%.4f" % hi, verdict])
        print()
        print("  'significant' = the interval excludes 0 on validation. This")
        print("  is a selection-stage statement, not a test-set claim.")
        if args.axes == "both":
            print("  Five seeds is a small sample; read the seed-axis intervals")
            print("  as a lower bound on how uncertain the comparison really is.")

    out = args.out_csv
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "metric", "point", "ci_lo", "ci_hi",
                    "split", "n_seeds", "axes"])
        for run in runs:
            for k in metrics:
                lo, hi = np.percentile(draws[run][k], [lo_q, hi_q])
                w.writerow([run, labels[k], "%.4f" % point[run][k],
                            "%.4f" % lo, "%.4f" % hi, "val", n_seed, args.axes])
        if rows:
            w.writerow([])
            w.writerow(["left", "right", "metric", "point_diff",
                        "ci_lo", "ci_hi", "verdict"])
            w.writerows(rows)
    print()
    print("wrote %s" % out)
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Isolate the contribution of SaProt's explicit 3Di "
                    "tokens. Validation split only; the test split is never "
                    "read.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--ablation_mode", required=True, choices=MODES)
        p.add_argument("--batch_size", type=int, default=2048)
        return p

    tr = common(sub.add_parser("train", help="train one run on train, "
                                             "report on val"))
    tr.add_argument("--train_seq", required=True,
                    help="sequence-feature .h5 (esms or prott5)")
    tr.add_argument("--train_struct", required=True,
                    help="structure-feature .h5 (a SaProt variant)")
    tr.add_argument("--val_seq", required=True)
    tr.add_argument("--val_struct", required=True)
    tr.add_argument("--out_dir", required=True,
                    help="e.g. results/esms_saprot_normal/seed42")
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--seed", type=int, required=True,
                    help="one of %s" % (", ".join(map(str, SEEDS))))
    tr.set_defaults(func=cmd_train)

    sm = sub.add_parser("summarize",
                        help="mean/spread across seeds + paired differences")
    sm.add_argument("--root", default=os.path.join(HERE, "results"))
    sm.add_argument("--runs", nargs="+", required=True)
    sm.add_argument("--compare", nargs=2, action="append", default=[],
                    metavar=("LEFT", "RIGHT"),
                    help="pair to contrast, e.g. --compare A B; repeatable")
    sm.add_argument("--out_csv",
                    default=os.path.join(HERE, "tables", "selection.csv"))
    sm.set_defaults(func=cmd_summarize)

    ga = sub.add_parser("gate", help="inspect the learned gate on validation")
    ga.add_argument("--checkpoint", required=True)
    ga.add_argument("--seq", required=True)
    ga.add_argument("--struct", required=True)
    ga.add_argument("--batch_size", type=int, default=2048)
    ga.add_argument("--out_dir", required=True)
    ga.set_defaults(func=cmd_gate)

    bs = sub.add_parser("bootstrap",
                        help="bootstrap CIs over seeds and/or val sequences")
    bs.add_argument("--root", default=os.path.join(HERE, "results"))
    bs.add_argument("--left", required=True)
    bs.add_argument("--right", default=None,
                    help="second run; omit for an interval on --left alone")
    bs.add_argument("--n_boot", type=int, default=2000)
    bs.add_argument("--seed", type=int, default=42,
                    help="bootstrap RNG seed (not the model seeds)")
    bs.add_argument("--axes", choices=("both", "seed", "seq"), default="both",
                    help="both=hierarchical (default), seed=training "
                         "randomness only, seq=validation-set sampling only")
    bs.add_argument("--alpha", type=float, default=0.05)
    bs.add_argument("--with_ap", action="store_true",
                    help="include AUC-PR; it sorts probabilities per class "
                         "and is much slower than the other metrics")
    bs.add_argument("--out_csv",
                    default=os.path.join(HERE, "tables", "bootstrap.csv"))
    bs.set_defaults(func=cmd_bootstrap)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
