#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate one trained 2x2 arm on one validation structure source.

Loads a checkpoint and the selected sequence and structure feature files, then
writes predictions and metrics for that cell.

Usage:
    python eval_2x2.py --checkpoint <path> --val_seq <h5> --val_struct <h5> --out_dir <dir>
"""

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ABL = os.path.join(ROOT, "ablation", "ablation.py")


def load_ablation(path=ABL):
    spec = importlib.util.spec_from_file_location("capos_ablation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val_seq", required=True)
    ap.add_argument("--val_struct", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--ablation_mode", default="full")
    ap.add_argument("--ablation-script", default=ABL)
    args = ap.parse_args()

    for p in (args.checkpoint, args.val_seq, args.val_struct):
        if not os.path.exists(p):
            sys.exit("missing %s" % p)
    os.makedirs(args.out_dir, exist_ok=True)

    ab = load_ablation(args.ablation_script)
    ab._load_stack()
    torch = ab._torch

    X_s, X_t, y, ids = ab._load_split(args.val_seq, args.val_struct, "Val")
    print("val  : %d sequences  seq %s  struct %s"
          % (len(ids), X_s.shape, X_t.shape))
    print("ckpt : %s" % args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ab.build_model(args.ablation_mode, X_s.shape[-1], X_t.shape[-1]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    loader = ab._DataLoader(ab._OligomerDataset(X_s, X_t, y),
                            batch_size=args.batch_size, shuffle=False)
    probs, preds, labels = [], [], []
    with torch.no_grad():
        for seq, struct, lab in loader:
            logits = model(seq.to(device), struct.to(device))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(axis=1))
            labels.append(lab.numpy())

    import numpy as np
    va_probs = np.concatenate(probs)
    va_preds = np.concatenate(preds)
    va_labels = np.concatenate(labels)

    f1, mcc = ab.macro_metrics(va_labels, va_preds, ab.NUM_CLASSES)
    acc = float((va_labels == va_preds).mean())
    print()
    print("  Macro-F1 %.4f   Macro-MCC %.4f   Accuracy %.4f"
          % (f1, mcc, acc))

    ab._save_metrics(va_probs, va_preds, va_labels, ids,
                     ab.NUM_CLASSES, ab.DISPLAY, ab.LABEL_MAP, args.out_dir)
    print("  wrote %s" % args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
