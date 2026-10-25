#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a trained CAPOS model over precomputed features.

This is the lightweight tier of the CLI: no folding, no encoder, no Foldseek.
Give it the precomputed ``.h5`` feature files and it predicts on a laptop-class
CPU.

Feature file schema
-------------------
Both inputs are HDF5 with three datasets, one row per protein:

    ids        (n,)      protein identifiers, utf-8
    labels     (n,)      the ORIGINAL oligomeric label (1, 2, ... 12). Unused
                         at inference time, kept so the file round-trips.
    sequences  (n,)      optional raw sequences, written by the featurizers
    features   (n, d)    float32 embeddings

The two files are aligned on the intersection of their id sets, exactly as the
training pipeline does, so a protein only receives a prediction when both
branches have a row.

Output CSV
----------
    ID, Sequence, Predicted_Label, Prob_Class_<c> for c in the nine classes

``Sequence`` is filled from an optional table of id -> sequence, or from the
``sequences`` dataset the featurizers write into the .h5; when neither is
available the column is left empty rather than fabricated.
"""

import csv
import os

import numpy as np

from .models import (CAPOSModel, CAPOSSeqModel, INVERSE_LABEL_MAP,
                     LABEL_NAMES, NUM_CLASSES)

__all__ = ["load_features", "read_h5_sequences", "run_prediction",
           "write_predictions"]


def load_features(seq_path, struct_path=None):
    """Read the two feature files and align them on shared ids.

    Returns ``(seq_feats, struct_feats, ids)`` with the rows in sorted-id
    order. When ``struct_path`` is None the structure branch is a zero block
    of the same shape, which is what CAPOS-seq expects.
    """
    import h5py

    def read(path):
        try:
            h = h5py.File(path, "r")
        except Exception as e:
            raise SystemExit("!! cannot open feature file %s: %s" % (path, e))
        with h:
            missing = [d for d in ("ids", "features") if d not in h]
            if missing:
                raise SystemExit(
                    "!! %s is not a CAPOS feature file: missing dataset(s) %s.\n"
                    "   Expected HDF5 with `ids` (n,) and `features` (n, d); "
                    "`labels` (n,) is optional."
                    % (path, ", ".join(missing)))
            ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                   for x in h["ids"][:]]
            feats = np.asarray(h["features"][:], dtype=np.float32)
        return {i: k for k, i in enumerate(ids)}, feats, ids

    seq_idx, seq_feats, seq_ids = read(seq_path)
    if struct_path is None:
        ids = sorted(seq_ids)
        sel = [seq_idx[i] for i in ids]
        return (seq_feats[sel], np.zeros_like(seq_feats[sel]), ids)

    struct_idx, struct_feats, struct_ids = read(struct_path)
    common = sorted(set(seq_ids) & set(struct_ids))
    if not common:
        raise SystemExit("!! the two feature files share no ids")
    dropped_s = len(seq_ids) - len(common)
    dropped_t = len(struct_ids) - len(common)
    if dropped_s or dropped_t:
        print("  aligned %d ids (dropped %d seq-only, %d struct-only)"
              % (len(common), dropped_s, dropped_t))
    return (seq_feats[[seq_idx[i] for i in common]],
            struct_feats[[struct_idx[i] for i in common]],
            common)


def run_prediction(seq_feats, struct_feats, weights, model="capos",
                   batch_size=1024, device=None):
    """Forward the aligned features. Returns ``(probs, pred_indices)``."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class _DS(Dataset):
        def __init__(self, s, t):
            self.s, self.t = s, t

        def __len__(self):
            return len(self.s)

        def __getitem__(self, i):
            return self.s[i], self.t[i]

    # The training pipeline stores one timestep per protein and the model
    # flattens it again; keep that shape so the weights line up.
    X_s = seq_feats.reshape(seq_feats.shape[0], 1, -1)
    X_t = struct_feats.reshape(struct_feats.shape[0], 1, -1)

    if model == "capos-seq":
        net = CAPOSSeqModel(input_dim=X_s.shape[-1], num_classes=NUM_CLASSES)
    else:
        net = CAPOSModel(seq_input_dim=X_s.shape[-1],
                         struct_input_dim=X_t.shape[-1],
                         num_classes=NUM_CLASSES)

    try:
        state = torch.load(weights, map_location=device)
        net.load_state_dict(state)
    except Exception as e:
        raise SystemExit(
            "!! cannot load %s into %s.\n"
            "   Most likely the checkpoint was trained with a different "
            "feature width than the .h5 you passed\n"
            "   (CAPOS expects ESM-S + SaProt at 1280; CAPOS-seq expects "
            "ProtT5 at 1024).\n"
            "   Original error: %s" % (weights, type(net).__name__, e))
    net = net.to(device)
    net.eval()

    loader = DataLoader(_DS(X_s, X_t), batch_size=batch_size, shuffle=False)
    probs, preds = [], []
    with torch.no_grad():
        for s, t in loader:
            logits = net(s.to(device), t.to(device))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(axis=1))
    return np.concatenate(probs), np.concatenate(preds)


def write_predictions(out_csv, ids, probs, pred_indices, sequences=None):
    """One CSV: id, sequence, predicted label, class probabilities."""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    seqs = sequences or {}
    cols = ["Prob_Class_%s" % c for c in LABEL_NAMES]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Sequence", "Predicted_Label"] + cols)
        for i, pid in enumerate(ids):
            w.writerow([pid, seqs.get(str(pid), ""),
                        INVERSE_LABEL_MAP[int(pred_indices[i])]]
                       + ["%.6f" % x for x in probs[i]])
    return out_csv


def read_h5_sequences(path):
    """id -> sequence from an optional ``sequences`` dataset, or ``{}``."""
    if path is None:
        return {}
    try:
        import h5py
        with h5py.File(path, "r") as h:
            if "sequences" not in h or "ids" not in h:
                return {}
            ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                   for x in h["ids"][:]]
            seqs = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                    for x in h["sequences"][:]]
        return {i: s for i, s in zip(ids, seqs)}
    except Exception:
        return {}


def read_id_sequence_table(path):
    """id -> sequence from a FASTA or a CSV with ID / Sequence columns."""
    if path is None:
        return {}
    if path.endswith((".fa", ".fasta", ".faa", ".fas")):
        out, name, buf = {}, None, []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if name:
                        out[name] = "".join(buf)
                    name = line[1:].split()[0]
                    buf = []
                elif line:
                    buf.append(line)
            if name:
                out[name] = "".join(buf)
        return out
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if rows and "ID" in rows[0] and "Sequence" in rows[0]:
        return {r["ID"].strip(): r["Sequence"].strip() for r in rows}
    raise SystemExit("!! %s is neither FASTA nor an ID/Sequence CSV" % path)
