# -*- coding: utf-8 -*-
"""Read the per-split feature files used for training.

The schema is the same one ``capos predict`` consumes: HDF5 with ``ids``,
``labels`` and ``features``. Training uses the labels; inference ignores them.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_h5(path, mode):
    try:
        h = h5py.File(path, "r")
    except Exception as e:
        raise SystemExit("!! cannot open %s features %s: %s" % (mode, path, e))
    with h:
        missing = [d for d in ("ids", "labels", "features") if d not in h]
        if missing:
            raise SystemExit(
                "!! %s is not a training feature file: missing dataset(s) %s.\n"
                "   Expected HDF5 with `ids`, `labels` and `features`."
                % (path, ", ".join(missing)))
        ids = [pid.decode("utf-8") if isinstance(pid, bytes) else str(pid)
               for pid in h["ids"][:]]
        return ids, h["features"][:], h["labels"][:]


def align_data(seq_path, struct_path, label_map, mode="Train"):
    """Load one split and align the two branches on shared ids.

    Returns ``(seq_features, struct_features, mapped_labels, ids)``.
    ``struct_features`` is ``None`` for the sequence-only model.
    """
    print("========== Loading %s Data ==========" % mode)
    seq_ids, seq_feats, seq_labels = _read_h5(seq_path, mode)

    if struct_path is None:
        try:
            y = np.array([label_map[int(l)] for l in seq_labels], dtype=np.int64)
        except KeyError as e:
            raise SystemExit(
                "!! %s contains label %s, which is not one of %s"
                % (seq_path, e.args[0], sorted(label_map)))
        print("[%s] Sequence-only mode. Loaded samples: %d" % (mode, len(seq_ids)))
        return (np.asarray(seq_feats, dtype=np.float32), None, y, seq_ids)

    struct_ids, struct_feats, _ = _read_h5(struct_path, mode)
    print("[%s] Raw samples - sequence: %d | structure: %d"
          % (mode, len(seq_ids), len(struct_ids)))

    seq_idx = {pid: i for i, pid in enumerate(seq_ids)}
    struct_idx = {pid: i for i, pid in enumerate(struct_ids)}
    common = sorted(set(seq_ids) & set(struct_ids))
    print("[%s] Aligned samples: %d" % (mode, len(common)))
    if not common:
        raise SystemExit("!! %s has no ids shared by the sequence and structure files"
                         % mode)

    try:
        y = np.array([label_map[int(seq_labels[seq_idx[pid]])] for pid in common],
                     dtype=np.int64)
    except KeyError as e:
        raise SystemExit(
            "!! %s contains label %s, which is not one of %s"
            % (seq_path, e.args[0], sorted(label_map)))

    seq_out = np.asarray([seq_feats[seq_idx[pid]] for pid in common], dtype=np.float32)
    struct_out = np.asarray([struct_feats[struct_idx[pid]] for pid in common],
                            dtype=np.float32)
    return seq_out, struct_out, y, common


class OligomerDataset(Dataset):
    def __init__(self, seq_feats, struct_feats, labels):
        self.seq_feats = seq_feats
        self.struct_feats = struct_feats
        self.labels = labels

    def __len__(self):
        return len(self.seq_feats)

    def __getitem__(self, idx):
        seq = self.seq_feats[idx]
        label = self.labels[idx]
        if self.struct_feats is not None:
            struct = self.struct_feats[idx]
        else:
            struct = np.zeros_like(seq, dtype=np.float32)
        return seq, struct, label
