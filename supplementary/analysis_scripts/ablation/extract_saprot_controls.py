# -*- coding: utf-8 -*-
"""Extract SaProt embeddings with controlled amino-acid and 3Di channels.

Supported channel modes are normal, mask_3di, mask_aa, and shuffle_3di. The
resulting embeddings are used by the ablation and structure-branch analyses.

Usage:
    python extract_saprot_controls.py --mode <normal|mask_3di|mask_aa|shuffle_3di> --csv <input.csv> --out <file>
"""
import argparse
import hashlib
import json
import os
import random
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# The two-character SaProt token is <AA><3Di lowercased>. '#' is the vocabulary's
# "no information" state for either half.
MASK_CHAR = "#"
AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
DI_ALPHABET = set("acdefghiklmnpqrstvwy")

MODES = ("normal", "mask_3di", "shuffle_3di", "mask_aa")


def _sanitise(chars, allowed, tag, counter):
    """Map out-of-alphabet characters to MASK_CHAR and count them."""
    out = []
    for c in chars:
        if c in allowed:
            out.append(c)
        else:
            out.append(MASK_CHAR)
            counter["bad_" + tag] += 1
    return out


def apply_mode(seq, struc_seq, mode, seed, prot_id, counter):
    """Turn the raw (aa, 3Di) strings into the controlled pair.

    Both strings are the same length and stay that way; only their contents
    change. Returns (seq, struc_seq) ready to be zipped into combined_seq.
    """
    if len(seq) != len(struc_seq):
        counter["length_mismatch"] += 1
        return None, None

    seq = _sanitise(seq, AA_ALPHABET, "aa", counter)
    # foldseek emits uppercase 3Di; the vocabulary's 3Di half is lowercase, so
    # lowercase first and only then check membership -- checking an uppercased
    # letter against the lowercase alphabet would mark every state invalid.
    struc_seq = _sanitise([c.lower() for c in struc_seq], DI_ALPHABET, "di",
                          counter)

    if mode == "normal":
        return seq, struc_seq

    if mode == "mask_3di":
        return seq, [MASK_CHAR] * len(struc_seq)

    if mode == "mask_aa":
        return [MASK_CHAR] * len(seq), struc_seq

    if mode == "shuffle_3di":
        # Per-chain permutation of the 3Di stream only. The amino-acid order is
        # untouched, so what is destroyed is the correspondence between a
        # residue's identity and its geometric state. The whole stream is
        # permuted, including any '#' the pLDDT mask already placed -- those
        # are structure-derived too, and leaving them in place would keep a
        # positional structure signal alive.
        key = "%d:%s" % (seed, prot_id)
        rng = random.Random(
            int.from_bytes(hashlib.sha256(key.encode()).digest(), "big"))
        shuffled = list(struc_seq)
        rng.shuffle(shuffled)
        return seq, shuffled

    raise ValueError("unknown mode %r" % mode)


def build_combined(seq, struc_seq):
    """Same construction as utils/foldseek_util.get_struc_seq."""
    return "".join(a + b for a, b in zip(seq, struc_seq))


def get_saprot_embedding(model, tokenizer, device, foldseek_bin, cif_path,
                         mode, seed, prot_id, counter, chain_id="A"):
    from utils.foldseek_util import get_struc_seq

    try:
        # plddt_mask=True matches the feature-extraction pipeline, so the
        # channel control is the only difference between modes.
        parsed_seqs = get_struc_seq(foldseek_bin, cif_path, [chain_id],
                                    plddt_mask=True)
        if parsed_seqs is None or len(parsed_seqs) == 0:
            counter["empty_parse"] += 1
            return None

        if chain_id in parsed_seqs:
            seq, struc_seq, _ = parsed_seqs[chain_id]
        else:
            first_chain = list(parsed_seqs.keys())[0]
            seq, struc_seq, _ = parsed_seqs[first_chain]
            counter["wrong_chain"] += 1

        seq, struc_seq = apply_mode(seq, struc_seq, mode, seed, prot_id,
                                    counter)
        if seq is None:
            return None

        combined_seq = build_combined(seq, struc_seq)
        inputs = tokenizer(combined_seq, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # A controlled chain should not contain unknown tokens.
        if tokenizer.unk_token_id in inputs["input_ids"]:
            counter["unk_token"] += 1

        with torch.no_grad():
            embeddings = model.get_hidden_states(inputs, reduction="mean")
            feat = embeddings[0].squeeze(0)
            return feat.float().cpu().numpy()
    except Exception:
        counter["exception"] += 1
        return None


def main():
    parser = argparse.ArgumentParser(
        description="SaProt embeddings with the 3Di/AA channels controlled.")
    parser.add_argument("--config", default=None,
                        help="Path to JSON configuration file")
    parser.add_argument("--csv", default=None,
                        help="Input CSV (ID, Label, path)")
    parser.add_argument("--out", default=None, help="Output .h5 path")
    parser.add_argument("--model_path", default=None,
                        help="SaProt weights directory")
    parser.add_argument("--saprot_repo", default=None,
                        help="SaProt repository root")
    parser.add_argument("--foldseek_bin", default=None,
                        help="Foldseek binary")
    parser.add_argument("--structure_dir", default=None,
                        help="Fallback if the CSV has no `path` column")
    parser.add_argument("--mode", default=None, choices=MODES,
                        help="Which channel to keep (required)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Shuffle seed; per-sequence stream is "
                             "sha256('seed:ID')")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config_args = json.load(f)
        cli_args = [a.lstrip("-").split("=")[0] for a in sys.argv[1:]]
        for key, value in config_args.items():
            if key not in cli_args:
                setattr(args, key, value)

    required = ["csv", "out", "model_path", "saprot_repo", "foldseek_bin",
                "mode"]
    missing = [k for k in required if getattr(args, k) is None]
    if missing:
        parser.error("missing required arguments: %s"
                     % ", ".join("--" + k for k in missing))

    sys.path.append(args.saprot_repo)
    try:
        from model.saprot.base import SaprotBaseModel
        from transformers import EsmTokenizer
    except ImportError:
        print("Error: import failed. %s must be the SaProt repository root."
              % args.saprot_repo)
        sys.exit(1)

    if not os.path.exists(args.foldseek_bin):
        print("Error: Foldseek not found at %s" % args.foldseek_bin)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("mode      : %s" % args.mode)
    print("seed      : %d" % args.seed)
    print("device    : %s" % device)

    model = SaprotBaseModel(task="base", config_path=args.model_path,
                            load_pretrained=True)
    tokenizer = EsmTokenizer.from_pretrained(args.model_path)
    if device.type == "cuda":
        model.half()
    model.to(device)
    model.eval()

    df = pd.read_csv(args.csv)
    df["ID"] = df["ID"].astype(str).str.strip()
    print("input     : %s  (n=%d)" % (args.csv, len(df)))

    counter = {"bad_aa": 0, "bad_di": 0, "length_mismatch": 0,
               "empty_parse": 0, "wrong_chain": 0, "unk_token": 0,
               "exception": 0}

    ids_list, labels_list, features_list = [], [], []
    pbar = tqdm(df.iterrows(), total=len(df),
                desc="SaProt/%s" % args.mode)

    for _, row in pbar:
        prot_id = row["ID"]
        label = row["Label"] if "Label" in row else row.get("label", -1)

        cif_path = row.get("path") if "path" in row else None
        if not isinstance(cif_path, str) or not cif_path.strip():
            if args.structure_dir is None:
                sys.exit("%s: CSV has no usable `path` column and "
                         "--structure_dir was not given" % prot_id)
            cif_path = os.path.join(args.structure_dir, "%s.cif" % prot_id)
        else:
            cif_path = cif_path.strip()

        emb = get_saprot_embedding(model, tokenizer, device, args.foldseek_bin,
                                   cif_path, args.mode, args.seed, prot_id,
                                   counter)
        if emb is not None:
            ids_list.append(prot_id)
            labels_list.append(label)
            features_list.append(emb)
        else:
            pbar.write("Failed: %s" % prot_id)

    print("done: %d ok / %d failed" % (len(ids_list), len(df) - len(ids_list)))
    for k in sorted(counter):
        if counter[k]:
            print("  %-16s %d" % (k, counter[k]))
    if counter["unk_token"]:
        print("  !! <unk> appeared in the controlled stream -- the features "
              "are not what this mode claims")

    if not ids_list:
        print("No features extracted.")
        sys.exit(1)

    features_array = np.vstack(features_list)
    labels_array = np.array(labels_list, dtype=np.int64)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Saving %s  shape=%s" % (args.out, features_array.shape))
    with h5py.File(args.out, "w") as f:
        dt_str = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("ids", data=np.array(ids_list, dtype=object),
                         dtype=dt_str)
        f.create_dataset("labels", data=labels_array)
        f.create_dataset("features", data=features_array,
                         compression="gzip", compression_opts=4)
        f.attrs["n_sequences"] = len(ids_list)
        f.attrs["feature_dim"] = features_array.shape[1]
        f.attrs["saprot_mode"] = args.mode
        f.attrs["shuffle_seed"] = args.seed
    print("Done.")


if __name__ == "__main__":
    main()
