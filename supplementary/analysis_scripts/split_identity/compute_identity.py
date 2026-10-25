#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute nearest-neighbour sequence identity between dataset splits.

Measures both the cluster-based split and a size-matched random split, using
full-length identity and MMseqs2 searches.

Usage:
    python compute_identity.py --base <CAPOS root> --mmseqs <mmseqs binary> [--out_dir <dir>]
"""
import argparse
import csv
import os
import random
import subprocess
import sys

SPLIT_DIR = None
RAW_CSV = None
CLUSTER_FILES = {}

# (name, query split, target split) -- the three panels of each row
PAIRS = [
    ("train_val", "val", "train"),
    ("train_test", "test", "train"),
    ("val_test", "val", "test"),
]

SEED = 42


def read_rows(path):
    rows = []
    seen = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            pid = r["ID"].strip()
            if pid in seen:
                continue
            seen.add(pid)
            rows.append((pid, r["Sequence"].strip()))
    return rows


def write_fasta(rows, path):
    with open(path, "w", encoding="utf-8") as out:
        for pid, seq in rows:
            out.write(">%s\n%s\n" % (pid, seq))
    return len(rows)


def build_splits():
    """The cluster-based split, plus a same-size random shuffle of the pool."""
    cluster = {name: read_rows(p) for name, p in CLUSTER_FILES.items()}
    sizes = {name: len(rows) for name, rows in cluster.items()}

    pool = read_rows(RAW_CSV)
    n_pool = len(pool)
    n_split = sum(sizes.values())
    if n_pool != n_split:
        print("  note: pool has %d sequences, the three splits sum to %d"
              % (n_pool, n_split))

    rng = random.Random(SEED)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    rand, i = {}, 0
    for name in ("train", "val", "test"):
        rand[name] = shuffled[i:i + sizes[name]]
        i += sizes[name]

    print("  cluster split sizes: %s" % sizes)
    print("  random split sizes : %s" % {k: len(v) for k, v in rand.items()})
    print("  (identical by construction; only the grouping differs)")
    return {"cluster": cluster, "random": rand}


def run_search(mmseqs, query_fa, target_fa, out_tsv, tmp, cov, sens, threads, reuse_tsv=False):
    cmd = [
        mmseqs, "easy-search", query_fa, target_fa, out_tsv, tmp,
        "--min-seq-id", "0.0",
        "-c", str(cov), "--cov-mode", "0",
        "-s", str(sens),
        # full-length identity: nident / max(qlen, tlen)
        "--seq-id-mode", "2",
        "--format-output",
        "query,target,nident,qlen,tlen,alnlen,pident,qcov,tcov",
        "--threads", str(threads), "-v", "1",
    ]
    if reuse_tsv and os.path.exists(out_tsv) and os.path.getsize(out_tsv) > 0:
        print("    reusing existing %s (%d bytes) -- search skipped"
              % (out_tsv, os.path.getsize(out_tsv)))
        return
    print("    $ " + " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit("    !! mmseqs failed (exit %d)" % r.returncode)


def main():
    global SPLIT_DIR, RAW_CSV, CLUSTER_FILES
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="CAPOS repository root")
    ap.add_argument("--mmseqs", required=True, help="MMseqs2 binary")
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "identity"))
    ap.add_argument("--cov", type=float, default=0.8)
    ap.add_argument("--sensitivity", type=float, default=7.5)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--reuse-tsv", action="store_true",
                    help="keep existing <pair>.tsv files and only re-parse "
                         "them into the CSVs (skips the mmseqs search)")
    args = ap.parse_args()

    SPLIT_DIR = os.path.join(args.base, "dataset/dataset_split/splits")
    RAW_CSV = os.path.join(args.base, "dataset/merged_datasets/dataset_raw.csv")
    CLUSTER_FILES = {
        "train": os.path.join(SPLIT_DIR, "dataset_raw_train.csv"),
        "val": os.path.join(SPLIT_DIR, "val.csv"),
        "test": os.path.join(SPLIT_DIR, "test.csv"),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 78)
    print("  building both splits")
    print("=" * 78)
    splits = build_splits()

    for mode in ("cluster", "random"):
        fa_dir = os.path.join(args.out_dir, mode, "fasta")
        os.makedirs(fa_dir, exist_ok=True)
        fasta = {}
        print()
        print("=" * 78)
        print("  %s split" % mode.upper())
        print("=" * 78)
        for name, rows in splits[mode].items():
            p = os.path.join(fa_dir, "%s.fasta" % name)
            n = write_fasta(rows, p)
            fasta[name] = p
            print("  %-6s %7d sequences -> %s" % (name, n, p))

        for pair, q, t in PAIRS:
            print()
            print("  --- %s :  query=%s  target=%s ---" % (pair, q, t))
            out_dir = os.path.join(args.out_dir, mode)
            tsv = os.path.join(out_dir, "%s.tsv" % pair)
            tmp = os.path.join(out_dir, "tmp_%s" % pair)
            run_search(args.mmseqs, fasta[q], fasta[t], tsv, tmp,
                       args.cov, args.sensitivity, args.threads,
                       reuse_tsv=args.reuse_tsv)

            # Columns: query target nident qlen tlen alnlen pident qcov tcov.
            # With --seq-id-mode 2, MMseqs2 reports pident/fident as
            # nident / max(qlen, tlen), the full-length identity. Use that
            # value rather than rebuilding it from nident.
            best = {}
            n_rows = 0
            with open(tsv, newline="", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 7:
                        continue
                    qid = parts[0]
                    fid = float(parts[6]) / 100.0   # pident, seq-id-mode 2
                    n_rows += 1
                    if qid not in best or fid > best[qid]:
                        best[qid] = fid
            print("      %d hits parsed "
                  "(identity = nident / max(qlen, tlen), mmseqs --seq-id-mode 2)"
                  % n_rows)

            rows = [(pid, best.get(pid)) for pid, _ in splits[mode][q]]
            out_csv = os.path.join(out_dir, "%s.csv" % pair)
            with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["query_id", "max_identity"])
                for pid, v in rows:
                    w.writerow([pid, "" if v is None else "%.4f" % v])

            vals = [v for _, v in rows if v is not None]
            n_none = len(rows) - len(vals)
            n_ge = sum(1 for v in vals if v >= 0.30)
            print("      queries %d   no hit %d (%.1f%%)   >=30%%: %d (%.1f%% of all)"
                  % (len(rows), n_none, 100.0 * n_none / len(rows),
                     n_ge, 100.0 * n_ge / len(rows)))

    print()
    print("  done. Now run plot_identity.py to draw the 2x3 figure.")


if __name__ == "__main__":
    main()
