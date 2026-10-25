#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare flat structure directories for the Foldseek-Trans baseline.

The selected monomer model of every protein is linked into one flat directory
per split. Foldseek's ``createdb`` and ``easy-search`` read files directly from
the directory they are given, so flat directories avoid searching every decoy
model and every nested experiment directory.

The selection table must contain ``ID``, ``path``, ``pass`` and ``Label``
columns. Only rows whose ``pass`` value is true are linked. Paths are resolved
relative to the selection table when they are not absolute.

Outputs:

    <baseline>/structures/train/<ID>.cif
    <baseline>/structures/test/<ID>.cif
    <baseline>/structures/train_ids.txt
    <baseline>/structures/test_ids.txt
"""
import argparse
import csv
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.dirname(CURRENT_DIR)
BENCH_DIR = os.path.dirname(BASELINE_DIR)
RELEASE_ROOT = os.path.dirname(BENCH_DIR)
DATASETS_DIR = os.path.join(RELEASE_ROOT, "datasets")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build flat structure directories for the Foldseek baseline")
    parser.add_argument("--selection_csv",
                        default=os.path.join(DATASETS_DIR,
                                             "structure_selection.csv"),
                        help="table with ID,path,pass,Label columns")
    parser.add_argument("--train_csv",
                        default=os.path.join(DATASETS_DIR,
                                             "dataset_5_train.csv"),
                        help="training split CSV")
    parser.add_argument("--test_csv",
                        default=os.path.join(DATASETS_DIR, "test.csv"),
                        help="test split CSV")
    parser.add_argument("--out_dir",
                        default=os.path.join(BASELINE_DIR, "structures"),
                        help="output directory for flat structure folders")
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    args = parse_args()
    if not os.path.exists(args.selection_csv):
        raise SystemExit("selection table not found: %s (pass --selection_csv)"
                         % args.selection_csv)
    selection_dir = os.path.dirname(os.path.abspath(args.selection_csv))

    selected = {}
    for row in read_csv(args.selection_csv):
        identifier = row["ID"].strip()
        source = (row.get("path") or "").strip()
        if source and not os.path.isabs(source):
            source = os.path.join(selection_dir, source)
        selected[identifier] = {
            "path": source,
            "pass": (row.get("pass") or "").strip().lower() == "true",
        }
    print("selection table: %d rows" % len(selected))

    splits = [("train", args.train_csv), ("test", args.test_csv)]
    for name, csv_path in splits:
        rows = read_csv(csv_path)
        before = len(rows)
        kept = [row for row in rows
                if selected.get(row["ID"].strip(), {}).get("pass")]
        print("%s: %d -> %d rows after the pass filter"
              % (name, before, len(kept)))

        destination = os.path.join(args.out_dir, name)
        os.makedirs(destination, exist_ok=True)

        linked_ids = []
        missing_path = missing_file = 0
        for row in kept:
            identifier = row["ID"].strip()
            source = selected[identifier]["path"]
            if not source:
                missing_path += 1
                continue
            if not os.path.exists(source):
                missing_file += 1
                continue
            link = os.path.join(destination, identifier + ".cif")
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(source, link)
            linked_ids.append(identifier)

        if missing_path or missing_file:
            raise SystemExit(
                "%s: %d rows have no recorded structure, %d paths are missing"
                % (name, missing_path, missing_file))

        id_file = os.path.join(args.out_dir, "%s_ids.txt" % name)
        with open(id_file, "w") as handle:
            handle.write("\n".join(linked_ids) + "\n")
        print("  wrote %s (%d links)" % (id_file, len(linked_ids)))
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
