#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``capos`` -- predict homo-oligomeric state from protein sequences.

    capos predict --seq-feat esms.h5 --struct-feat saprot.h5 \
                  --weights capos_model.pth --out predictions.csv

    capos run input.fasta --work-dir out/            # full pipeline

The command is split so that the expensive half is optional:

    predict     features -> predictions. CPU-only, minutes, no encoder needed.
                This is the out-of-the-box example: hand it the precomputed
                .h5 files that ship with the release.
    featurize   sequences / structures -> features. Needs ESM-S and SaProt,
                or ProtT5 for the sequence-only model.
    fold        FASTA -> Chai-1 monomer models. Needs a GPU and chai-lab.
    run         fold + featurize + predict in one call (CAPOS).
    train       features + labels -> a CAPOS or CAPOS-seq checkpoint.
"""

import argparse
import os
import sys

__all__ = ["main"]


def _repo_root():
    """Source tree that holds the full-tier helper scripts, or None.

    The light-tier wheel contains no helper scripts: ``featurize`` and ``fold``
    shell out to ``tools/``. A source checkout is found automatically; an
    installed package needs ``CAPOS_REPO_ROOT`` (the full Docker image sets it).
    """
    env = os.environ.get("CAPOS_REPO_ROOT")
    if env:
        return os.path.abspath(env)

    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in (package_root, os.path.dirname(package_root), os.getcwd()):
        if os.path.isdir(os.path.join(candidate, "tools")):
            return candidate
    return None


def _checkpoint_candidates(name):
    """Places a bundled checkpoint may live, in priority order."""
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = []

    env = os.environ.get("CAPOS_WEIGHTS")
    if env:
        # CAPOS_WEIGHTS names the CAPOS checkpoint; CAPOS-seq sits beside it.
        candidates.append(env if name == "capos_model.pth"
                          else os.path.join(os.path.dirname(env), name))

    candidates.append(os.path.join(package_root, "weights", name))
    repo = _repo_root()
    if repo:
        # weights/ sits at the repository root in this distribution.
        candidates.append(os.path.join(repo, "weights", name))
        candidates.append(os.path.join(repo, "packaging", "weights", name))
        sub = "capos" if name == "capos_model.pth" else "capos-seq"
        candidates.append(os.path.join(repo, "model", "weight", sub, name))
    return candidates


def _find_checkpoint(name):
    for path in _checkpoint_candidates(name):
        if os.path.exists(path):
            return path
    return None


def build_parser():
    from . import __version__

    p = argparse.ArgumentParser(
        prog="capos",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1] if "\n\n" in __doc__ else None)
    p.add_argument("--version", action="version", version="capos %s" % __version__)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("predict", help="features -> oligomeric-state calls")
    sp.add_argument("--seq-feat", required=True,
                    help="sequence-feature .h5 (ESM-S)")
    sp.add_argument("--struct-feat", default=None,
                    help="structure-feature .h5 (SaProt). Omit for the "
                         "sequence-only model.")
    sp.add_argument("--weights", required=True, help="model checkpoint .pth")
    sp.add_argument("--model", default="capos", choices=["capos", "capos-seq"],
                    help="which architecture the checkpoint belongs to")
    sp.add_argument("--sequences", default=None,
                    help="FASTA or ID/Sequence CSV, only used to fill the "
                         "Sequence column of the output")
    sp.add_argument("--out", default="capos_predictions.csv")
    sp.add_argument("--batch-size", type=int, default=1024)

    sp = sub.add_parser("featurize", help="sequences/structures -> features")
    sp.add_argument("--input", required=True,
                    help="CSV with ID, Sequence (and optionally Label; "
                         "structure paths for SaProt)")
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--structure-dir", default=None,
                    help="directory of per-id CIF models for the SaProt branch")
    sp.add_argument("--esms-model-dir", default=None,
                    help="directory holding the ESM-S weights")
    sp.add_argument("--esms-ckpt", default=None,
                    help="ESM-S pre-trained weights (.pth). Required by "
                         "extract_esms.py; falls back to $CAPOS_ESMS_CKPT")
    sp.add_argument("--saprot-model-path", default=None,
                    help="SaProt weights directory (with config.json)")
    sp.add_argument("--saprot-repo", default=None,
                    help="SaProt repository root. Required by "
                         "extract_saprot.py; falls back to $CAPOS_SAPROT_REPO")
    sp.add_argument("--foldseek-bin", default=None,
                    help="path to the Foldseek binary. Required by "
                         "extract_saprot.py; falls back to $CAPOS_FOLDSEEK_BIN")
    sp.add_argument("--prott5-model", default=None,
                    help="ProtT5 model name or local path; produces prott5.h5 "
                         "for CAPOS-seq. Falls back to $CAPOS_PROTT5_MODEL.")
    sp.add_argument("--plddt-mask", default="auto",
                    choices=["true", "false", "auto"],
                    help="SaProt B-factor masking: should the B-factor be read "
                         "as pLDDT? 'true' masks low-confidence 3Di tokens -- "
                         "correct for PREDICTED structures (Chai-1, AlphaFold). "
                         "'false' never masks -- REQUIRED for EXPERIMENTAL "
                         "structures, whose B-factor is a crystallographic "
                         "displacement (low = well ordered) and masking would "
                         "empty the structural channel. 'auto' (default) infers "
                         "which case it is from the structure file.")
    sp.add_argument("--batch-size", type=int, default=16)

    sp = sub.add_parser("fold", help="FASTA -> Chai-1 monomer models")
    sp.add_argument("--input", required=True, help="FASTA of query sequences")
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--workers", type=int, default=1,
                    help="reserved for future parallel folding; folds are "
                         "currently run serially, one at a time.")

    sp = sub.add_parser("run", help="fold + featurize + predict")
    sp.add_argument("fasta", help="input FASTA")
    sp.add_argument("--work-dir", default="capos_out")
    sp.add_argument("--weights", default=None,
                    help="default: the checkpoint shipped with the package")
    sp.add_argument("--model", default="capos", choices=["capos"],
                    help="CAPOS full pipeline. For CAPOS-seq, extract ProtT5 "
                         "features separately and use `capos predict "
                         "--model capos-seq`.")
    sp.add_argument("--batch-size", type=int, default=1024)
    sp.add_argument("--workers", type=int, default=1)
    sp.add_argument("--plddt-mask", default="auto",
                    choices=["true", "false", "auto"],
                    help="SaProt B-factor masking (see 'featurize'). This "
                         "subcommand folds with Chai-1, so its structures are "
                         "predictions and masking should be on -- leave the "
                         "default or pass 'true'.")

    sp = sub.add_parser("train", help="features + labels -> checkpoint")
    sp.add_argument("--model", default="capos", choices=["capos", "capos-seq"],
                    help="CAPOS takes ESM-S + SaProt; CAPOS-seq takes ProtT5")
    sp.add_argument("--train-seq", required=True,
                    help="training sequence-feature .h5")
    sp.add_argument("--train-struct", default=None,
                    help="training structure-feature .h5 (CAPOS only)")
    sp.add_argument("--val-seq", required=True,
                    help="validation sequence-feature .h5")
    sp.add_argument("--val-struct", default=None,
                    help="validation structure-feature .h5 (CAPOS only)")
    sp.add_argument("--out-dir", default="./capos_training",
                    help="checkpoint, history and validation report")
    sp.add_argument("--epochs", type=int, default=100)
    sp.add_argument("--batch-size", type=int, default=2048)
    sp.add_argument("--lr", type=float, default=None,
                    help="default: 1e-4 for capos, 1e-3 for capos-seq")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    sub.add_parser("check",
                   help="report what is installed and what each "
                        "command needs")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.command == "predict":
        return _cmd_predict(args)

    if args.command == "featurize":
        return _cmd_featurize(args)

    if args.command == "train":
        return _cmd_train(args)

    if args.command == "fold":
        return _cmd_fold(args)

    if args.command == "run":
        return _cmd_run(args)

    if args.command == "check":
        return _cmd_check(args)
    raise SystemExit("unknown command %r" % args.command)


def _cmd_predict(args):
    from . import predict as P

    if not os.path.exists(args.seq_feat):
        sys.exit("!! missing %s" % args.seq_feat)
    if args.model == "capos" and not args.struct_feat:
        sys.exit("!! --model capos needs --struct-feat")
    if not os.path.exists(args.weights):
        sys.exit("!! missing %s" % args.weights)

    seq, struct, ids = P.load_features(args.seq_feat, args.struct_feat)
    probs, preds = P.run_prediction(
        seq, struct, args.weights, model=args.model,
        batch_size=args.batch_size)
    seqs = P.read_h5_sequences(args.seq_feat)
    seqs.update(P.read_id_sequence_table(args.sequences))
    P.write_predictions(args.out, ids, probs, preds, sequences=seqs)

    print("\n%d proteins -> %s" % (len(ids), args.out))
    return 0


def _cmd_featurize(args):
    # The encoders are installed separately (see environment.yml) and are
    # invoked as subprocesses, so this package does not vendor torchdrug /
    # SaProt and the light tier stays installable without them.
    import subprocess
    import sys as _sys

    repo = _repo_root()
    if repo is None:
        sys.exit("!! feature extractors not found: set CAPOS_REPO_ROOT to the "
                 "CAPOS source checkout (the full Docker image sets it)")
    tools_dir = os.path.join(repo, "tools")
    if not os.path.isdir(tools_dir):
        sys.exit("!! feature extractors not found at %s" % tools_dir)

    # extract_esms.py needs csv/out/model_dir/ckpt; extract_saprot.py needs
    # csv/out/model_path/saprot_repo/foldseek_bin. The last group can be taken
    # from the environment so `capos run` does not need every flag spelled out.
    prott5_model = getattr(args, "prott5_model", None)
    if prott5_model is None:
        prott5_model = os.environ.get("CAPOS_PROTT5_MODEL")
    if prott5_model == "":
        prott5_model = None

    if not args.esms_model_dir and not args.saprot_model_path and not prott5_model:
        sys.exit("!! nothing to extract: pass --esms-model-dir, "
                 "--saprot-model-path and/or --prott5-model")

    os.makedirs(args.out_dir, exist_ok=True)

    esms_ckpt = getattr(args, "esms_ckpt", None) or os.environ.get("CAPOS_ESMS_CKPT")
    saprot_repo = getattr(args, "saprot_repo", None) or os.environ.get("CAPOS_SAPROT_REPO")
    foldseek_bin = getattr(args, "foldseek_bin", None) or os.environ.get("CAPOS_FOLDSEEK_BIN")

    if args.esms_model_dir:
        if not esms_ckpt:
            sys.exit("!! --esms-model-dir needs --esms-ckpt (or $CAPOS_ESMS_CKPT)")
        out = os.path.join(args.out_dir, "esms.h5")
        subprocess.run([_sys.executable,
                        os.path.join(tools_dir, "extract_esms.py"),
                        "--csv", args.input, "--out", out,
                        "--model_dir", args.esms_model_dir,
                        "--ckpt", esms_ckpt,
                        "--batch_size", str(args.batch_size)],
                       check=True)

    if args.saprot_model_path:
        if not args.structure_dir:
            sys.exit("!! --saprot-model-path needs --structure-dir")
        if not saprot_repo:
            sys.exit("!! --saprot-model-path needs --saprot-repo "
                     "(or $CAPOS_SAPROT_REPO)")
        if not foldseek_bin:
            sys.exit("!! --saprot-model-path needs --foldseek-bin "
                     "(or $CAPOS_FOLDSEEK_BIN)")
        out = os.path.join(args.out_dir, "saprot.h5")
        cmd = [_sys.executable,
               os.path.join(tools_dir, "extract_saprot.py"),
               "--csv", args.input, "--out", out,
               "--structure_dir", args.structure_dir,
               "--model_path", args.saprot_model_path,
               "--saprot_repo", saprot_repo,
               "--foldseek_bin", foldseek_bin,
               "--plddt_mask", args.plddt_mask]
        subprocess.run(cmd, check=True)

    if prott5_model:
        out = os.path.join(args.out_dir, "prott5.h5")
        cmd = [_sys.executable,
               os.path.join(tools_dir, "extract_prott5.py"),
               "--csv", args.input, "--out", out,
               "--model_name", prott5_model,
               "--batch_size", str(args.batch_size)]
        subprocess.run(cmd, check=True)

    return 0


def _cmd_train(args):
    try:
        from .training import run_training
    except ImportError as e:
        sys.exit("!! training dependencies missing: pip install 'capos[train]' "
                 "(import error: %s)" % e)
    run_training(args)
    return 0


def _chai1_cmd():
    """How to invoke Chai-1.

    Structure prediction is environment #2 and is never imported here. The
    command defaults to a bare ``chai-lab`` so it resolves inside whatever
    environment is active; set ``CAPOS_CHAI1_CMD`` to call into a different
    one, e.g. the full path to the binary in a dedicated ``chai1`` env.
    """
    return os.environ.get("CAPOS_CHAI1_CMD", "chai-lab").split()


def _pick_best_cif(src_dir, cifs):
    """Chai-1 candidate with the highest pTM, or the first CIF as a fallback.

    Chai-1 writes ``scores.model_idx_<i>.npz`` next to each
    ``pred.model_idx_<i>.cif``. The research pipeline keeps the highest-pTM
    candidate; using the first file instead would feed a different structure
    to SaProt.
    """
    import numpy as np

    best = None
    for cif in cifs:
        idx = None
        for token in os.path.basename(cif).replace(".cif", "").split("."):
            if token.startswith("model_idx_"):
                idx = token.rsplit("_", 1)[-1]
        if idx is None:
            continue
        score_path = os.path.join(src_dir, "scores.model_idx_%s.npz" % idx)
        if not os.path.exists(score_path):
            continue
        try:
            with np.load(score_path) as d:
                ptm = float(np.asarray(d["ptm"]).ravel()[0])
                agg = (float(np.asarray(d["aggregate_score"]).ravel()[0])
                       if "aggregate_score" in d else float("-inf"))
        except Exception:
            continue
        key = (ptm, agg)
        if best is None or key > best[0]:
            best = (key, cif)
    if best is not None:
        return best[1]
    return next((c for c in cifs if "model_idx_0" in c), cifs[0])


def _cmd_fold(args):
    import subprocess

    repo = _repo_root()
    if repo is None:
        sys.exit("!! folding helpers not found: set CAPOS_REPO_ROOT to the "
                 "CAPOS source checkout (the full Docker image sets it)")
    # The folding helper lives under tools/.
    scripts = os.path.join(repo, "tools")
    prep = os.path.join(scripts, "fold_preparition.py")
    if not os.path.exists(prep):
        sys.exit("!! folding helpers not found at %s" % scripts)

    chai = _chai1_cmd()
    print("  Chai-1 command  : %s" % " ".join(chai))
    print("  (structure prediction is its own environment -- see "
          "environment_chai1.yml)")

    work = args.out_dir
    os.makedirs(work, exist_ok=True)
    fasta = os.path.join(work, "prepared_seq.fasta")

    subprocess.run([sys.executable, prep,
                    "--input", args.input, "--output", fasta], check=True)

    # Chai-1 is invoked the way the research pipeline does it: one fold
    # per sequence, into its own directory. Run sequentially by default;
    # raising --workers is the caller's decision because each fold wants
    # its own GPU and long chains run out of memory when they collide.
    out_root = os.path.join(work, "fold_result")
    os.makedirs(out_root, exist_ok=True)
    for name, seq in _iter_fasta(fasta):
        dest = os.path.join(out_root, name)
        os.makedirs(dest, exist_ok=True)
        one = os.path.join(work, "_one_%s.fasta" % name)
        with open(one, "w") as f:
            f.write(">%s\n%s\n" % (name, seq))
        try:
            subprocess.run(chai + ["fold", one, dest], check=True)
        except FileNotFoundError:
            sys.exit("!! Chai-1 not found (%s). Install it in its own "
                     "environment and point CAPOS_CHAI1_CMD at it -- see "
                     "environment_chai1.yml" % " ".join(chai))
        os.remove(one)

    return 0


def _cmd_run(args):
    import csv as _csv
    import shutil

    _preflight_run(args)

    work = args.work_dir
    os.makedirs(work, exist_ok=True)
    struct_dir = os.path.join(work, "structure")
    feat_dir = os.path.join(work, "features")

    fold_args = argparse.Namespace(
        input=args.fasta, out_dir=work, workers=args.workers)
    _cmd_fold(fold_args)

    # extract_* read a CSV with ID, Sequence -- not a FASTA. Build one.
    seq_csv = os.path.join(work, "queries.csv")
    rows = list(_iter_fasta(args.fasta))
    with open(seq_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["ID", "Sequence"])
        for name, seq in rows:
            w.writerow([name, seq])

    # fold writes one directory per sequence under fold_result/<ID>/;
    # extract_saprot.py reads <structure_dir>/<ID>.cif. Flatten the former
    # into the latter so the SaProt branch finds its structures.
    fold_root = os.path.join(work, "fold_result")
    os.makedirs(struct_dir, exist_ok=True)
    n_cif = 0
    missing = []
    for name, _ in rows:
        # fold_preparition.py rewrites every FASTA header to
        # ``>protein|name=<ID>``, so Chai-1 writes
        # fold_result/protein|name=<ID>/. Accept both spellings; the staged
        # file keeps the bare ID, because extract_saprot.py looks for
        # <structure_dir>/<ID>.cif.
        src_dir = next(
            (c for c in (os.path.join(fold_root, name),
                         os.path.join(fold_root, "protein|name=%s" % name))
             if os.path.isdir(c)),
            None)
        if src_dir is None:
            missing.append(name)
            continue
        cifs = [p for p in sorted(os.listdir(src_dir)) if p.endswith(".cif")]
        if not cifs:
            missing.append("%s (no .cif)" % name)
            continue
        # Keep the highest-pTM candidate, matching the research pipeline.
        pick = _pick_best_cif(src_dir, cifs)
        dst = os.path.join(struct_dir, "%s.cif" % name)
        if not os.path.exists(dst):
            shutil.copyfile(os.path.join(src_dir, pick), dst)
        n_cif += 1
    print("  structures staged: %d -> %s" % (n_cif, struct_dir))
    # Missing structures must stop the run here; continuing would give
    # SaProt nothing to read and fail much later with a vague message.
    if missing:
        sys.exit("!! no fold output for %d sequence(s): %s"
                 % (len(missing), ", ".join(missing[:10])))

    feats = argparse.Namespace(
        input=seq_csv, out_dir=feat_dir, structure_dir=struct_dir,
        esms_model_dir=os.environ.get("CAPOS_ESMS_MODEL"),
        esms_ckpt=os.environ.get("CAPOS_ESMS_CKPT"),
        saprot_model_path=os.environ.get("CAPOS_SAPROT_MODEL"),
        saprot_repo=os.environ.get("CAPOS_SAPROT_REPO"),
        foldseek_bin=os.environ.get("CAPOS_FOLDSEEK_BIN"),
        plddt_mask=args.plddt_mask, batch_size=16,
        prott5_model="")
    _cmd_featurize(feats)

    weights = args.weights or _find_checkpoint("capos_model.pth")
    if weights is None:
        sys.exit("!! --weights not given and no bundled checkpoint found")

    out = os.path.join(work, "capos_predictions.csv")
    pred_args = argparse.Namespace(
        seq_feat=os.path.join(feat_dir, "esms.h5"),
        struct_feat=os.path.join(feat_dir, "saprot.h5"),
        weights=weights, model=args.model,
        sequences=args.fasta, out=out, batch_size=args.batch_size)
    _cmd_predict(pred_args)

    return 0


def _cmd_check(args):
    """Report the install state and what each subcommand would need."""
    from . import __version__

    print("capos %s" % __version__)
    print("python %s" % sys.version.split()[0])

    try:
        import numpy, pandas, h5py, torch
        print("torch %s   numpy %s   pandas %s   h5py %s"
              % (torch.__version__, numpy.__version__,
                 pandas.__version__, h5py.__version__))
        print("device: %s" % ("cuda (%s)" % torch.cuda.get_device_name(0)
                              if torch.cuda.is_available() else "cpu"))
    except Exception as e:
        print("!! light-tier dependency missing: %s" % e)

    print()
    print("bundled checkpoints")
    for name in ("capos_model.pth", "capos-seq_model.pth"):
        p = _find_checkpoint(name)
        print("  %-22s %s" % (name, "%s (%.1f MB)" % (p, os.path.getsize(p) / 1e6)
                              if p else "MISSING"))

    print()
    print("full tier (only needed for `capos fold` / `featurize` / `run`)")
    repo = _repo_root()
    print("  %-20s %-8s %s" % ("source tree", "ok" if repo else "unset",
                               repo or "set CAPOS_REPO_ROOT"))
    for label, key in (("chai-lab", "CAPOS_CHAI1_CMD"),
                       ("ESM-S model dir", "CAPOS_ESMS_MODEL"),
                       ("ESM-S checkpoint", "CAPOS_ESMS_CKPT"),
                       ("SaProt model", "CAPOS_SAPROT_MODEL"),
                       ("SaProt repo", "CAPOS_SAPROT_REPO"),
                       ("ProtT5 model", "CAPOS_PROTT5_MODEL"),
                       ("foldseek binary", "CAPOS_FOLDSEEK_BIN")):
        v = os.environ.get(key, "")
        if not v and key == "CAPOS_CHAI1_CMD":
            # bare `chai-lab` on PATH is a valid setting
            if shutil_which("chai-lab"):
                v = "chai-lab (bare, on PATH)"
                mark = "ok"
            else:
                mark = "unset"
        elif not v:
            mark = "unset"
        elif key == "CAPOS_CHAI1_CMD" and not v.startswith("/"):
            mark = "ok"
        elif (key == "CAPOS_PROTT5_MODEL" and not v.startswith("/")
              and not os.path.exists(v)):
            # ProtT5 may be a HuggingFace model name rather than a local path.
            mark = "name"
        else:
            mark = "ok" if os.path.exists(v) else "MISSING"
        print("  %-20s %-8s %s" % (label, mark, v or "--"))

    print()
    print("training (only needed for `capos train`)")
    for mod in ("sklearn", "matplotlib", "seaborn"):
        try:
            __import__(mod)
            mark, detail = "ok", ""
        except Exception as e:
            mark, detail = "MISSING", str(e)
        print("  %-20s %-8s %s" % (mod, mark, detail))

    print()
    print("what you can run now")
    print("  capos predict   precomputed features -> predictions   (CPU)")
    print("  capos train     features + labels -> checkpoint       (CPU/GPU)")
    print("  capos fold      FASTA -> monomer structures           (GPU)")
    print("  capos featurize sequences/structures -> features       (GPU)")
    print("  capos run       fold + featurize + predict             (GPU)")
    print()
    print("Start with `cd quickstart && ./run_quickstart.sh` -- it needs nothing but "
          "the light tier.")
    return 0


def _preflight_run(args):
    """Fail before any GPU work if the environment is incomplete."""
    problems = []

    fasta = args.fasta
    if not os.path.exists(fasta):
        problems.append("input FASTA not found: %s" % fasta)
    weights = args.weights or _find_checkpoint("capos_model.pth")
    if weights is None:
        problems.append("--weights not given and no checkpoint found "
                        "(pass --weights or set CAPOS_WEIGHTS)")
    elif not os.path.exists(weights):
        problems.append("checkpoint not found: %s" % weights)
    else:
        args.weights = weights

    if not os.environ.get("CAPOS_CHAI1_CMD") and not shutil_which("chai-lab"):
        problems.append("Chai-1 not found: set CAPOS_CHAI1_CMD to the chai-lab "
                        "binary (see environment_chai1.yml)")

    repo = _repo_root()
    if repo is None:
        problems.append("source tree not found: set CAPOS_REPO_ROOT to the "
                        "CAPOS checkout (the full Docker image sets it)")
    elif not os.path.isfile(os.path.join(
            repo, "tools", "fold_preparition.py")):
        problems.append("folding helper not found under tools/")

    for key, label in (("CAPOS_ESMS_MODEL", "ESM-S model dir"),
                       ("CAPOS_ESMS_CKPT", "ESM-S checkpoint"),
                       ("CAPOS_SAPROT_MODEL", "SaProt model"),
                       ("CAPOS_SAPROT_REPO", "SaProt repo"),
                       ("CAPOS_FOLDSEEK_BIN", "foldseek binary")):
        v = os.environ.get(key)
        if not v:
            problems.append("%s not set ($%s)" % (label, key))
        elif not os.path.exists(v):
            problems.append("%s not found: %s" % (label, v))

    if problems:
        print("!! `capos run` cannot start yet:")
        for p in problems:
            print("     - %s" % p)
        print()
        print("   Run `capos check` for a full report.")
        sys.exit(1)
    return 0


def shutil_which(cmd):
    from shutil import which
    return which(cmd)


def _iter_fasta(path):
    name, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    yield name, "".join(buf)
                name = line[1:].split()[0]
                buf = []
            elif line:
                buf.append(line)
        if name:
            yield name, "".join(buf)


if __name__ == "__main__":
    raise SystemExit(main())
