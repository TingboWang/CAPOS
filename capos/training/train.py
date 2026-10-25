# -*- coding: utf-8 -*-
"""Train CAPOS (ESM-S + SaProt) or CAPOS-seq (ProtT5) from feature files."""

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models import (CAPOSModel, CAPOSSeqModel, LABEL_MAP, LABEL_NAMES,
                      NUM_CLASSES)
from .data import OligomerDataset, align_data
from .engine import evaluate, macro_metrics, save_metrics, train_one_epoch


def set_seed(seed):
    """Pin Python/NumPy/torch RNGs and cuDNN for a reproducible run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(args):
    if args.epochs < 1:
        raise SystemExit("!! --epochs must be at least 1")
    if args.model == "capos" and (not args.train_struct or not args.val_struct):
        raise SystemExit(
            "!! --model capos requires --train-struct and --val-struct")
    if args.lr is None:
        args.lr = 1e-3 if args.model == "capos-seq" else 1e-4

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("!! --device cuda requested but CUDA is not available")

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    print("Device: %s | model: %s | seed: %d" % (device, args.model, args.seed))

    train_struct = args.train_struct if args.model == "capos" else None
    val_struct = args.val_struct if args.model == "capos" else None
    X_train_seq, X_train_struct, y_train, _ = align_data(
        args.train_seq, train_struct, LABEL_MAP, mode="Train")
    X_val_seq, X_val_struct, y_val, val_ids = align_data(
        args.val_seq, val_struct, LABEL_MAP, mode="Validation")

    X_train_seq = X_train_seq.reshape(X_train_seq.shape[0], 1, -1)
    X_val_seq = X_val_seq.reshape(X_val_seq.shape[0], 1, -1)
    if X_train_struct is not None:
        X_train_struct = X_train_struct.reshape(X_train_struct.shape[0], 1, -1)
        X_val_struct = X_val_struct.reshape(X_val_struct.shape[0], 1, -1)

    train_loader = DataLoader(
        OligomerDataset(X_train_seq, X_train_struct, y_train),
        batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        OligomerDataset(X_val_seq, X_val_struct, y_val),
        batch_size=args.batch_size, shuffle=False)

    seq_input_dim = X_train_seq.shape[-1]
    if args.model == "capos-seq":
        model = CAPOSSeqModel(seq_input_dim, num_classes=NUM_CLASSES)
    else:
        model = CAPOSModel(seq_input_dim=seq_input_dim,
                           struct_input_dim=X_train_struct.shape[-1],
                           num_classes=NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = []
    print("Training on %d samples, validating on %d, %d epochs"
          % (len(train_loader.dataset), len(val_loader.dataset), args.epochs))
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_probs, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device)
        val_f1, val_mcc = macro_metrics(val_labels, val_preds, NUM_CLASSES)
        print("Epoch [%d/%d] Train Loss: %.4f Acc: %.4f | "
              "Val Loss: %.4f Acc: %.4f F1: %.4f MCC: %.4f"
              % (epoch + 1, args.epochs, train_loss, train_acc,
                 val_loss, val_acc, val_f1, val_mcc))
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "train_accuracy": train_acc, "val_loss": val_loss,
            "val_accuracy": val_acc, "val_macro_f1": val_f1,
            "val_macro_mcc": val_mcc})

    checkpoint = os.path.join(args.out_dir, "%s_model.pth" % args.model)
    torch.save(model.state_dict(), checkpoint)
    print("Model saved to: %s" % checkpoint)

    pd.DataFrame(history).to_csv(
        os.path.join(args.out_dir, "history.csv"), index=False)
    save_metrics(val_probs, val_preds, val_labels, val_ids, NUM_CLASSES,
                 LABEL_NAMES, LABEL_MAP, args.out_dir,
                 split_name="Validation")
    print("Validation report and predictions saved under: %s" % args.out_dir)
    return {"n_train": len(train_loader.dataset),
            "n_val": len(val_loader.dataset), "checkpoint": checkpoint}
