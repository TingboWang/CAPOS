# -*- coding: utf-8 -*-
"""Training/evaluation loop and the validation report.

The report layout matches the research pipeline so a checkpoint trained here
can be compared with the published runs.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, matthews_corrcoef,
                             precision_recall_curve)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs_seq, inputs_struct, labels in dataloader:
        inputs_seq = inputs_seq.to(device)
        inputs_struct = inputs_struct.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs_seq, inputs_struct)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs_seq.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for inputs_seq, inputs_struct, labels in dataloader:
            inputs_seq = inputs_seq.to(device)
            inputs_struct = inputs_struct.to(device)
            labels = labels.to(device)

            outputs = model(inputs_seq, inputs_struct)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            test_loss += loss.item() * inputs_seq.size(0)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_probs.append(probs.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return (test_loss / test_total, test_correct / test_total,
            np.concatenate(all_probs), np.concatenate(all_preds),
            np.concatenate(all_labels))


def macro_metrics(y_true, y_pred, num_classes):
    """Macro-F1 and macro one-vs-rest MCC, the metric used in the paper."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    f1 = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1e-12)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(denom > 0, (tp * tn - fp * fn) / np.maximum(denom, 1e-12), 0.0)
    return float(f1.mean()), float(mcc.mean())


def save_metrics(test_probs, test_preds, test_labels, test_ids, num_classes,
                 label_names_display, label_map, output_dir, split_name="Validation"):
    os.makedirs(output_dir, exist_ok=True)
    valid_labels = list(range(num_classes))

    cm = confusion_matrix(test_labels, test_preds, labels=valid_labels,
                          normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_names_display,
                yticklabels=label_names_display)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (%s Set)" % split_name)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.pdf"),
                bbox_inches="tight")
    plt.close()

    reverse_label_map = {v: k for k, v in label_map.items()}
    df_results = pd.DataFrame(
        test_probs[:, valid_labels],
        columns=["Prob_Class_%s" % cls for cls in label_names_display])
    df_results.insert(0, "ID", test_ids)
    df_results["True_Label"] = [reverse_label_map.get(lbl, -1) for lbl in test_labels]
    df_results["Pred_Label"] = [reverse_label_map.get(lbl, -1) for lbl in test_preds]
    df_results.to_csv(os.path.join(output_dir, "prediction_results.csv"), index=False)

    y_true_onehot = np.eye(num_classes)[test_labels]
    auc_pr_per_class = {}
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        if np.sum(y_true_onehot[:, i]) > 0:
            auc = average_precision_score(y_true_onehot[:, i], test_probs[:, i])
            auc_pr_per_class[i] = auc
            precision_curve, recall_curve, _ = precision_recall_curve(
                y_true_onehot[:, i], test_probs[:, i])
            plt.plot(recall_curve, precision_curve,
                     label="Class %s (AUC=%.2f)" % (label_names_display[i], auc))
        else:
            auc_pr_per_class[i] = 0.0
    plt.title("Precision-Recall Curves (%s Set)" % split_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, "PR_curves_per_class.pdf"))
    plt.close()

    mcc_per_class = {
        i: matthews_corrcoef((test_labels == i).astype(int),
                             (test_preds == i).astype(int))
        for i in range(num_classes)
    }
    report_dict = classification_report(test_labels, test_preds,
                                        output_dict=True, labels=valid_labels)
    accuracy_score = report_dict.pop("accuracy")
    df_report = pd.DataFrame(report_dict).transpose()
    df_report = df_report.rename(columns={
        "precision": "Precision", "recall": "Recall",
        "f1-score": "F1-score", "support": "Support"})
    df_report["AUC-PR"] = None
    df_report["MCC"] = None
    for i in range(num_classes):
        df_report.loc[str(i), "AUC-PR"] = auc_pr_per_class[i]
        df_report.loc[str(i), "MCC"] = mcc_per_class[i]
    df_report.loc["macro avg", "AUC-PR"] = np.mean(list(auc_pr_per_class.values()))
    df_report.loc["macro avg", "MCC"] = np.mean(list(mcc_per_class.values()))
    df_report.loc["macro avg", "Support"] = np.nan

    df_classes = df_report.drop(["macro avg", "weighted avg"])
    df_macro = df_report.loc[["macro avg"]].rename(index={"macro avg": "Macro avg"})
    df_accuracy = pd.DataFrame({
        "Precision": [accuracy_score], "Recall": [np.nan],
        "F1-score": [np.nan], "AUC-PR": [np.nan], "MCC": [np.nan],
        "Support": [np.nan]}, index=["Accuracy"])
    df_report = pd.concat([df_classes, df_macro, df_accuracy])

    idx_map = {str(i): label_names_display[i] for i in range(num_classes)}
    df_report = df_report.rename(index=idx_map)
    for col in ("Precision", "Recall", "F1-score", "AUC-PR", "MCC", "Support"):
        df_report[col] = pd.to_numeric(df_report[col], errors="coerce")
    df_report = df_report[
        ["Precision", "Recall", "F1-score", "AUC-PR", "MCC", "Support"]].round(4)
    df_report.to_csv(os.path.join(output_dir, "classification_report.csv"),
                     encoding="utf-8-sig", na_rep="")
