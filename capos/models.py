#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAPOS model definitions.

The layer order, dropout rates and forward pass are the published
architecture, reproduced here so the package is self-contained.

    CAPOSModel
        seq_mlp     Linear(d -> 256) + GELU + Dropout(0.3)
        struct_mlp  Linear(d -> 256) + GELU + Dropout(0.3)
        gate_mlp    Linear(512 -> 256) + Sigmoid
        x           gate * seq + (1 - gate) * struct
        layer_norm  LayerNorm(256)
        classifier  Linear(256 -> 128) + GELU + Dropout(0.2) + Linear(128 -> K)

    CAPOSSeqModel
        BiGRU(128) + Attention + Dropout(0.1) + Linear

``CAPOSModel`` takes the two branch widths separately (``seq_input_dim`` and
``struct_input_dim``) because the sequence and structure encoders do not have
to agree: ESM-S and SaProt are 1280 each, ProtT5 is 1024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 9
# Row order of the classifier output, and of every prediction CSV this package
# writes. Fixed by the training pipeline; do not reorder.
LABEL_NAMES = ["1", "2", "3", "4", "5", "6", "8", "10", "12"]
LABEL_MAP = {int(c): i for i, c in enumerate(LABEL_NAMES)}
INVERSE_LABEL_MAP = {i: int(c) for i, c in enumerate(LABEL_NAMES)}


class CAPOSModel(nn.Module):
    def __init__(self, seq_input_dim, struct_input_dim, num_classes=NUM_CLASSES,
                 vector_dim=256):
        super(CAPOSModel, self).__init__()
        self.vector_dim = vector_dim

        self.seq_mlp = nn.Sequential(
            nn.Linear(seq_input_dim, vector_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.struct_mlp = nn.Sequential(
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

    def forward(self, input_seq, input_struct):
        input_seq = input_seq.view(input_seq.size(0), -1)
        input_struct = input_struct.view(input_struct.size(0), -1)

        seq_feat = self.seq_mlp(input_seq)
        struct_feat = self.struct_mlp(input_struct)

        combined_feat = torch.cat([seq_feat, struct_feat], dim=1)
        gate = self.gate_mlp(combined_feat)

        x_fused = gate * seq_feat + (1.0 - gate) * struct_feat

        x_fused = self.layer_norm(x_fused)

        return self.classifier(x_fused)


class Attention(nn.Module):
    def __init__(self, seq_input_dim, attention_size=32):
        super(Attention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(seq_input_dim, attention_size))
        self.b = nn.Parameter(torch.Tensor(attention_size))
        self.u = nn.Parameter(torch.Tensor(attention_size, 1))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)
        nn.init.xavier_uniform_(self.u)

    def forward(self, x):
        e = torch.tanh(torch.matmul(x, self.W) + self.b)

        scores = torch.matmul(e, self.u)

        alpha = F.softmax(scores, dim=1)

        output = torch.sum(x * alpha, dim=1)
        return output


class CAPOSSeqModel(nn.Module):
    def __init__(self, input_dim, num_classes=NUM_CLASSES, gru_hidden=128,
                 attention_size=32):
        super(CAPOSSeqModel, self).__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        gru_output_dim = gru_hidden * 2

        self.attention = Attention(gru_output_dim, attention_size)

        self.dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(gru_output_dim, num_classes)

    def forward(self, x, input_struct=None):

        gru_out, _ = self.gru(x)

        attn_out = self.attention(gru_out)

        out = self.dropout(attn_out)

        logits = self.fc(out)

        return logits
