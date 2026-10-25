#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAPOS -- Context-Aware Protein Oligomeric State prediction.

Predicts homo-oligomeric stoichiometry (1, 2, 3, 4, 5, 6, 8, 10, 12-mer) from
a protein sequence, by fusing a sequence protein-language-model embedding
(ESM-S) with a structure-aware embedding (SaProt / 3Di) through a learned
gate.

See ``__main__`` / ``capos --help`` for the command line.
"""

__version__ = "1.0.0"

from .models import (LABEL_MAP, LABEL_NAMES, NUM_CLASSES,  # noqa: F401
                     CAPOSModel, CAPOSSeqModel)
