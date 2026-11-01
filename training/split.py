"""Leakage-safe data splitting (object / mould / duplicate-group level; see admin split planner for institution- and culture-level holdouts).

Splits are made at the OBJECT / MOULD-GROUP level, never at the image level:
all photographs of one vessel — and all vessels believed to share a mould —
land in the same fold.  Optionally also stratified by museum so a model
cannot learn a photographer's lighting as a proxy for the label.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def group_key(rec: dict) -> str:
    """Grouping identity for a record: mould group > object > id."""
    return str(rec.get("mold_group") or rec.get("object_id") or rec.get("object_number") or rec["id"])


def grouped_split(records: list[dict], test_frac: float = 0.2, val_frac: float = 0.1, seed: int = 42,
                  label_key: str = "expression") -> dict[str, list[dict]]:
    """Deterministic group-aware split, roughly label-stratified."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[group_key(r)].append(r)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    # stratify by the majority label of each group
    by_label: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        labels = [r.get(label_key) for r in groups[k] if r.get(label_key)]
        maj = max(set(labels), key=labels.count) if labels else "__none__"
        by_label[maj].append(k)

    out = {"train": [], "val": [], "test": []}
    for lab, ks in by_label.items():
        n = len(ks)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        for i, k in enumerate(ks):
            part = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
            out[part].extend(groups[k])
    return out


def grouped_kfold(records: list[dict], k: int = 5, seed: int = 42):
    """Yield (train, test) lists with group integrity for cross-validation."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[group_key(r)].append(r)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    folds = [keys[i::k] for i in range(k)]
    for i in range(k):
        test_keys = set(folds[i])
        test = [r for key in folds[i] for r in groups[key]]
        train = [r for key in keys if key not in test_keys for r in groups[key]]
        yield train, test


def assert_no_leakage(train: list[dict], test: list[dict]):
    a = {group_key(r) for r in train}
    b = {group_key(r) for r in test}
    leak = a & b
    if leak:
        raise AssertionError(f"Group leakage between train and test: {sorted(leak)[:10]}")
    # also warn on identical image hashes
    ha = {r.get("sha256") for r in train if r.get("sha256")}
    hb = {r.get("sha256") for r in test if r.get("sha256")}
    if ha & hb:
        raise AssertionError("Identical image hashes in train and test")
