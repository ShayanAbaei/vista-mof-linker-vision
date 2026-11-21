#!/usr/bin/env python3
"""Read-only source-evidence checks for VISTA; no model inference or GPU training.

Uses only Python's standard library to confirm selected frozen manifest and
metric contracts. It is not a reconstruction of historical model weights.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    file = ROOT / path
    if not file.is_file():
        raise AssertionError(f"Missing required source: {path}")
    with file.open(newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def status(path):
    file = ROOT / path
    if not file.is_file():
        raise AssertionError(f"Missing report: {path}")
    obj = json.loads(file.read_text(encoding="utf-8"))
    assert obj.get("status") == "PASS", (path, obj.get("status"))


def near(actual, expected, tol=1e-8):
    assert abs(float(actual)-expected) <= tol, (actual, expected)


def main():
    status("data/frozen/PHASE_A_VALIDATION_REPORT.json")
    status("phase_c_results/report/PHASE_C_STATUS.json")

    counts = []
    seen = []
    for path in ("data/frozen/current_sample_split.csv",
                 "data/frozen/canonical_group_disjoint_split.csv"):
        counter = Counter()
        groups = defaultdict(set)
        n = 0
        for row in read_csv(path):
            n += 1
            counter[row["split"]] += 1
            groups[row["canonical_first_linker_smiles"]].add(row["split"])
        assert n == 9414, (path, n)
        assert counter == Counter(train=6590, validation=1412, test=1412), (path,counter)
        assert len(groups) == 5727, (path, len(groups))
        counts.append(counter)
        seen.append(sum(len(s) > 1 for s in groups.values()))
    assert seen == [598, 0], seen

    vista = list(read_csv("reference_current_results/metrics/test_metrics_overall.csv"))
    assert len(vista) == 1
    near(vista[0]["macro_auprc"], 0.9879242367586224)
    near(vista[0]["macro_f1"], 0.9604032151282652)
    near(vista[0]["macro_auroc"], 0.9971890915473081)

    baselines = list(read_csv("phase_c_results/metrics/baseline_metrics_overall.csv"))
    for split, model, auprc in (
        ("current", "morgan_logreg", 0.9168851789005575),
        ("current", "morgan_rf", 0.9099902689837649),
        ("current", "rdkit_descriptor_lro", 0.9406953598067348),
        ("strict", "morgan_logreg", 0.9257664765719819),
        ("strict", "morgan_rf", 0.8758097030672329),
        ("strict", "rdkit_descriptor_lro", 0.9204987787708919),
    ):
        matched = [x for x in baselines if x["split_scheme"] == split
                   and x["model"] == model and x["evaluation_split"] == "test"]
        assert len(matched) == 1, (split,model,len(matched))
        near(matched[0]["macro_auprc"], auprc)

    similarity = {x["split_scheme"]:x for x in read_csv("phase_c_results/metrics/tanimoto_summary.csv")}
    assert int(similarity["current"]["exact_canonical_matches"]) == 630
    assert int(similarity["strict"]["exact_canonical_matches"]) == 0
    near(similarity["current"]["median"], 0.84)
    near(similarity["strict"]["median"], 0.7022727272727273)

    bootstrap = list(read_csv("phase_c_results/bootstrap/historical_vista_current_cluster_bootstrap_summary.csv"))
    assert bootstrap and all(int(row["n_resamples"]) == 2000 for row in bootstrap)

    for path in (
        "reference_current_results/predictions/validation_predictions.csv",
        "reference_current_results/predictions/test_predictions.csv",
        "reference_current_results/metrics/thresholds.csv",
        "phase_c_results/source_data/strict__morgan_logreg__test_predictions.csv",
        "phase_c_results/source_data/strict__morgan_rf__test_predictions.csv",
        "phase_c_results/source_data/strict__rdkit_descriptor_lro__test_predictions.csv",
    ):
        assert (ROOT / path).is_file(), f"Missing archived source: {path}"

    print("PASS — frozen manifests, historical VISTA metrics and completed molecular baselines.")
    print("SCOPE — source-integrity check only; no checkpoint recovery or new GPU training.")


if __name__ == "__main__":
    main()
