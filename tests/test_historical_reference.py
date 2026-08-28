from pathlib import Path
import json
import pandas as pd

REPO=Path(__file__).resolve().parents[1]

def test_historical_reference_capture():
    facts=REPO/"reference_current_results/training/HISTORICAL_REFERENCE_FACTS.json"
    hist=REPO/"reference_current_results/training/checkpoint_fullmodel_history.json"
    assert facts.exists(), "Run scripts/03_capture_historical_reference.py from the Phase-B package first."
    assert hist.exists()
    f=json.loads(facts.read_text())
    assert f["executed_epochs"]==50
    assert f["last_epoch"]==50
    assert f["best_epoch_by_val_macro_auprc"]==44
    assert abs(f["best_val_macro_auprc"]-0.9777503420277581)<1e-12
    assert f["scheduler_total_epochs"]==50
    assert f["lr_plot_rows_configured_n_epochs"]==100

def test_historical_best_val_matches_reference_table():
    facts=json.loads((REPO/"reference_current_results/training/HISTORICAL_REFERENCE_FACTS.json").read_text())
    val=pd.read_csv(REPO/"reference_current_results/metrics/validation_metrics_overall.csv").iloc[0]
    assert abs(float(val["macro_auprc"])-facts["best_val_macro_auprc"])<1e-12
