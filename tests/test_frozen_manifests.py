from pathlib import Path
import json
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
LABELS=[
"hydrophobicity","polarity","flexibility","aromaticity","h_bond_donors",
"h_bond_acceptors","planarity_single_ring","macrocycle_gt6_rings",
"charge_present","conjugation_present"]

def test_phase_a_report_pass():
    p=json.loads((REPO/"data/frozen/PHASE_A_VALIDATION_REPORT.json").read_text())
    assert p["status"]=="PASS"
    assert p["strict_cross_split_canonical_groups"]==0

def test_frozen_manifests():
    for name,strict in [("current_sample_split.csv",False),("canonical_group_disjoint_split.csv",True)]:
        df=pd.read_csv(REPO/"data/frozen"/name)
        assert len(df)==9414
        assert df["filename"].is_unique
        assert df["split"].value_counts().to_dict()=={"train":6590,"test":1412,"validation":1412}
        for c in LABELS:
            assert c in df
            assert set(df[c].unique()).issubset({0,1})
        if strict:
            assert df.groupby("canonical_first_linker_smiles")["split"].nunique().max()==1
