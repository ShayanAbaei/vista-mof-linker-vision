#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",default="runs/epfl/e0_vista_sample_seed42")
    ap.add_argument("--approver",default="Shayan")
    ap.add_argument("--note",required=True)
    args=ap.parse_args()
    run=Path(args.run_dir)
    gate=json.loads((run/"E0_GATE.json").read_text())
    if gate.get("numeric_gate")!="PASS":
        raise RuntimeError("Cannot approve: E0 numeric gate is not PASS")
    d=Path("approvals");d.mkdir(exist_ok=True)
    payload={
        "approved_by":args.approver,
        "approved_at_utc":datetime.now(timezone.utc).isoformat(),
        "note":args.note,
        "e0_gate":str(run/"E0_GATE.json"),
    }
    (d/"E0_APPROVED.txt").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print("E0 approval recorded. E1/E2/E3 launch guards are now satisfied.")
if __name__=="__main__":
    main()
