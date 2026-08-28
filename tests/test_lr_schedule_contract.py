from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"src/vista"))
import optim

def test_lr_contract():
    cfg=optim.LRScheduleConfig()
    assert cfg.warmup_epochs==10
    assert cfg.total_epochs==50
    assert abs(cfg.min_lr_ratio-0.01)<1e-12
    assert abs(optim.schedule_factor(1,cfg)-0.1)<1e-12
    assert abs(optim.schedule_factor(10,cfg)-1.0)<1e-12
    assert abs(optim.schedule_factor(50,cfg)-0.01)<1e-12
    s=optim.compute_lr_schedule(50,1e-4,0.1,cfg)
    for h,b in zip(s["head_lr"],s["backbone_lr"]):
        assert abs((h/b)-10.0)<1e-8
