"""
Generate notebooks/molecular_property_attention_analysis.ipynb.

The notebook is an explicit, top-to-bottom research workflow. Each conceptual step
has a short Markdown explainer followed by a single code cell that calls the
reusable project modules. Heavy steps are guarded by config flags and degrade
gracefully when the dataset/checkpoint is absent, so the notebook runs anywhere.

This build reflects the current design:
  * the label self-attention diagonal mask was REMOVED (it hurt performance); the
    default model is unmasked, and there is no mask-comparison experiment,
  * the Bayesian attention-graph was REMOVED as redundant; label-to-label analysis
    is summarised by publication-quality heatmaps (mean attention, conditional
    preference, and the imbalance-corrected directed effect),
  * the directed edge mask survives only for the directed-communication ablation,
  * dominant-supervision and directed-communication ablations each run SIX cases
    (full model + remove each dominant + remove first two + remove first three),
  * label-to-image PDFs skip the (time-consuming) train split,
  * an OPTIONAL loss-function ablation (CE / weighted CE) is appended last.

Run:  python build_notebook.py
"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


def step(n, title, body_md, code_text=None):
    md(f"### {n}. {title}\n\n{body_md.strip()}")
    if code_text is not None:
        code(code_text)


# ============================================================== title / hypotheses
md(r"""
# Molecular-property attention analysis

**Objective.** Predict ten non-mutually-exclusive binary molecular properties from
2D molecular images with a Swin-B backbone + ten learned label tokens + a 6-layer,
12-head Transformer, and test whether **representations of common (dominant)
properties provide useful information to rare-property tokens**, improving rare
detection.

The model is **unmasked**: each label token attends freely to the other nine label
tokens and all 49 image tokens (the label self-attention diagonal mask was tested
and removed because it hurt performance). The remaining controlled intervention is
to **block specific directed label->label edges** during training/evaluation, used
only in the directed-communication ablation.
""")

md(r"""
## Hypotheses and non-causal interpretation

- **H1.** Label tokens route image evidence selectively (label->image attention is
  structured, not uniform).
- **H2.** Rare-label tokens attend to dominant-label tokens more when both
  properties co-occur (imbalance-corrected directed effect `delta[i,j] > 0`).
- **H3.** Removing dominant-label **supervision** or **communication** (controlled
  retraining) degrades rare-label test AUPRC.

**Interpretation guardrails.** Attention is **information routing**, not causation.
We report *attention-derived information flow*, *probabilistic evidence of directed
token interaction*, and *performance effects under controlled model interventions*.
We never claim property A chemically causes property B.
""")

# =========================================================== Google Colab setup
md(r"""
## Google Colab setup (run this first on Colab; it is a no-op locally)

**Use this when your images are in a single ZIP and each split has its own CSV.**

1. Upload to a Google Drive folder (e.g. `MyDrive/MOF/`):
   - the project's `.py` modules (`ml_vit.py`, `masking.py`, `attention*.py`,
     `label_label_plots.py`, `experiments.py`, `baselines.py`, `evaluation.py`,
     `optim.py`, `train.py`, `data.py`, ...),
   - the single **images ZIP** (all PNGs),
   - the three CSVs `train.csv`, `validation.csv`, `test.csv`
     (each has a `filename` column matching the image names, plus the 10 label columns).
2. Edit the paths in the code cell below and run it.
3. Then run the rest of the notebook top-to-bottom.
""")
code(r"""
# Run FIRST on Colab. No-op on local machines (uses the local data/ dir instead).
import sys, os, zipfile, glob
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
PROJECT_DIR = globals().get("PROJECT_DIR")
IMAGE_ROOT = globals().get("IMAGE_ROOT")
CSV_OVERRIDES = globals().get("CSV_OVERRIDES", {})
OUTPUT_DIR = globals().get("OUTPUT_DIR")

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

    # ===================== EDIT THESE to your Drive layout =====================
    PROJECT_DIR  = "/content/drive/MyDrive/MOF"
    IMAGES_ZIP   = "/content/drive/MyDrive/MOF/images.zip"
    CSV_OVERRIDES = {
        "train":      "/content/drive/MyDrive/MOF/train.csv",
        "validation": "/content/drive/MyDrive/MOF/validation.csv",
        "test":       "/content/drive/MyDrive/MOF/test.csv",
    }
    OUTPUT_DIR   = "/content/drive/MyDrive/MOF/outputs"
    EXTRACT_TO   = "/content/mof_images"
    # ===========================================================================

    if PROJECT_DIR and PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    try:
        import networkx  # noqa: F401
    except ImportError:
        os.system("pip -q install networkx")

    os.makedirs(EXTRACT_TO, exist_ok=True)
    if not glob.glob(os.path.join(EXTRACT_TO, "**", "*.png"), recursive=True):
        print("extracting", IMAGES_ZIP, "...")
        with zipfile.ZipFile(IMAGES_ZIP) as z:
            z.extractall(EXTRACT_TO)
    pngs = glob.glob(os.path.join(EXTRACT_TO, "**", "*.png"), recursive=True)
    assert pngs, f"No .png files found under {EXTRACT_TO} after extracting {IMAGES_ZIP}"
    parents = sorted({str(Path(p).parent) for p in pngs})
    if len(parents) == 1:
        IMAGE_ROOT = parents[0]
    else:
        IMAGE_ROOT = os.path.commonpath(pngs)
        print(f"NOTE: images span {len(parents)} subfolders; CSV 'filename' must be relative to {IMAGE_ROOT}")
    ok = Path(PROJECT_DIR, "ml_vit.py").exists()
    print(f"Colab ready | {len(pngs)} images | IMAGE_ROOT={IMAGE_ROOT}")
    print("project modules:", "FOUND" if ok else f"NOT FOUND in {PROJECT_DIR} (upload the .py files there)")
else:
    print("Not on Colab -> the config cell will use the local data/ directory.")
""")

# ============================================================== 3 imports/paths
step(3, "Imports and project paths", """
**Purpose.** Import the reusable project modules and define the output tree.
On Colab this honours `PROJECT_DIR`/`OUTPUT_DIR` from the setup cell.
""", r"""
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch

ROOT = Path(globals().get("PROJECT_DIR") or
            (Path.cwd() if (Path.cwd() / "ml_vit.py").exists() else Path.cwd().parent))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import seeding, plotting, label_groups, evaluation, inference
import attention as attn_utils
import attention_label_image as ali
import attention_label_label as all_
import label_label_plots as llp
import experiments, baselines, compare_plots, checkpointing, optim
from masking import AttentionMaskConfig
from data import image_augmentation, MultiLabelMoleculeDataset

OUT = Path(globals().get("OUTPUT_DIR") or (ROOT / "outputs"))
DIRS = {k: OUT / k for k in [
    "configs", "checkpoints", "predictions", "metrics", "figures",
    "attention_label_image", "attention_label_label",
    "ablations", "baselines", "reports"]}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)
print("project root:", ROOT, "| outputs:", OUT)
""")

# ============================================================== 4 reproducibility
step(4, "Reproducibility and random seeds", """
**Method.** Seed Python/NumPy/PyTorch(+CUDA), DataLoader workers and the Bayesian
bootstrap RNG. **Output.** `SEEDS` dict stored in checkpoints/manifests.
""", r"""
SEED = 1234
SEEDS = seeding.set_all_seeds(SEED, deterministic_torch=True)
GLOBAL_RNG = seeding.make_rng(SEED)
WORKER_INIT = seeding.seed_worker
GENERATOR = seeding.make_generator(SEED)
print("seeds:", SEEDS)
""")

# ============================================================== 5 config
step(5, "Central experiment configuration", """
**Purpose.** One place for paths, label order, thresholds and the long-running
guard flags. The model is unmasked. Flags default to **not** retraining and **not**
generating every PDF.
""", r"""
# ---- data paths (Colab globals win; else local data/ dir) ----
DATA_DIR = ROOT / "data"
_IMAGE_ROOT = globals().get("IMAGE_ROOT")
_CSV_OVERRIDES = globals().get("CSV_OVERRIDES") or {}
if _IMAGE_ROOT is not None:
    IMAGE_DIRS = {s: Path(_IMAGE_ROOT) for s in ("train", "validation", "test")}
else:
    IMAGE_DIRS = {"train": DATA_DIR / "train", "validation": DATA_DIR / "validation", "test": DATA_DIR / "test"}
if _CSV_OVERRIDES:
    CSV_PATHS = {s: Path(_CSV_OVERRIDES[s]) for s in ("train", "validation", "test")}
else:
    CSV_PATHS = {s: DATA_DIR / f"{s}.csv" for s in ("train", "validation", "test")}
CHECKPOINT_PATH = DIRS["checkpoints"] / "best_model.pth"

# ---- label order (authoritative; verified against checkpoint in step 7) ----
LABEL_NAMES = [
    "Hydrophobicity", "Polarity", "Flexibility", "Aromaticity",
    "Hydrogen-Bond Donors", "Hydrogen-Bond Acceptors",
    "Planarity (Single-Ring)", "Macrocycle (>6 Rings)",
    "Charge Present", "Conjugation Present",
]
N_LABELS = len(LABEL_NAMES)

# ---- model / analysis (UNMASKED model: no diagonal self-mask) ----
# Label-label analysis uses the FINAL encoder layer only (mean over heads). Set to
# "last_three" / "all" / [3,4,5] to average more layers; per-head is available via
# the raw NPZ tensors. The label self-attention diagonal is shown (not masked).
ATTENTION_LAYERS = "last"
THRESHOLD_OBJECTIVE = "f1"
SELECTION_METRIC = "macro_auprc"
MIN_SUPPORT = 10                  # min samples per stratum for the directed effect
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- per-sample attention PDFs: label-balanced subset (rare labels covered) ----
LABEL_IMAGE_SPLITS = ["validation", "test"]   # train excluded (time-consuming)
ATTENTION_PDF_MAX_SAMPLES = 200   # per split (label-balanced) unless GENERATE_ALL_ATTENTION_PDFS
MIN_POS_PER_LABEL = 10            # guarantee >= this many positives per label in the subset

# ---- ablation settings ----
ABLATION_SEEDS = (0, 1, 2)        # >=3 seeds; reduce to (0,) for a quick look

# ---- long-running guards (default off) ----
RUN_TRAINING = False
RUN_BASELINES = False
RUN_ABLATIONS = False
RUN_LOSS_ABLATION = False         # optional CE / weighted-CE comparison (very last)
GENERATE_ALL_ATTENTION_PDFS = False
SAVE_FULL_RAW_ATTENTION = False

with open(DIRS["configs"] / "experiment_config.json", "w") as f:
    json.dump({"LABEL_NAMES": LABEL_NAMES, "masked": False,
               "ATTENTION_LAYERS": ATTENTION_LAYERS, "SELECTION_METRIC": SELECTION_METRIC,
               "MIN_SUPPORT": MIN_SUPPORT, "SEED": SEED,
               "ABLATION_SEEDS": list(ABLATION_SEEDS),
               "LABEL_IMAGE_SPLITS": LABEL_IMAGE_SPLITS}, f, indent=2)
print("config saved; device:", DEVICE)
""")

# ============================================================== 6 device report
step(6, "Device and dependency report", "", r"""
import torchvision, sklearn, matplotlib
DEP_REPORT = {
    "python": sys.version.split()[0], "torch": torch.__version__,
    "torchvision": torchvision.__version__, "sklearn": sklearn.__version__,
    "numpy": np.__version__, "matplotlib": matplotlib.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(DEP_REPORT, indent=2))
""")

# ============================================================== 7 label order
step(7, "Load label names and enforce saved label order", """
**Method.** If a checkpoint exists, assert its `label_names` match `LABEL_NAMES`
(token i -> head i -> label i).
""", r"""
CHECKPOINT_AVAILABLE = CHECKPOINT_PATH.exists()
if CHECKPOINT_AVAILABLE:
    _ck = checkpointing.load_checkpoint(CHECKPOINT_PATH, map_location="cpu")
    saved = _ck.get("label_names")
    if saved is not None:
        assert list(saved) == LABEL_NAMES, f"label order mismatch: {saved} != {LABEL_NAMES}"
    print("checkpoint label order verified.")
else:
    print(f"No checkpoint at {CHECKPOINT_PATH}; using configured LABEL_NAMES.")
""")

# ============================================================== 8 load annotations
step(8, "Load train, validation and test annotations", "", r"""
DATA_AVAILABLE = all(p.exists() for p in CSV_PATHS.values())
ANNOT = {}
if DATA_AVAILABLE:
    for s, p in CSV_PATHS.items():
        ANNOT[s] = pd.read_csv(p)
        print(f"{s}: {len(ANNOT[s])} rows")
else:
    missing = [str(p) for p in CSV_PATHS.values() if not p.exists()]
    print("DATA NOT FOUND -> data-dependent steps will be skipped.\nMissing:", *missing, sep="\n  ")
""")

# ============================================================== 9 validate schema
step(9, "Validate CSV schemas and binary labels", "", r"""
if DATA_AVAILABLE:
    for s, df in ANNOT.items():
        missing = [c for c in (["filename"] + LABEL_NAMES) if c not in df.columns]
        assert not missing, f"{s} missing columns: {missing}"
        vals = np.unique(df[LABEL_NAMES].values)
        assert set(np.unique(vals)).issubset({0, 1}), f"{s} has non-binary labels: {vals}"
    print("schema + binary-label validation passed.")
else:
    print("skipped (no data).")
""")

# ============================================================== 10 leakage
step(10, "Check missing images, duplicates and split leakage", "", r"""
if DATA_AVAILABLE:
    sets = {s: set(ANNOT[s]["filename"]) for s in ANNOT}
    for s in ANNOT:
        dup = ANNOT[s]["filename"].duplicated().sum()
        miss = sum(not (IMAGE_DIRS[s] / fn).exists() for fn in ANNOT[s]["filename"])
        print(f"{s}: duplicates={dup}, missing_images={miss}")
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        ov = sets[a] & sets[b]
        print(f"leakage {a} & {b}: {len(ov)}" + (f"  e.g. {list(ov)[:3]}" if ov else ""))
else:
    print("skipped (no data).")
""")

# ============================================================== 11 prevalence
step(11, "Calculate split-level label prevalence", "", r"""
if DATA_AVAILABLE:
    LABELS = {s: ANNOT[s][LABEL_NAMES].values.astype(int) for s in ANNOT}
    PREV_TABLE = label_groups.split_prevalence_table(LABELS, LABEL_NAMES)
    display(PREV_TABLE.pivot(index="label", columns="split", values="prevalence").reindex(LABEL_NAMES))
else:
    LABELS, PREV_TABLE = None, None
    print("skipped (no data).")
""")

# ============================================================== 12 save dist data
step(12, "Save label-distribution data", "", r"""
if DATA_AVAILABLE:
    p = DIRS["figures"] / "label_distribution"; p.mkdir(parents=True, exist_ok=True)
    PREV_TABLE.to_csv(p / "label_distribution_table.csv", index=False)
    print("saved", p / "label_distribution_table.csv")
else:
    print("skipped (no data).")
""")

# ============================================================== 13 plot dist
step(13, "Plot label distribution", """
**Output.** Publication-quality PDF (percent + log-count panels) with professional
per-split coloring (no rare/intermediate/dominant color encoding) + its CSV.
""", r"""
if DATA_AVAILABLE:
    GROUPS = label_groups.build_label_groups(LABELS["train"], LABEL_NAMES)
    label_groups.plot_label_distribution(
        LABELS, LABEL_NAMES, GROUPS,
        DIRS["figures"] / "label_distribution" / "label_distribution")
    print("label distribution figure written.")
else:
    print("skipped (no data).")
""")

# ============================================================== 14 groups
step(14, "Define rare, intermediate and dominant label groups", """
**Method.** From **training** prevalence only: rare <0.20, intermediate 0.20-0.70,
dominant >0.70. Groups drive the ablation design, not chemistry.
""", r"""
if DATA_AVAILABLE:
    GROUPS = label_groups.build_label_groups(LABELS["train"], LABEL_NAMES)
    RARE = GROUPS.names_for("rare"); DOMINANT = GROUPS.names_for("dominant")
    RARE_IDX, DOM_IDX = GROUPS.rare_idx, GROUPS.dominant_idx
    print("rare:", RARE, "\ndominant:", DOMINANT)
    display(GROUPS.to_frame())
else:
    GROUPS = RARE = DOMINANT = RARE_IDX = DOM_IDX = None
    print("skipped (no data).")
""")

# ============================================================== 15 transforms
step(15, "Construct deterministic evaluation transforms", """
**Method.** Non-augmented `val_transforms` for **all** attention extraction; a
chemistry-safe (no hue/grayscale) training transform for any retraining.
""", r"""
_, EVAL_TF = image_augmentation(normalize=True)
TRAIN_TF_CHEM, _ = image_augmentation(chemistry_safe=True)
print("deterministic eval transforms ready.")
""")

# ============================================================== 16 loaders
step(16, "Construct train/validation/test evaluation loaders", """
**Note.** Eval loaders are deterministic (`shuffle=False`, eval transforms).
""", r"""
from torch.utils.data import DataLoader
EVAL_LOADERS = {}
if DATA_AVAILABLE:
    for s in ("train", "validation", "test"):
        ds = MultiLabelMoleculeDataset(IMAGE_DIRS[s], CSV_PATHS[s], LABEL_NAMES, transform=EVAL_TF)
        EVAL_LOADERS[s] = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                                     worker_init_fn=WORKER_INIT, generator=GENERATOR)
    print({s: len(EVAL_LOADERS[s].dataset) for s in EVAL_LOADERS})
else:
    print("skipped (no data).")
""")

# ============================================================== 17 model
step(17, "Instantiate Swin-B and the label-token Transformer (unmasked)", """
**Output.** `MODEL` = `BackboneWithModel(swin_b, MultiLabelVisionTransformer)`,
**unmasked** (no diagonal self-mask). Pretrained Swin weights used when available.
""", r"""
CFG = experiments.ExperimentConfig(
    name="full_model", backbone="swin_b", n_labels=N_LABELS,
    attention_mask_mode="none", selection_metric=SELECTION_METRIC, seeds=ABLATION_SEEDS)
MODEL = experiments.build_label_token_model(CFG, weights="default").to(DEVICE)
print("model:", type(MODEL).__name__, "| masking active:", MODEL.model.mask_config.is_active())
""")

# ============================================================== 18 arch print
step(18, "Print architecture and verify (10+49)=59 tokens", "", r"""
inner = MODEL.model
print(f"label tokens={inner.n_labels}, image patches={inner.n_patches}, seq_len={inner.seq_len}")
assert inner.seq_len == 59
print("parameters:", sum(p.numel() for p in MODEL.parameters()))
""")

# ============================================================== 19 attn shapes
step(19, "Verify attention tensor shapes", "", r"""
MODEL.eval()
with torch.no_grad():
    _logits, _attn = MODEL(torch.randn(2, 3, 224, 224, device=DEVICE), return_attention_weights=True)
print("logits:", tuple(_logits.shape), "| #layers:", len(_attn), "| each:", tuple(_attn[0].shape))
assert _logits.shape == (2, N_LABELS) and len(_attn) == 6 and _attn[0].shape == (2, 12, 59, 59)
""")

# ============================================================== 20 mask mechanics
step(20, "Verify unmasked attention and the directed-edge mechanism", """
**Checks.** (a) the default model masks nothing (label self-attention is nonzero);
(b) the directed-edge block used by the communication ablation zeroes exactly the
requested cell `A[i,j]` after softmax with no NaNs.
""", r"""
with torch.no_grad():
    _, a0 = MODEL(torch.randn(2, 3, 224, 224, device=DEVICE), return_attention_weights=True)
assert MODEL.model.mask_config.is_active() is False
assert torch.stack([a0[-1][:, :, i, i] for i in range(N_LABELS)]).sum() > 0   # self-attn present
# sanity-check the edge-block path on a throwaway model
from ml_vit import MultiLabelVisionTransformer
_m = MultiLabelVisionTransformer(feature_size=(7,7), feature_channels=1024, embed_dim=384,
        n_encoder_layers=6, n_heads=12, n_labels=10, blocked_label_edges=[(0,3)]).eval()
with torch.no_grad():
    _, _ae = _m(torch.randn(1,1024,7,7), return_attention_weights=True)
assert all(not torch.isnan(x).any() for x in _ae)
assert torch.allclose(_ae[-1][:, :, 0, 3], torch.zeros(1,12), atol=1e-6)
print("unmasked default OK; directed-edge block zeroes A[0,3] exactly, no NaNs.")
""")

# ============================================================== 21 rowsums
step(21, "Verify attention row sums and absence of NaNs", "", r"""
for li, mat in enumerate(a0):
    assert not torch.isnan(mat).any()
    assert torch.allclose(mat.sum(-1), torch.ones_like(mat.sum(-1)), atol=1e-4)
print("all attention rows sum to ~1, no NaNs.")
""")

# ============================================================== 22 load/init ckpt
step(22, "Load an existing checkpoint or initialize a new experiment", "", r"""
if CHECKPOINT_AVAILABLE:
    CK = checkpointing.load_checkpoint(CHECKPOINT_PATH, map_location=DEVICE)
    MODEL, LOAD_REPORT = checkpointing.build_model_from_checkpoint(CK, backbone="swin_b", weights=None)
    MODEL = MODEL.to(DEVICE)
    print("loaded checkpoint; compatible:", LOAD_REPORT["compatible"])
else:
    CK, LOAD_REPORT = None, {"compatible": None}
    print("no checkpoint -> using initialized model (set RUN_TRAINING=True to train).")
""")

# ============================================================== 23 ckpt metadata
step(23, "Display checkpoint metadata and compatibility report", "", r"""
if CHECKPOINT_AVAILABLE:
    rep = checkpointing.checkpoint_report(CK)
    print(json.dumps({k: rep[k] for k in ["epoch", "selection_metric", "best_score",
          "best_val_loss"]}, indent=2, default=str))
    print("missing keys:", LOAD_REPORT.get("missing_keys"), "| unexpected:", LOAD_REPORT.get("unexpected_keys"))
else:
    print("no checkpoint metadata.")
""")

# ============================================================== 24 loss/opt/sched
step(24, "Construct loss, optimizer and (cosine) scheduler", """
**Method.** Frequency-adapted Asymmetric Loss, optimizer with head+backbone groups
from the start (0.1 ratio), and a **warmup + cosine** schedule.
""", r"""
if DATA_AVAILABLE:
    FREQS = label_groups.compute_prevalence(LABELS["train"]).tolist()
else:
    FREQS = [0.155, 0.387, 0.149, 0.704, 0.045, 0.814, 0.278, 0.021, 0.581, 0.841]
CRITERION = experiments.build_criterion(FREQS).to(DEVICE)
OPTIMIZER = optim.build_optimizer(MODEL, base_lr=1e-4, backbone_lr_ratio=0.1)
SCHEDULER = optim.build_scheduler(OPTIMIZER, optim.LRScheduleConfig())
print("criterion gamma_pos:", CRITERION.gamma_pos.tolist())
""")

# ============================================================== 25 lr plot
step(25, "Generate and plot the LR schedule before training", "", r"""
optim.save_lr_schedule(DIRS["figures"] / "learning_rate", n_epochs=50, base_lr=1e-4,
                       backbone_lr_ratio=0.1, cfg=optim.LRScheduleConfig())
print("LR schedule (warmup+cosine) CSV+PDF saved.")
""")

# ============================================================== 26 training
step(26, "Controlled training cell (guarded by RUN_TRAINING)", """
**Guard.** Trains only if `RUN_TRAINING and DATA_AVAILABLE`. Selection = macro-AUPRC
with val-loss tie-break; backbone unfreezes at epoch 10.
""", r"""
if RUN_TRAINING and DATA_AVAILABLE:
    from train import MultiLabelTrainer
    TRAIN_LOADER = DataLoader(
        MultiLabelMoleculeDataset(IMAGE_DIRS["train"], CSV_PATHS["train"], LABEL_NAMES,
                                  transform=TRAIN_TF_CHEM),
        batch_size=32, shuffle=True, num_workers=0, worker_init_fn=WORKER_INIT, generator=GENERATOR)
    trainer = MultiLabelTrainer(MODEL, TRAIN_LOADER, EVAL_LOADERS["validation"], CRITERION,
                                LABEL_NAMES, OPTIMIZER, SCHEDULER, device=DEVICE,
                                checkpoint_dir=str(DIRS["checkpoints"]), experiment_name=CFG.name,
                                selection_metric=SELECTION_METRIC,
                                experiment_config=CFG.to_dict(), seeds=SEEDS)
    trainer.train(num_epochs=50, patience=10, unfreeze_epoch=10, unfreeze_lr_ratio=0.1)
else:
    print("training skipped (RUN_TRAINING=%s, DATA_AVAILABLE=%s)." % (RUN_TRAINING, DATA_AVAILABLE))
""")

# ============================================================== 27 load best
step(27, "Load the selected best checkpoint", "", r"""
BEST_PATH = DIRS["checkpoints"] / CFG.name / "best_model.pth"
if BEST_PATH.exists():
    bc = checkpointing.load_checkpoint(BEST_PATH, map_location=DEVICE)
    checkpointing.load_state_dict_tolerant(MODEL, bc["model_state_dict"])
    THRESHOLDS = bc.get("thresholds")
    print("loaded best checkpoint from epoch", bc.get("epoch"))
elif CHECKPOINT_AVAILABLE:
    THRESHOLDS = checkpointing.checkpoint_report(CK).get("thresholds")
    print("using pre-existing checkpoint model.")
else:
    THRESHOLDS = None
    print("no trained checkpoint; downstream eval uses the initialized model.")
""")

# ============================================================== 28 val probs
step(28, "Generate validation probabilities", "", r"""
READY = DATA_AVAILABLE
if READY:
    VAL_PROBS, VAL_TGTS = inference.collect_predictions(MODEL, EVAL_LOADERS["validation"], DEVICE)
    print("val probs:", VAL_PROBS.shape)
else:
    VAL_PROBS = VAL_TGTS = None
    print("skipped (no data).")
""")

# ============================================================== 29 thresholds
step(29, "Select per-label validation thresholds", """
**Method.** Optimize F1 per label on validation only; freeze for test. Never 0.5.
""", r"""
if READY:
    SEL = evaluation.select_thresholds(VAL_PROBS, VAL_TGTS, LABEL_NAMES, objective=THRESHOLD_OBJECTIVE)
    THRESHOLDS = SEL["thresholds"]
    display(pd.DataFrame({"label": LABEL_NAMES, "threshold": THRESHOLDS, "val_score": SEL["scores"]}))
else:
    print("skipped (no data).")
""")

# ============================================================== 30 save thr
step(30, "Save threshold table", "", r"""
if READY:
    evaluation.save_thresholds(THRESHOLDS, LABEL_NAMES, DIRS["metrics"] / "thresholds.csv",
                               SEL["scores"], THRESHOLD_OBJECTIVE)
    print("thresholds saved.")
else:
    print("skipped (no data).")
""")

# ============================================================== 31 predictions
step(31, "Generate train/validation/test predictions", "", r"""
PROBS, TGTS = {}, {}
if READY:
    for s in ("train", "validation", "test"):
        PROBS[s], TGTS[s] = inference.collect_predictions(MODEL, EVAL_LOADERS[s], DEVICE)
    print({s: PROBS[s].shape for s in PROBS})
else:
    print("skipped (no data).")
""")

# ============================================================== 32 save preds
step(32, "Save all prediction tables", "", r"""
if READY:
    for s in ("train", "validation", "test"):
        ids = inference.dataset_sample_ids(EVAL_LOADERS[s])
        fns = inference.dataset_filenames(EVAL_LOADERS[s])
        df = evaluation.build_prediction_table(s, ids, fns, TGTS[s], PROBS[s], THRESHOLDS, LABEL_NAMES)
        evaluation.save_prediction_table(df, DIRS["predictions"] / f"{s}_predictions.csv")
    print("prediction tables saved.")
else:
    print("skipped (no data).")
""")

# ============================================================== 33 metrics
step(33, "Calculate full metric tables", """
**Output.** Per-label (AUPRC primary, AUROC, P/R/specificity/F1/BA/accuracy, support,
Brier) and overall (macro/micro AUPRC, F1, BA, Hamming, subset accuracy).
""", r"""
METRICS = {}
if READY:
    for s in ("validation", "test"):
        METRICS[s] = evaluation.compute_metric_table(PROBS[s], TGTS[s], THRESHOLDS, LABEL_NAMES)
        evaluation.save_metric_table(METRICS[s], DIRS["metrics"] / f"{s}_metrics")
    display(METRICS["test"]["per_label"][["auprc", "auroc", "f1", "recall", "precision", "balanced_accuracy"]])
else:
    print("skipped (no data).")
""")

# ============================================================== 34 conf raw
step(34, "Plot raw confusion matrices", "", r"""
if READY:
    preds_test = evaluation.apply_thresholds(PROBS["test"], THRESHOLDS)
    CONF = evaluation.per_label_confusion(TGTS["test"], preds_test, LABEL_NAMES)
    CONF.to_csv(DIRS["metrics"] / "confusion_matrices.csv")
    evaluation.plot_confusion_matrices(CONF, LABEL_NAMES,
        DIRS["figures"] / "confusion_matrices" / "confusion_raw", normalized=False)
    print("raw confusion matrices saved.")
else:
    print("skipped (no data).")
""")

# ============================================================== 35 conf norm
step(35, "Plot normalized confusion matrices", "", r"""
if READY:
    evaluation.plot_confusion_matrices(CONF, LABEL_NAMES,
        DIRS["figures"] / "confusion_matrices" / "confusion_normalized", normalized=True)
    print("normalized confusion matrices saved.")
else:
    print("skipped (no data).")
""")

# ============================================================== 36 attn config
step(36, "Configure attention extraction", """
**Note.** Deterministic eval loaders. Full raw 59x59x12x6 tensors are stored only
if `SAVE_FULL_RAW_ATTENTION`; otherwise compressed summaries.
""", r"""
ATTN_SPLITS = ["train", "validation", "test"] if DATA_AVAILABLE else []
print("attention extracted for:", ATTN_SPLITS or "(none - no data)")
""")

# ============================================================== 37 direct attn
step(37, "Extract and save direct attention (summaries)", "", r"""
if DATA_AVAILABLE:
    for s in ATTN_SPLITS:
        d = all_.collect_label_label(MODEL, EVAL_LOADERS[s], N_LABELS, DEVICE, layers=ATTENTION_LAYERS)
        np.savez_compressed(DIRS["attention_label_label"] / f"{s}_direct.npz",
                            A=d["A"].astype(np.float32), P=d["P"].astype(np.float32), Y=d["Y"].astype(np.int8))
        print(s, "label-label summary saved:", d["P"].shape)
else:
    print("skipped (no data).")
""")

# ============================================================== 38 rollout
step(38, "Calculate attention rollout", """
**Method.** Mean-over-heads per layer, add residual identity, row-normalize,
multiply through all six layers; extract the label->image block.
""", r"""
if DATA_AVAILABLE:
    imgs, tg, pr, attn = next(inference.iter_attention(MODEL, EVAL_LOADERS["test"], DEVICE))
    R = attn_utils.attention_rollout(attn)
    print("rollout:", tuple(R.shape), "| rows sum~1:",
          bool(torch.allclose(R.sum(-1), torch.ones(R.shape[0], R.shape[1]), atol=1e-3)))
else:
    print("skipped (no data).")
""")

# ============================================================== 39 split blocks
step(39, "Split attention into label-label and label-image blocks", "", r"""
if DATA_AVAILABLE:
    blocks = attn_utils.split_attention_blocks(attn[-1], N_LABELS)
    print({k: tuple(v.shape) for k, v in blocks.items()})
else:
    print("skipped (no data).")
""")

# ============================================================== 40 color limits
step(40, "Calculate label-to-image color limits (fixed from training)", """
**Method.** Per-label robust 1-99th percentile color limits from the **training**
split (bounded by a few batches so this stays fast). Used as the fixed scale for
all label->image overlays.
""", r"""
if DATA_AVAILABLE:
    COLOR_LIMITS = ali.accumulate_training_color_limits(
        MODEL, EVAL_LOADERS["train"], N_LABELS, DEVICE, method="direct",
        max_batches=None if GENERATE_ALL_ATTENTION_PDFS else 6)
    print("fixed per-label color limits (train, robust 1-99%):\n", np.round(COLOR_LIMITS, 5))
else:
    print("skipped (no data).")
""")

# ============================================================== 41-42 per-sample PDFs
md(r"""
### Per-sample attention PDFs on a LABEL-BALANCED subset (validation and test)

For ~`ATTENTION_PDF_MAX_SAMPLES` (200) samples per split, chosen so that **every
label has at least `MIN_POS_PER_LABEL` (10) positives** (rare labels are selected
first so Macrocycle / H-Bond Donors are represented), we save: the **label->image
overlays**, the **per-sample 10x10 label->label heatmap** (diagonal/self-attention
shown), and the **full 59x59 attention map**. This lets you verify directly that
dataset-averaging is not hiding per-sample structure and see how little of each
label token's budget goes to other labels vs. the 49 image patches. The per-sample
10x10 is mean-over-heads of the configured layers; raw per-head tensors are kept in
the NPZ when `SAVE_FULL_RAW_ATTENTION`. The train split is excluded (time-consuming).
""")
for n, split in [(41, "validation"), (42, "test")]:
    step(n, f"Generate {split} per-sample PDFs (label-balanced subset)", "", f"""
if DATA_AVAILABLE and ("{split}" in LABEL_IMAGE_SPLITS):
    _loader = EVAL_LOADERS["{split}"]
    if GENERATE_ALL_ATTENTION_PDFS:
        sample_loader = _loader; sub = None
    else:
        Y_split = np.asarray(_loader.dataset.labels)
        idx, counts = inference.select_balanced_sample_indices(
            Y_split, ATTENTION_PDF_MAX_SAMPLES, MIN_POS_PER_LABEL, seed=SEED)
        sample_loader = inference.subset_loader(_loader, idx)
        sub = None  # render all 10 labels for these representative samples
        print("{split}: balanced subset =", len(idx), "samples | positives/label:",
              dict(zip(LABEL_NAMES, counts.tolist())))
    res_img = ali.generate_split_label_image_outputs(
        MODEL, sample_loader, "{split}", LABEL_NAMES,
        DIRS["attention_label_image"], THRESHOLDS, COLOR_LIMITS,
        device=DEVICE, method="direct", max_samples=None, labels_subset=sub)
    res_ll = all_.generate_split_label_label_pdfs(
        MODEL, sample_loader, "{split}", LABEL_NAMES,
        DIRS["attention_label_label"], device=DEVICE, layers=ATTENTION_LAYERS,
        max_samples=None, save_full=True, save_full_raw=SAVE_FULL_RAW_ATTENTION)
    print("{split}: label-image", res_img, "| per-sample label-label/59x59", res_ll)
else:
    print("skipped ({split} not in LABEL_IMAGE_SPLITS or no data).")
""")

# ============================================================== 43 whole-dataset LL
step(43, "Collect per-sample label-to-label matrices (whole dataset)", """
**Method.** Because the data is limited, pool **train+val+test** for the label->label
analysis. `LL` holds absolute (A) and conditional (P) 10x10 matrices per sample plus
labels Y. Attention is post-softmax (the conditional preference is a normalized,
post-softmax quantity).
""", r"""
if DATA_AVAILABLE:
    LL = all_.collect_label_label_multi(MODEL, EVAL_LOADERS, N_LABELS, DEVICE, layers=ATTENTION_LAYERS)
    print("pooled samples:", LL["P"].shape[0], "| A:", LL["A"].shape, "P:", LL["P"].shape)
else:
    LL = None; print("skipped (no data).")
""")

# ============================================================== 44 masses
step(44, "Calculate absolute off-diagonal masses", "", r"""
if DATA_AVAILABLE:
    LL_MASS = all_.absolute_label_label_mass(LL["A"], LL["Y"])
    display(pd.DataFrame({"label": LABEL_NAMES, "abs_offdiag_mass": LL_MASS}))
else:
    print("skipped (no data).")
""")

# ============================================================== 45 cond pref
step(45, "Calculate conditional label preferences", """
**Definition.** `P[i,j] = A[i,j] / (sum_{k!=i} A[i,k] + eps)` (already in `LL['P']`;
the diagonal/self-attention is excluded from the denominator).
""", r"""
if DATA_AVAILABLE:
    P_MEAN = all_.naive_mean_attention(LL["P"])
    A_MEAN = all_.mean_label_label_attention(LL["A"])
    # Programmatically identify the attention HUBS: the source labels other tokens
    # attend to most. These drive the attention-informed ablations (cells 53-54).
    HUB_IDX = all_.top_source_labels(P_MEAN, k=2)
    HUBS = [LABEL_NAMES[i] for i in HUB_IDX]
    print("mean conditional preference and mean attention computed:", P_MEAN.shape, A_MEAN.shape)
    print("attention hubs (top-2 most-attended source labels):", HUBS)
else:
    HUB_IDX, HUBS = None, None
    print("skipped (no data).")
""")

# ============================================================== 46 naive diag
step(46, "Naive raw attention diagnostic (do not use for conclusions)", """
**Warning.** The naive sample-mean of P is **dominant-label biased**; shown only for
contrast with the imbalance-corrected effect.
""", r"""
if DATA_AVAILABLE:
    np.savez_compressed(DIRS["attention_label_label"] / "naive_mean_diagnostic.npz", naive=P_MEAN)
    print("naive diagnostic saved.")
else:
    print("skipped (no data).")
""")

# ============================================================== 47 balanced effect
step(47, "Run imbalance-balanced conditional analysis (whole dataset)", """
**Method.** `delta[i,j] = E[P|y_i=1,y_j=1] - E[P|y_i=1,y_j=0]`, equal stratum weight,
min support; unsupported relations stay NaN (never zero). Computed on the pooled
dataset for maximum support.
""", r"""
if DATA_AVAILABLE:
    EFFECT = all_.balanced_directed_effect(LL["P"], LL["Y"], min_support=MIN_SUPPORT)
    n_def = int(np.isfinite(EFFECT["delta"]).sum())
    print(f"defined off-diagonal relations: {n_def} of {N_LABELS*(N_LABELS-1)}"
          f" | reliable (>= {MIN_SUPPORT}/stratum): {int(EFFECT['reliable'].sum())}")
    np.savez_compressed(DIRS["attention_label_label"] / "directed_effect.npz",
                        delta=EFFECT["delta"], n_present=EFFECT["n_present"],
                        n_absent=EFFECT["n_absent"], reliable=EFFECT["reliable"])
else:
    EFFECT = None; print("skipped (no data).")
""")

# ============================================================== 48 effect heatmap
step(48, "Plot the corrected directed label-attention effect heatmap", """
**Output.** Publication-quality diverging heatmap of `delta[i,j]` centered at zero
(rows=receiver i, cols=source j). No credible stars, no hatched cells; the diagonal
and any zero-support cells are simply blank. Saved as PDF + data sidecar.
""", r"""
if DATA_AVAILABLE:
    llp.plot_label_label_effect(EFFECT, LABEL_NAMES,
        DIRS["figures"] / "label_label_attention" / "directed_effect_heatmap")
    print("corrected effect heatmap saved.")
else:
    print("skipped (no data).")
""")

# ============================================================== 49 attention heatmap
step(49, "Plot mean label-to-label attention heatmap", """
**Output.** Sequential heatmap of the dataset-mean label->label attention (diagonal
blank), publication-quality PDF + data.
""", r"""
if DATA_AVAILABLE:
    llp.plot_mean_attention(A_MEAN, LABEL_NAMES,
        DIRS["figures"] / "label_label_attention" / "mean_attention_heatmap")
    print("mean attention heatmap saved.")
else:
    print("skipped (no data).")
""")

# ============================================================== 50 preference heatmap
step(50, "Plot mean conditional label-preference heatmap", """
**Output.** Sequential heatmap of the dataset-mean conditional preference P[i,j].
""", r"""
if DATA_AVAILABLE:
    llp.plot_conditional_preference(P_MEAN, LABEL_NAMES,
        DIRS["figures"] / "label_label_attention" / "conditional_preference_heatmap")
    print("conditional preference heatmap saved.")
else:
    print("skipped (no data).")
""")

# ============================================================== 51 layerwise
step(51, "Run layer-wise sensitivity analysis", """
**Output.** Per-layer directed-effect tables (each layer analyzed on its own) for
the sensitivity appendix.
""", r"""
if DATA_AVAILABLE:
    LAYERWISE = all_.layerwise_sensitivity(MODEL, EVAL_LOADERS["test"], N_LABELS, 6, DEVICE, MIN_SUPPORT)
    for l, eff in LAYERWISE.items():
        np.savez_compressed(DIRS["attention_label_label"] / f"layer{l}_effect.npz",
                            delta=eff["delta"], reliable=eff["reliable"])
    print("per-layer effect tables saved for layers", list(LAYERWISE))
else:
    print("skipped (no data).")
""")

# ============================================================== 52 ablation setup
md(r"""
## Controlled ablations (dominant supervision & directed communication)

Both ablations run **six cases**: the full model plus five removal sets — remove
each dominant label individually, remove the first two dominants, and remove the
first three (all) dominants. Each case is retrained for every seed in
`ABLATION_SEEDS` (>=3). The primary outcome is the change in rare-label test AUPRC
vs. the full model. A negative change means the rare predictor *benefited* from that
supervision / communication path under the controlled intervention — never chemical
causation.

**Compute warning.** This trains 6 cases x len(ABLATION_SEEDS) Swin-B models per
ablation. Reduce `ABLATION_SEEDS=(0,)` for a quick look.
""")
step(52, "Ablation common setup + full-model reference", "", r"""
ABL = {}  # name -> list of per-seed result dicts
FULL_AGG = None
if RUN_ABLATIONS and DATA_AVAILABLE:
    ABLATION_TRAIN_LOADER = DataLoader(
        MultiLabelMoleculeDataset(IMAGE_DIRS["train"], CSV_PATHS["train"], LABEL_NAMES,
                                  transform=TRAIN_TF_CHEM),
        batch_size=32, shuffle=True, num_workers=0, worker_init_fn=WORKER_INIT, generator=GENERATOR)
    LOADERS_FULL = {"train": ABLATION_TRAIN_LOADER, "val": EVAL_LOADERS["validation"], "test": EVAL_LOADERS["test"]}
    BASE_CFG = experiments.ExperimentConfig(
        name="full_model", n_labels=N_LABELS, selection_metric=SELECTION_METRIC, seeds=ABLATION_SEEDS)
    ABL["full_model"] = experiments.run_experiment(BASE_CFG, LOADERS_FULL, LABEL_NAMES, FREQS, DEVICE, str(DIRS["ablations"]))
    FULL_AGG = experiments.aggregate_seeds(ABL["full_model"], "auprc")
    print("full-model reference trained over seeds", ABLATION_SEEDS)
else:
    print("ablations not executed (RUN_ABLATIONS=%s)." % RUN_ABLATIONS)
""")

# ============================================================== 53 supervision
step(53, "Dominant-supervision ablation + attention-informed hub ablation", """
**Design (9B).** Set the target label(s)' loss weight to zero and retrain
(architecture and all ten tokens unchanged). We run two families: the **dominant**
labels (full + remove{each, first two, first three}) and the **attention hubs**
`HUB_IDX` (full + remove{each, both}) discovered programmatically in cell 45.
""", r"""
SUP_AGG = {}
if RUN_ABLATIONS and DATA_AVAILABLE:
    sup_cfgs = experiments.supervision_ablation_configs(BASE_CFG, DOM_IDX, LABEL_NAMES)
    sup_cfgs += experiments.supervision_ablation_configs(BASE_CFG, HUB_IDX, LABEL_NAMES, name_prefix="drop_sup_hub__")
    seen = set()
    for c in sup_cfgs:
        if c.name in seen:
            continue
        seen.add(c.name)
        ABL[c.name] = experiments.run_experiment(c, LOADERS_FULL, LABEL_NAMES, FREQS, DEVICE, str(DIRS["ablations"]))
        SUP_AGG[c.name] = experiments.aggregate_seeds(ABL[c.name], "auprc")
    print("supervision ablation complete (dominant + hubs):", list(SUP_AGG))
else:
    print("skipped (RUN_ABLATIONS=%s)." % RUN_ABLATIONS)
""")

# ============================================================== 54 communication
step(54, "Directed-communication ablation + attention-informed hub block", """
**Design (9C, primary).** Block directed label->label edges during train+eval and
measure the change. Two families: **dominant -> rare** (block each dominant source
to all rare receivers) and the **hub -> all** block (block each attention hub source
to every other label). The hubs are the highest-traffic routes, so this is the most
direct test of whether the label-to-label channel is load-bearing.
""", r"""
COMM_AGG = {}
if RUN_ABLATIONS and DATA_AVAILABLE:
    comm_cfgs = experiments.communication_ablation_configs(BASE_CFG, DOM_IDX, RARE_IDX, LABEL_NAMES)
    comm_cfgs += experiments.communication_ablation_configs(BASE_CFG, HUB_IDX, list(range(N_LABELS)), LABEL_NAMES, name_prefix="block_comm_hub__")
    seen = set()
    comm_cfgs = [c for c in comm_cfgs if not (c.name in seen or seen.add(c.name))]
    for c in comm_cfgs:
        ABL[c.name] = experiments.run_experiment(c, LOADERS_FULL, LABEL_NAMES, FREQS, DEVICE, str(DIRS["ablations"]))
        COMM_AGG[c.name] = experiments.aggregate_seeds(ABL[c.name], "auprc")
    print("communication ablation complete:", list(COMM_AGG))
else:
    print("skipped (RUN_ABLATIONS=%s)." % RUN_ABLATIONS)
""")

# ============================================================== 55 test-time
step(55, "Optional test-time edge interventions", """
**Note.** OOD test-time probe (mask at inference only); **not** equivalent to
controlled retraining.
""", r"""
TTI = {}
if DATA_AVAILABLE and CHECKPOINT_AVAILABLE:
    base_probs, _ = inference.collect_predictions(MODEL, EVAL_LOADERS["test"], DEVICE)
    for j in (DOM_IDX or []):
        for i in (RARE_IDX or []):
            with experiments.test_time_edge_intervention(MODEL, blocked_label_edges=[(i, j)]):
                p, _ = inference.collect_predictions(MODEL, EVAL_LOADERS["test"], DEVICE)
            TTI[(LABEL_NAMES[j], LABEL_NAMES[i])] = float(np.mean(np.abs(p[:, i] - base_probs[:, i])))
    print("test-time interventions probed:", len(TTI), "edges.")
else:
    print("test-time intervention skipped (needs data + checkpoint).")
""")

# ============================================================== 56 change matrices
step(56, "Plot rare-label AUPRC change matrices", """
**Output.** Two diverging matrices (rows = rare targets, cols = removed dominant
set), values = AUPRC change vs full. Negative => the rare predictor benefited.
""", r"""
if FULL_AGG is not None and (SUP_AGG or COMM_AGG):
    if SUP_AGG:
        CH_SUP = experiments.rare_auprc_change_matrix(FULL_AGG, SUP_AGG, RARE)
        compare_plots.plot_change_matrix(CH_SUP, DIRS["figures"] / "ablations" / "supervision_change_matrix",
            title="Rare-label AUPRC change: dominant-supervision removal")
    if COMM_AGG:
        CH_COMM = experiments.rare_auprc_change_matrix(FULL_AGG, COMM_AGG, RARE)
        compare_plots.plot_change_matrix(CH_COMM, DIRS["figures"] / "ablations" / "communication_change_matrix",
            title="Rare-label AUPRC change: directed-communication block")
    print("change matrices plotted.")
else:
    print("change matrices skipped (run ablations first).")
""")

# ============================================================== 57 forest
step(57, "Plot rare-label performance-change forest plot", """
**Output.** Forest plot of per-rare-label AUPRC change (mean ± seed SD) for the
remove-all-three-dominant communication block (the strongest intervention).
""", r"""
if FULL_AGG is not None and COMM_AGG:
    key = [k for k in COMM_AGG if k.count("+") == 2]  # the 'first three' (all dominant) set
    if key:
        ab = COMM_AGG[key[0]]
        forest = pd.DataFrame({
            "delta_mean": (ab.loc[RARE, "auprc_mean"] - FULL_AGG.loc[RARE, "auprc_mean"]).values,
            "delta_ci_low": (ab.loc[RARE, "auprc_mean"] - FULL_AGG.loc[RARE, "auprc_mean"]).values - ab.loc[RARE, "auprc_std"].values,
            "delta_ci_high": (ab.loc[RARE, "auprc_mean"] - FULL_AGG.loc[RARE, "auprc_mean"]).values + ab.loc[RARE, "auprc_std"].values,
        }, index=RARE)
        compare_plots.plot_rare_change_forest(forest, DIRS["figures"] / "ablations" / "rare_change_forest",
            title="Rare-label AUPRC change (block all dominant communication)")
        print("forest plot saved.")
    else:
        print("forest skipped (need the all-dominant block).")
else:
    print("forest skipped (run ablations first).")
""")

# ============================================================== 58 baselines explain
md(r"""
## Baselines

We compare three model families to isolate where the benefit comes from:

1. **Independent single-label** — ten *separately trained* Swin-B binary classifiers
   (Swin-B + global average pooling + one sigmoid head per property). No information
   is shared across properties at all.
2. **Shared multi-label CNN** — one Swin-B backbone + global average pooling + ten
   independent linear heads. Shares the backbone but has **no label tokens and no
   label-to-label attention**.
3. **Label-token model** — the main (unmasked) model with ten label tokens and a
   6-layer Transformer (label-to-label + label-to-image attention).

`(1) vs (2)` measures the value of sharing a backbone; `(2) vs (3)` measures the
value of the label tokens and their attention. Each best model is checkpointed as
`best_model.pth` under `outputs/baselines/`.
""")
step(58, "Load or execute independent single-label baselines", "", r"""
BASE_RESULTS = {}
def build_loader(ds, shuffle):
    return DataLoader(ds, batch_size=32, shuffle=shuffle, num_workers=0,
                      worker_init_fn=WORKER_INIT, generator=GENERATOR)
if RUN_BASELINES and DATA_AVAILABLE:
    DATASETS = {s: MultiLabelMoleculeDataset(IMAGE_DIRS[s], CSV_PATHS[s], LABEL_NAMES, transform=EVAL_TF)
                for s in ("train", "validation", "test")}
    BASE_RESULTS["independent_single_label"] = [
        baselines.train_independent_single_label(DATASETS, LABEL_NAMES, FREQS, build_loader, DEVICE, str(DIRS["baselines"]))]
    print("single-label baselines complete (best models saved under outputs/baselines/single_label/).")
else:
    print("single-label baselines not executed (RUN_BASELINES=%s)." % RUN_BASELINES)
""")

# ============================================================== 59 shared base
step(59, "Load or execute shared multi-label baseline", "", r"""
if RUN_BASELINES and DATA_AVAILABLE:
    LOADERS_B = {"train": build_loader(DATASETS["train"], True),
                 "val": EVAL_LOADERS["validation"], "test": EVAL_LOADERS["test"]}
    BASE_RESULTS["shared_cnn"] = [baselines.train_shared_cnn(LOADERS_B, LABEL_NAMES, FREQS, DEVICE, str(DIRS["baselines"]))]
    print("shared-CNN baseline complete (best model saved under outputs/baselines/shared_cnn/).")
else:
    print("shared-CNN baseline not executed (RUN_BASELINES=%s)." % RUN_BASELINES)
""")

# ============================================================== 60 agg base
step(60, "Aggregate baseline metrics", "", r"""
if BASE_RESULTS:
    BASE_LONG = baselines.aggregate_baseline_metrics(BASE_RESULTS, metric="auprc")
    BASE_LONG.to_csv(DIRS["baselines"] / "baseline_auprc.csv", index=False)
    display(BASE_LONG.head())
else:
    BASE_LONG = None; print("no baseline results to aggregate.")
""")

# ============================================================== 61 base plot
step(61, "Plot baseline comparison", "", r"""
if BASE_LONG is not None:
    compare_plots.plot_baseline_comparison(BASE_LONG, DIRS["figures"] / "baselines" / "baseline_comparison", metric="auprc")
    print("baseline comparison plotted.")
else:
    print("skipped (no baseline results).")
""")

# ============================================================== 62 loss ablation (LAST, optional)
md(r"""
## (Optional, last) Loss-function ablation

Compare the frequency-adapted **Asymmetric Loss** against plain **cross-entropy**
(`bce`) and **weighted cross-entropy** (`weighted_bce`, per-label
`pos_weight=(1-freq)/freq`). This is appended last and disabled by default — set
`RUN_LOSS_ABLATION=True` only if you decide to run it.
""")
step(62, "Run optional loss-function ablation", "", r"""
LOSS_AGG = {}
if RUN_LOSS_ABLATION and DATA_AVAILABLE:
    for lt in ("asl", "bce", "weighted_bce"):
        c = experiments.ExperimentConfig(name=f"loss_{lt}", n_labels=N_LABELS,
                                         loss_type=lt, selection_metric=SELECTION_METRIC, seeds=ABLATION_SEEDS)
        res = experiments.run_experiment(c, LOADERS_FULL, LABEL_NAMES, FREQS, DEVICE, str(DIRS["ablations"]))
        LOSS_AGG[lt] = experiments.aggregate_seeds(res, "auprc").reset_index()
        LOSS_AGG[lt]["model"] = lt
    long = pd.concat(LOSS_AGG.values(), ignore_index=True)
    compare_plots.plot_baseline_comparison(long, DIRS["figures"] / "ablations" / "loss_ablation", metric="auprc")
    print("loss-function ablation complete:", list(LOSS_AGG))
else:
    print("loss-function ablation not executed (RUN_LOSS_ABLATION=%s)." % RUN_LOSS_ABLATION)
""")

# ============================================================== 63 manifest
step(63, "Generate final machine-readable result manifest", "", r"""
MANIFEST = {
    "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    "seeds": SEEDS, "ablation_seeds": list(ABLATION_SEEDS),
    "dependencies": DEP_REPORT, "device": DEVICE,
    "label_names": LABEL_NAMES, "masked": False,
    "data_available": DATA_AVAILABLE, "checkpoint_available": CHECKPOINT_AVAILABLE,
    "flags": {"RUN_TRAINING": RUN_TRAINING, "RUN_BASELINES": RUN_BASELINES,
              "RUN_ABLATIONS": RUN_ABLATIONS, "RUN_LOSS_ABLATION": RUN_LOSS_ABLATION,
              "GENERATE_ALL_ATTENTION_PDFS": GENERATE_ALL_ATTENTION_PDFS},
    "label_image_splits": LABEL_IMAGE_SPLITS,
    "outputs": {k: str(v) for k, v in DIRS.items()},
}
with open(DIRS["reports"] / "manifest.json", "w") as f:
    json.dump(MANIFEST, f, indent=2, default=str)
print("manifest written to", DIRS["reports"] / "manifest.json")
""")

# ============================================================== 64 summary (md)
md(r"""
### 64. Concise research summary and limitations

**Summary.** The unmasked label-token model is trained/loaded, validation-selected
per-label thresholds are frozen, and imbalance-appropriate metrics (AUPRC primary)
are reported. Label->image attention (validation/test) and pooled label->label
attention are extracted; the imbalance-corrected directed effect `delta[i,j]` is
rendered as a publication heatmap alongside mean-attention and conditional-preference
heatmaps. Six-case dominant-supervision and directed-communication ablations and the
single-label / shared-CNN baselines quantify whether rare-label predictors benefit
from dominant-label supervision/communication.

**Limitations.**
- Attention is **information routing**, not chemical causation.
- Ablation effects are *performance effects of a controlled model intervention*.
- 7x7 maps are coarse Swin-feature localization, not atom-level attention.
- The directed effect is computed on post-softmax conditional preferences pooled
  over all splits; conclusions require >=3 seeds, and unsupported relations are
  reported as undefined (never imputed to zero).
""")
code(r"""
LIMITATIONS = [
    "attention = information routing, not chemical causation",
    "ablation = performance effect of a controlled model intervention",
    "7x7 maps = coarse Swin-feature localization, not atom-level",
    "directed effect computed on post-softmax conditional preferences (pooled splits)",
]
with open(DIRS["reports"] / "limitations.json", "w") as f:
    json.dump(LIMITATIONS, f, indent=2)
print("\n".join("- " + x for x in LIMITATIONS))
""")

# ============================================================== 65 locations
step(65, "Print all output locations", "", r"""
print("Output locations:")
for k, v in DIRS.items():
    n = len(list(v.rglob('*'))) if v.exists() else 0
    print(f"  {k:24s} {v}  ({n} files)")
""")

# ----------------------------------------------------------------- assemble
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
out = Path(__file__).parent / "notebooks" / "molecular_property_attention_analysis.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out))
print(f"wrote {out} with {len(cells)} cells")
