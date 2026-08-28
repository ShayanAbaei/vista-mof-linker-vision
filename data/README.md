# VISTA data layout

The heavy molecular-image archive is intentionally stored outside normal Git.

## Frozen dataset record

Zenodo DOI: `10.5281/zenodo.YOUR_DOI_NUMBER`  
Zenodo record ID: `YOUR_RECORD_ID`

## Local folders

- `source/` â€” curated original CSV/audit inputs imported from the historical VISTA workspace.
- `frozen/` â€” normalized manifests, label definitions, image manifest, validation report, and checksums produced by `scripts/validate_dataset.py`.
- `raw/` â€” ignored local download location for `vista_images_9414.zip`.
- `images_224/` â€” ignored extracted molecular images.

## Validation

Run:

```bash
python scripts/validate_dataset.py
```

The strict frozen manifest must contain 9,414 rows with train/validation/test counts 6,590 / 1,412 / 1,412 and zero cross-split canonical first-linker identities.

## QMOF provenance

The data derive from QMOF/MOFid-linked records. QMOF data are CC BY 4.0 and require attribution. VISTA adds linker selection, canonicalization, 2D image rendering, deterministic proxy labels, and split/audit products.


