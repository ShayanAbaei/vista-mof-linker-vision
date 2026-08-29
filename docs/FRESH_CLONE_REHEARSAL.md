# Fresh-clone rehearsal

There are two gates.

## A. Local Windows static/data gate

On Windows we deliberately do **not** call `bash -n D:\...`.
Windows `bash` normally resolves to WSL, while WSL expects mounted POSIX paths such
as `/mnt/d/...`. The local Python rehearsal therefore performs Windows-safe textual
checks. Bash syntax is release-tested on Linux and is checked again on EPFL/Linux.

After installing the final patch:

```text
python scripts\pre_epfl_rehearsal.py --repo-root .
```

## B. Actual fresh clone after the branch is pushed

```text
python scripts\fresh_clone_rehearsal.py ^
  --repo-url https://github.com/ShayanAbaei/vista-mof-linker-vision.git ^
  --ref pre-epfl-handover
```

Optionally include the full Zenodo fetch:

```text
python scripts\fresh_clone_rehearsal.py ^
  --repo-url https://github.com/ShayanAbaei/vista-mof-linker-vision.git ^
  --ref pre-epfl-handover ^
  --fetch-images
```

The Torch/GPU gate is intentionally deferred to EPFL because local Windows does not
need a PyTorch installation for handover engineering.
