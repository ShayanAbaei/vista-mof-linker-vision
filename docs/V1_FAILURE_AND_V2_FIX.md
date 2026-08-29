# Why the v1 rehearsal failed on Windows

The v1 rehearsal executed:

```text
bash -n D:\UNIVERSITY\...\launch\00_runtime_gate.sh
```

from a Windows Python process.

On Windows, `bash` resolved to WSL. WSL expects Linux-mounted paths such as:

```text
/mnt/d/UNIVERSITY/...
```

A raw `D:\...` argument therefore became an invalid Linux path. The separate
localhost-proxy warning was not the scientific problem.

This was an orchestration bug in v1, not a failure of the launch scripts.

Final v2 fixes it by:
- never invoking WSL bash with native Windows paths;
- doing Windows-safe textual launch checks locally;
- bash-syntax checking the release package on Linux;
- repeating `bash -n` on EPFL/Linux with native POSIX paths.

Final v2 also supersedes v1 scientifically/engineering-wise:
- exact frozen shared-Swin architecture;
- explicit negative-label attention control;
- explicit performance-vs-Tanimoto output;
- optional but explicit MOF-level consequence screen;
- E0 numerical gate + manual Shayan approval;
- exact EPFL environment instructions;
- actual fresh-clone helper;
- automated EPFL return-bundle builder.
