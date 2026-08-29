# Frozen GPU launch order

1. `bash launch/00_runtime_gate.sh`
2. `bash launch/01_e0_vista_seed42_gpu0.sh`
3. **STOP. Send E0 to Shayan.**
4. After Shayan approval, record it with `scripts/record_e0_approval.py`.
5. E1 strict VISTA: launch 02, 03, then 04.
6. E2 exact shared-Swin: 05, 06, 07.
7. E3 GIN-family: 08 tuning, then 09, 10, 11.
8. Seed-42 exports/controls: 12.
9. Repeated-seed summaries: 13.
10. Build return bundle: 99.

Use `tmux`. One independent seed per GPU. Do not use both GPUs for one seed.
