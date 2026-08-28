# VISTA Test-Holdout Policy

The final strengthened VISTA study must keep the test split isolated from experiment design.

## Allowed before final test evaluation
- training-split label prevalence;
- validation-based checkpoint selection;
- validation-only per-label threshold tuning;
- training-only attention hub discovery;
- predeclared train+validation exploratory summaries if explicitly labeled development-only.

## Not allowed for confirmatory experiment design
- selecting attention hubs from pooled train+validation+test attention;
- choosing ablation targets because they look strongest on test;
- changing augmentation, loss, seed set, or architecture after inspecting strict-test performance;
- selecting representative success/failure examples from strict test and then using them to modify the model.

## Historical notebook caveat

The historical notebook computes:

`LL = collect_label_label_multi(MODEL, EVAL_LOADERS, ...)`

over train + validation + test and then derives `HUB_IDX` from the pooled conditional-preference matrix.

That is acceptable as a historical descriptive analysis, but those pooled hubs must **not**
be used to define the final confirmatory EPFL hub-ablation experiments.

For final ablations:
1. derive hubs from training attention only;
2. freeze the hub list;
3. train all intervention models;
4. tune thresholds on validation only;
5. evaluate the fixed experiments on test.

## Interpretability figures

Historical sample-split attention examples remain useful as reference provenance.
Final strict-split interpretability figures should be regenerated from the strict model.
