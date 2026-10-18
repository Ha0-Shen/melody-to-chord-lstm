# Experimental results

The uploaded source archive does not include metric files, checkpoints, training histories, figures or saved notebook outputs. No numerical comparison is published in this version.

The notebooks implement three conditions: Top-200 unweighted, Top-250 unweighted, and Top-200 class-weighted. To add a traceable results section, use outputs from those exact completed conditions. Earlier 805-label experiments are a different task and should not be substituted into this comparison.

For each run, retain:

- `test_metrics.json` with the experimental configuration and split hashes;
- `training_history.csv` and the selected checkpoint epoch;
- `compatibility_preflight.json`;
- `causality_and_latency.json`, with device/runtime details when reporting latency;
- the corresponding preprocessing/encoding summaries and `data_provenance.json`;
- the final experiment comparison and any figures used to discuss findings.

Run `scripts/collect_results.py` as described in the reproducibility guide to create a compact public comparison from completed metrics. It does not independently rerun evaluation. A useful discussion should cover rare-label behaviour, unknown-label rates and onset errors as well as aggregate accuracy.

Trained weights can be distributed separately once their provenance, size and sharing terms have been checked. They are not required to review the implementation, but are required for a trained-model demonstration without retraining.
