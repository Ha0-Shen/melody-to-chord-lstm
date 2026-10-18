# Repository checks

Checked on 18 October 2026 in a Linux environment with Python 3.12. The optional model smoke check used CPU PyTorch 2.14.1+cpu. These versions describe the check environment, not the original training environment.

| Check | Result | Scope |
| --- | --- | --- |
| Notebook structure and code-cell syntax | PASS | Three research notebooks; 56 code cells; no preserved outputs or execution counters. |
| Baseline/weighted configuration consistency | PASS | Checked literal scientific settings and opposite class-weighting flags. |
| Relative Markdown links | PASS | Repository-local targets exist. |
| Synthetic data-interface tests | PASS | Eight tests: release order/end handling, gold-label isolation, invalid alignment, unsupported events, full evaluation tails, training windows, deterministic splits and mandatory no-chord selection. |
| Future-input perturbation | PASS | Both actual notebook model classes: maximum earlier-logit change 0.0. |
| Full-sequence/chunked inference | PASS | Both model classes: maximum difference 0.0. |
| Full-sequence/token-at-a-time inference | PASS | Both model classes: maximum absolute difference approximately 2.38e-7. |
| Synthetic optimiser update | PASS | Both model classes: finite loss/gradients and changed classifier weights. |
| Results exporter | PASS | Synthetic fixtures exercise a valid three-condition export and rejection of batch-limited or incompatible runs; generated fixture results are not included. |

The model smoke checks use random weights, small dimensions and synthetic inputs. They verify the inference interface and a training update; they are not model accuracy or musical-quality measurements.

The checks did not include a full dataset download, corpus preprocessing, model training or end-to-end live latency test. No trained checkpoint is included. GitHub Actions results are separate from these local checks.
