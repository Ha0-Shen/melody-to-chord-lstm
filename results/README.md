# Experimental results

The figures below come from my dissertation runs on Google Colab with an NVIDIA T4 GPU. Metric files, checkpoints and saved notebook outputs are not stored in this repository, so these tables are the published summary.

## Dataset

- 6,672 MusicXML files attempted; 5,022 accepted; 1,650 rejected.
- Rejection reasons: ambiguous chord annotation 1,098; no part with both melody and chords 434; overlapping melody notes 109; other parsing errors 9.
- Split by song group: 3,979 training, 535 validation, 508 test.
- At 12 PPQ, three accepted songs had timing values that could not be represented safely. At 24 PPQ there were none, so 24 PPQ was used.

## Test metrics

| Metric | Top-200 | Top-250 | Top-200 class-weighted |
| --- | ---: | ---: | ---: |
| Cross-entropy | 2.7500 | 2.7851 | 2.9701 |
| Perplexity | 15.6434 | 16.2008 | 19.4943 |
| Slot accuracy | 0.3401 | 0.3384 | 0.3063 |
| Retained-only accuracy | 0.3487 | 0.3452 | 0.3144 |
| Top-3 accuracy | 0.5487 | 0.5461 | 0.5012 |
| Macro F1 excluding `<UNK>` | 0.0497 | 0.0410 | 0.0634 |
| Weighted F1 excluding `<UNK>` | 0.2975 | 0.2943 | 0.2945 |
| Chord-onset accuracy excluding `<UNK>` | 0.2229 | 0.2174 | 0.1930 |
| `<UNK>` target rate | 2.71% | 2.02% | 2.71% |

Majority-class baseline slot accuracy: 0.0944. The Top-200 model has 530,506 trainable parameters.

## Causality and latency

- Chunked versus full-sequence inference: maximum absolute logit difference 9.54e-6, below the 1e-4 tolerance.
- Model-only step time on a T4 GPU: median about 0.542 ms, 95th percentile about 0.858 ms.

## Notes

Earlier experiments used an 805-label output space. That is a different task and is not comparable with the tables above.

To rebuild a comparison table from your own completed runs, use `scripts/collect_results.py` as described in the [reproducibility guide](../docs/REPRODUCIBILITY.md).
