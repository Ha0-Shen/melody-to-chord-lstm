# Method and evaluation

## Task definition

The task is causal sequence labelling: predict the current annotated chord at each 24-PPQ clock position from the symbolic melody observed so far. It does not generate audio, recognise melody from a microphone, or synthesise chord voicings.

PPQ means pulses per quarter note. The 24-PPQ grid has 24 subdivisions per quarter note; it is a musical-time resolution, not a fixed millisecond latency. At 120 BPM, one step is approximately 20.8 ms. This is a grid interval, not a measured response time.

## Source alignment and quality control

The preprocessing notebook reads melody notes and chord symbols from the same compressed MusicXML file. A candidate part must contain both. Repeat expansion is applied to the complete part, retaining the original part when the repeat expander reports inconsistency. Ties are consolidated before notes are stored as onset, end and pitch.

The pipeline checks for genuinely overlapping melody notes and ambiguous chord states. Quantisation audits distinguish ordinary boundary rounding from unsafe cases, such as collapsed durations, merged distinct onsets or false overlaps. Failed records are logged with reasons and excluded.

Chord symbols are converted with ChoCo's Wikifonia converter. A pitch-content fallback constructs and validates a Harte-compatible label where required. Root spellings are canonicalised and chord inversions are stripped by the checked-in policy; these are modelling choices that reduce representational detail.

## Splits and vocabulary

Record identifiers are hashes of relative file paths. Group identifiers use the normalised basename after removing `.mxl` or numeric suffixes such as `.mxl.1`. A hash of the group identifier assigns approximately 80% of groups to training, 10% to validation and 10% to test. Hash buckets do not guarantee exact split proportions.

Only training data rank chord labels, using supervised-slot frequency with deterministic tie-breakers. The semantic no-chord label `N` is always retained. Labels outside the selected vocabulary map to `<UNK>`; `<PAD>` is used for padding and unsupervised event positions.

This split protects against the filename variants represented by the grouping rule. More extensive composition-level deduplication would need additional checks.

## From offline notes to observable events

The preprocessing stage initially has completed note durations, which are needed to construct an aligned offline representation. The training notebooks then replace duration events with releases before passing inputs to the LSTM. A new `NOTE_OFF` input ID is added to the MIDITok vocabulary.

The following is an illustrative event schedule, not a model prediction or a corpus excerpt. A MIDI pitch of 60 sounds over steps 0 and 1; pitch 64 starts at step 2.

| Grid step | Newly observable input | Supervised prediction position |
| ---: | --- | --- |
| 0 | `Pitch(60)`, clock | At the clock token |
| 1 | clock | At the clock token |
| 2 | `NOTE_OFF`, `Pitch(64)`, clock | At the clock token |
| 3 | clock | At the clock token |
| 4 | `NOTE_OFF`, clock | At the clock token |

The future length of either note is absent at its onset. When release and attack coincide, the old note is released first. `BOS` and `EOS` delimit offline sequences; a live implementation would need explicit session boundaries. Clock positions carry chord targets, while note events and boundary tokens are masked out of the loss.

## Model and training

The model is an embedding followed by input dropout, a two-layer unidirectional LSTM, layer normalisation, output dropout and a linear classifier. Hidden and cell states can be supplied to the next inference call.

Training uses shuffled, overlapping windows with fresh state for each window. Validation and test traverse songs in order, retaining state across consecutive non-overlapping windows and resetting it at song boundaries. Short evaluation tails are retained so that no supervised tail position is silently discarded.

The weighted variant uses training class frequencies: square-root inverse-frequency weights, mean normalisation over seen classes and a cap of 5. The padding weight is zero. Validation and test likelihoods remain unweighted. Early stopping and checkpoint selection use unweighted validation cross-entropy.

## Evaluation protocol

| Metric | Interpretation and qualification |
| --- | --- |
| Slot accuracy | Fraction of supervised clock positions with the correct mapped label; frequent sustained chords can dominate. |
| Retained-only slot accuracy | Accuracy after removing positions whose gold target is `<UNK>`. |
| Top-3 accuracy | Whether the mapped gold label appears among the three highest-scoring classifier outputs. |
| Macro F1 including `<UNK>` | Equal weight per gold-supported class in the evaluated positions, including the unknown class. |
| Macro F1 excluding `<UNK>` | First removes unknown-gold positions, then averages across retained classes present in the remaining gold labels. This is not an average over every possible vocabulary class. |
| Weighted F1 | Uses gold class support as weights; both including-unknown and retained-only variants are saved. |
| Chord-onset accuracy | Accuracy at the first grid position and mapped-label changes; an excluding-unknown variant is also saved. |
| Cross-entropy | Mean unweighted negative log-likelihood at supervised positions. |
| Perplexity | `exp(cross_entropy)`; the implementation caps the exponent at 50 for numerical stability. |
| Normalised cross-entropy | Divides CE by `log(classifier output count)`. That count includes the padding output; it is the implemented uniform-output reference. |
| Unknown and invalid-pad rates | Reveal how often a prediction is unusable as a retained chord. |
| Majority baseline | Always predicts the most frequent training chord, evaluated on the test labels. |

Top-200 and Top-250 define different label spaces and unknown-label coverage. Their perplexities alone do not establish which model is better. Coverage, retained-only metrics, onset accuracy and class balance must be read together. Choosing one final method after inspecting test results can also make that test set exploratory; a further untouched evaluation set would strengthen a final selection claim.

## Causality and latency checks

The notebooks implement two checks:

1. Change inputs after a cutoff and confirm that earlier logits are unchanged within tolerance.
2. Compare full-sequence evaluation with chunked evaluation that carries LSTM state.

These checks assess architectural causality and state handling. They do not prove the absence of every possible data leak or establish subjective musical quality. A model can be causal and still predict poor chords.

The latency benchmark measures repeated model calls with device synchronisation where appropriate. It excludes event acquisition, tempo tracking, network transport, audio rendering and the scheduling of accompaniment changes. Saved benchmark outputs are not included in the repository.
