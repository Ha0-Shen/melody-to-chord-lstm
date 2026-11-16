# Melody-to-Chord Prediction with a Causal LSTM

[![Checks](https://github.com/Ha0-Shen/melody-to-chord-lstm/actions/workflows/checks.yml/badge.svg)](https://github.com/Ha0-Shen/melody-to-chord-lstm/actions/workflows/checks.yml)

This was my MSc dissertation project in Data Science and Artificial Intelligence at the University of Liverpool (2026). I explored whether a model could predict the current chord from a symbolic melody without using notes from the future.

**In short**

- **Data:** 6,672 MusicXML lead sheets attempted, 5,022 kept, 1,650 rejected with a logged reason.
- **Result:** 34.0% slot accuracy and 54.9% Top-3 accuracy on 508 held-out songs, against a 9.4% majority-class baseline.
- **Causality:** changing later inputs does not change earlier outputs, and chunked inference matches full-sequence inference.
- **Speed:** about 0.54 ms per step for the model alone on an NVIDIA T4 GPU.

The project covers the full notebook workflow: reading MusicXML lead sheets, checking and aligning melody and chord data, preparing model inputs, training LSTM classifiers and evaluating the predictions.

**Python · PyTorch · pandas · NumPy · music21 · MIDITok · scikit-learn**

## Approach

The source data comes from the Wikifonia partition of the [ChoCo dataset](https://github.com/smashub/choco). Melody notes and chord symbols are extracted from the same MusicXML part and placed on a 24-PPQ time grid.

The main modelling constraint is causality. At note-on time, a live system knows the pitch but not how long the note will last. I therefore converted duration events into `NOTE_OFF` events and used a unidirectional LSTM whose hidden state can be carried between consecutive input windows. Chord labels are targets only and never appear in the model input.

I compared three settings:

| Experiment | Chord vocabulary | Loss |
| --- | ---: | --- |
| Baseline | Top 200 | Cross-entropy |
| Baseline | Top 250 | Cross-entropy |
| Class-weighted | Top 200 | Square-root inverse-frequency weighting |

Labels outside the training-derived vocabulary are mapped to `<UNK>`. `<PAD>` is used only for padding, while `N` remains a valid no-chord label.

## Results

These are the figures reported in my dissertation, from runs on Google Colab with an NVIDIA T4 GPU.

### Data

| Stage | Files |
| --- | ---: |
| MusicXML files attempted | 6,672 |
| Passed validation | 5,022 |
| Rejected: ambiguous chord annotation | 1,098 |
| Rejected: no part with both melody and chords | 434 |
| Rejected: overlapping melody notes | 109 |
| Rejected: other parsing errors | 9 |

I rejected ambiguous files instead of repairing them, because a wrong melody-to-chord alignment would have produced misleading training labels. The 5,022 songs were split by song group into 3,979 training, 535 validation and 508 test songs, with no group appearing in more than one split.

### Held-out test set (508 songs)

| Metric | Top-200 | Top-250 | Top-200 class-weighted |
| --- | ---: | ---: | ---: |
| Slot accuracy | **0.340** | 0.338 | 0.306 |
| Top-3 accuracy | **0.549** | 0.546 | 0.501 |
| Macro F1 (excluding `<UNK>`) | 0.050 | 0.041 | **0.063** |
| Weighted F1 (excluding `<UNK>`) | **0.298** | 0.294 | 0.294 |
| Chord-onset accuracy (excluding `<UNK>`) | **0.223** | 0.217 | 0.193 |
| Cross-entropy (lower is better) | **2.750** | 2.785 | 2.970 |
| Perplexity (lower is better) | **15.64** | 16.20 | 19.49 |

The majority-class baseline has a slot accuracy of 0.094.

What I took from this:

- Top-200 beat Top-250 on every predictive metric, even though Top-250 covers slightly more of the training labels.
- Class weighting raised macro F1 but lowered accuracy, Top-3 accuracy and onset accuracy, so I kept the unweighted Top-200 model.
- Macro F1 stays low in every setting. The model favours frequent chords and is weaker where the chord changes. This is the main limitation of the work.
- Getting the data representation, alignment and split right mattered at least as much as the model architecture.

### Causality and speed

- Chunked inference with carried state matched full-sequence inference to within 9.54e-6 (tolerance 1e-4).
- Median model-only step time was about 0.542 ms on a T4 GPU, with a 95th percentile of about 0.858 ms. This does not include MIDI input, audio scheduling or a user interface.

## Repository contents

| File | Description |
| --- | --- |
| [01_Wikifonia_Preprocessing_and_TopN_Encoding.ipynb](notebooks/01_Wikifonia_Preprocessing_and_TopN_Encoding.ipynb) | Extracts the MusicXML data, runs quality checks, creates the train/validation/test split and prepares Top-200 or Top-250 targets. |
| [02_Causal_LSTM_Top200_Top250_Baseline.ipynb](notebooks/02_Causal_LSTM_Top200_Top250_Baseline.ipynb) | Trains and evaluates the two unweighted baseline models. |
| [03_Causal_LSTM_Top200_ClassWeighted.ipynb](notebooks/03_Causal_LSTM_Top200_ClassWeighted.ipynb) | Repeats the Top-200 experiment with a class-weighted loss. |
| [METHOD.md](docs/METHOD.md) | More detail on alignment, model design and evaluation. |
| [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Setup, paths and execution order. |

## Data preparation

The preprocessing notebook:

- selects a lead-sheet part containing both melody and chord symbols;
- expands repeats and consolidates tied notes;
- rejects ambiguous chord states and overlapping melody lines;
- audits quantisation at 12 and 24 PPQ;
- keeps filename variants in the same data split; and
- selects the chord vocabulary from the training split only.

The checked-in settings use 128-dimensional embeddings, two unidirectional LSTM layers with a hidden size of 128, dropout of 0.30 and 512-token windows. Training windows overlap by 256 tokens. Validation and test windows are processed in sequence, carrying the LSTM state within each song.

## Evaluation

The notebooks calculate slot accuracy, Top-3 accuracy, macro and weighted F1, chord-onset accuracy, cross-entropy, perplexity and unknown-label rates. They also include two checks for causal behaviour:

1. changing later inputs must not change earlier outputs;
2. chunked inference with carried state should match full-sequence inference.

Notebook outputs and trained checkpoints are not included in the repository; the figures above are summarised in [results/README.md](results/README.md). The notebooks save the metrics and figures when run, and [`scripts/collect_results.py`](scripts/collect_results.py) can combine completed runs into a comparison table.

## Running the project

The notebooks were written for Python 3.12 and Google Colab. For a local environment:

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS/Linux, then run:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/verify_repository.py
python -m unittest discover -s tests -v
python -m jupyterlab
```

The full notebook order and data paths are described in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Full training is best run with a GPU.

## Limitations

- A melody can support more than one reasonable accompaniment, so matching one annotated chord sequence does not fully measure musical quality.
- Grouping by normalised filenames prevents known score variants from crossing splits, but does not identify every possible duplicate arrangement.
- Rare chords are merged into `<UNK>`, which loses some harmonic detail.
- The latency code measures the PyTorch model only; it does not cover MIDI input, audio scheduling or a user interface.

## Data and licences

Wikifonia files and chord-conversion utilities are obtained from [ChoCo](https://github.com/smashub/choco). Raw scores, generated datasets and trained weights are excluded from Git. See [THIRD_PARTY.md](THIRD_PARTY.md) for the upstream data and licence information.

**Author:** Hao-Shen Yang
