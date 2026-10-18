# Melody-to-Chord Prediction with a Causal LSTM

This was my MSc project in Data Science and Artificial Intelligence at the University of Liverpool. I explored whether a model could predict the current chord from a symbolic melody without using notes from the future.

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

Notebook outputs and trained checkpoints are not included in the repository. The notebooks save the metrics and figures when run, and [`scripts/collect_results.py`](scripts/collect_results.py) can combine completed runs into a comparison table.

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

**Author:** Hao Shen Yang
