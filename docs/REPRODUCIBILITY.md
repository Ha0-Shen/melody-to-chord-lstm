# Reproducibility and execution

## What is provided

Three notebooks, dependency specifications, synthetic checks and documentation are included. The original notebook metadata identifies Python 3.12. The supplied README reports Google Colab with an NVIDIA Tesla T4; the archive does not contain a complete environment lock, the original ChoCo commit, a checkpoint or preserved execution outputs.

Accordingly, the dependency file is an installation specification, not an exact reconstruction of the historical environment. The primary music packages retain their submitted versions; PyTorch is not pinned because its historical version is unknown. Match the PyTorch build to the intended CPU/CUDA environment. Local checks for this portfolio copy are recorded in `docs/VALIDATION.md` once run.

## Installation

Use Python 3.12 and Git. From the repository root:

```bash
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m jupyterlab
```

On macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m jupyterlab
```

The notebook `%pip` cells install their music/analysis dependencies. Colab normally supplies PyTorch; a local environment should install the repository requirements first. If installation changes an already imported package, restart the kernel before continuing.

## Storage and data

Local defaults place generated artifacts in `artifacts/` and the sparse ChoCo checkout in `data/choco/`, relative to the repository root. Both directories are ignored by Git. In Colab, the default artifact path is `/content/drive/MyDrive/melody-to-chord/artifacts`; set an override to reuse existing experiment files.

Optional environment variables, set **before** running configuration cells:

| Variable | Purpose |
| --- | --- |
| `XML_ONLY_OUTPUT_ROOT` | Common artifact root for all three notebooks. |
| `CHOCO_DATA_DIR` | Location of the ChoCo Git checkout. |
| `WIKIFONIA_RAW_DIR` | Existing MXL directory; skips the automatic raw-partition download. A converter checkout may still be downloaded. |

In a notebook, for example:

```python
import os
os.environ['XML_ONLY_OUTPUT_ROOT'] = '/absolute/path/to/existing/artifacts'
```

Use `git --version` to check that Git is available. The first notebook sparse-clones `https://github.com/smashub/choco.git` and selects the Wikifonia partition and converter utilities. It writes a provenance JSON recording the available checkout commit and package versions. The default download follows the current upstream branch; it is not pinned to the original dissertation's data snapshot. A supplied custom raw directory can come from a separate source, in which case its revision needs to be recorded separately.

The converter loader conditionally removes an unused `lib2to3` import from its local upstream checkout for compatibility. It does not vendor the converter into this repository.

## Execution sequence

Use the same artifact root throughout. Run each notebook from top to bottom and restart its kernel between experimental conditions.

| Step | Notebook | Configuration | Expected output |
| ---: | --- | --- | --- |
| 1 | `01_Wikifonia_Preprocessing_and_TopN_Encoding.ipynb` | `TOP_N_RETAINED_CHORDS = 200` | `encoding_miditok_tsd24_top200_unk_v3/` |
| 2 | Same preprocessing notebook | `TOP_N_RETAINED_CHORDS = 250` | `encoding_miditok_tsd24_top250_unk_v3/` |
| 3 | `02_Causal_LSTM_Top200_Top250_Baseline.ipynb` | `ACTIVE_TOP_N = 200` | `causal_lstm_tsd24_top200_unk_v3_audited/` |
| 4 | Same baseline notebook | `ACTIVE_TOP_N = 250` | `causal_lstm_tsd24_top250_unk_v3_audited/` and Top-N comparison |
| 5 | `03_Causal_LSTM_Top200_ClassWeighted.ipynb` | Fixed to Top-200 | `causal_lstm_tsd24_top200_unk_v3_classweighted_audited/` and loss comparison |

Preprocessing can reuse completed batches when its existing markers match. Top-200 and Top-250 encodings are now written to separate directories automatically. The training notebooks also recognise older generic encoding directories when their summaries match the requested vocabulary.

Training preflight deliberately expects 5,022 songs and a base MIDITok vocabulary of 1,874 entries. These checks preserve the submitted experiment interface. If current upstream data or package behaviour changes the counts, inspect the generated manifests and versions before changing the guard; accepting a different corpus is a new experiment, not a reproduction of the original run.

## Controlled settings

| Setting | Value |
| --- | --- |
| Random seed | 42 |
| PPQ | 24 |
| Vocabulary ranking | Training supervised-slot frequency |
| Training/evaluation token windows | 512 |
| Training hop | 256 |
| Validation/test hop | 512 |
| Minimum training window | 128 tokens |
| Embedding dimension / hidden size | 128 / 128 |
| LSTM layers / direction | 2 / unidirectional |
| Dropout | 0.30 |
| Batch size | 16 |
| Maximum epochs / early-stopping patience | 20 / 4 |
| Learning rate / weight decay | 0.001 / 0.0001 |
| Gradient clip | 1.0 |
| Checkpoint selection | Lowest unweighted validation CE |
| Evaluation | Stateful by song, non-overlapping windows |

GPU kernels and package changes can affect exact numerical repeatability even when seeds are fixed. CPU execution is supported in the code but can be slow. Dataset and window artifacts are loaded into memory; a machine with limited RAM may need changes, which should be recorded as a new experimental configuration.

## Quick verification

```bash
python scripts/verify_repository.py
python -m unittest discover -s tests -v
python scripts/smoke_test_model.py
```

The first command checks notebook structure, syntax and relative Markdown links with the standard library. The tests use NumPy and synthetic note data to check releases, leakage boundaries, splits, vocabulary selection and evaluation tails. The final command requires PyTorch and directly loads the model class from each training notebook to check future invariance, streaming equivalence and a finite training update. It uses randomly initialised weights, not a trained accompaniment model.

These checks do not execute the full corpus or demonstrate training convergence. The GitHub workflow runs structural and synthetic-data checks, not full training or the PyTorch smoke test.

## Publishing results

Once all three full runs finish, inspect the reports and provenance, then run:

```bash
python scripts/collect_results.py --artifact-root /absolute/path/to/artifacts
```

This copies selected metrics/configuration fields into `results/metrics_summary.json` and writes `results/comparison.md`. It refuses partial or batch-limited runs and checks the controlled configurations and split identities. It does not copy raw scores, model weights, examples or personal absolute paths. Review the generated files and commit them explicitly.
