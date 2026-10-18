# Data and third-party acknowledgements

## ChoCo and Wikifonia

- Repository: https://github.com/smashub/choco
- Upstream licence statement: https://github.com/smashub/choco/blob/main/LICENSE.md
- Wikifonia partition: https://github.com/smashub/choco/tree/main/partitions/wikifonia

The project downloads the Wikifonia source partition and imports ChoCo chord-conversion utilities at runtime. Upstream describes its data and code as CC BY 4.0, with specified other collections under CC BY-NC-SA 4.0. Refer to the upstream statement and collection provenance for the material actually used. This repository does not redistribute the raw scores or those utilities, and does not claim ownership of the underlying compositions.

The conditional removal of an unused `lib2to3` import modifies only the local downloaded converter copy. A runtime provenance file records this compatibility step and the available checkout revision.

## Libraries

The notebooks depend on music21, MIDITok, symusic, harte-library, Lark, PyTorch, scikit-learn, pandas, NumPy, Matplotlib and tqdm. JupyterLab/IPython provide the notebook interface. Each dependency retains its own licence; see its package distribution and upstream project.

## This project's code

No software licence has been added to this repository. Public visibility does not grant permission to redistribute the project code.
