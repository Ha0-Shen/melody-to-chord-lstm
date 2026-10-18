# Changes from the dissertation supplementary archive

The source is `HaoShen_COMP702_Supplementary_Material.zip`, supplied by the project author. The original archive contains three notebooks, a README and requirements. Its notebooks have no saved outputs or execution counters.

This public copy retains the original research implementation, with the following packaging and configuration edits:

1. Move the three notebooks into `notebooks/`, retaining their filenames.
2. Add a public-copy notice, stable cell IDs and clean notebook metadata.
3. Make local storage relative to the repository, support an output-root override in Colab, and replace the private Drive folder with a generic default.
4. Support optional existing raw-data and ChoCo checkout locations. Replace fixed `/content/...` and `/home/...` candidate directories with configured paths. Verify the ChoCo origin before applying sparse-checkout operations, so an unrelated parent repository is not modified.
5. Write Top-200 and Top-250 encodings to separate directories, preventing one vocabulary run from overwriting the other.
6. Set the baseline notebook's initial `ACTIVE_TOP_N` to 200 to match the documented execution order; 250 remains the second condition.
7. Record the available ChoCo checkout commit and preprocessing package versions in a new provenance file. This does not reconstruct the original commit.
8. Replace stale cell-number rerun instructions with configuration-based instructions.
9. Add public documentation, repository exclusions, structural checks, synthetic behavioural tests and an optional model smoke test.

The public-copy notice adds one cell, so later cell numbers differ from the submission. Original notebook SHA-256 hashes and the indices of intentionally edited source cells are retained in `submission_sources.json`.

The model architecture, training losses, evaluation implementation, split function, vocabulary ranking and scientific hyperparameters were not changed. No dataset preprocessing or model training was rerun to create this release. Do not treat its source checks as reproduced predictive results.
