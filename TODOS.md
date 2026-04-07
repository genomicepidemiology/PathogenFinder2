# PathogenFinder2 — Outstanding TODOs

This file tracks known incomplete or placeholder items in the codebase.

---

## CLI wiring for standalone align/map/test modes

The following features are **fully implemented as class methods** but are **not yet exposed via CLI subparsers**:

- `PathogenFinder2.align_proteins()` — align attention-highlighted proteins to a protein database
- `PathogenFinder2.map_embeddings()` — map proteome embeddings to the Bacterial Pathogenic Landscape
- `PathogenDLModel.test_model()` — evaluate a trained model on labelled data

All three work when invoked programmatically or through the `predict` CLI action (align/map are triggered by `--attProteins align` and `--embedProteome map` flags). To use them as standalone CLI modes, subparsers need to be added to `cli.py` and dispatch logic to `main()`.

---

## Software result metadata not automatically updated

`CGEResults.add_software_result` ([src/pathogenfinder2/utils/output.py](src/pathogenfinder2/utils/output.py))
contains `# TODO UPDATE AUTOMATIC`.  Several fields (`software_log`, `databases`, etc.) are set to
empty defaults and are not populated from runtime state automatically.  These should be filled
in from the pipeline run context rather than left blank.

---

## Software execution metadata incomplete

`CGEResults.add_software_exec` contains `# TODO`.  The `parameters` dict passed in is a shallow
snapshot of the CLI args; it should be extended to capture the full runtime configuration (model
version, embedding model path, etc.) for complete reproducibility.

---

## Prodigal multi-sequence file support unverified

`ProdigalRunner` ([src/pathogenfinder2/preprocessing/prodigal.py](src/pathogenfinder2/preprocessing/prodigal.py))
contains `# TODO: Check if multiple sequence in one file also works`.  Multi-FASTA genomes (more
than one contig per file) need to be validated end-to-end.
