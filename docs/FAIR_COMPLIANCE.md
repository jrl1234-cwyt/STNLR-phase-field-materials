# FAIR release checklist

## Findable

- `CITATION.cff` provides title, authors, version, keywords, and release date.
- A Zenodo DOI must be added after the GitHub release is archived.

## Accessible

- Source and lightweight metadata can be served by GitHub.
- Binary datasets and checkpoints are configured for Git LFS and should also be deposited on Zenodo.
- External pretrained weights have a documented download route and checksum.

## Interoperable

- Results and metadata use JSON, tabular text, YAML, CFF, and standard NumPy/PyTorch formats.
- Dataset field names, shapes, dtypes, and parameters are inventoried in `data/MANIFEST.json`.

## Reusable

- Code, data provenance, random seeds, environment locks, paired checkpoints, and checksums are included.
- The package has an explicit license and distinguishes external dependencies.
- Training, evaluation, uncertainty, and solver-validation programs are supplied.

The package is technically ready for a public versioned release. FAIR completion requires replacing the repository/DOI metadata after upload and confirming that the chosen license is authorized for every redistributed component.
