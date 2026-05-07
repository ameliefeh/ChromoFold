# Fluorescent-Protein GNN

Predicts brightness and emission wavelength of a fluorescent protein from
its 3D structure (PDB) plus its molecular weight (kDa).

Two switchable architectures:
- **A (disjoint)** — protein and chromophore graphs encoded independently, fused at graph level.
- **B (merged)** — same plus residue↔chromophore-atom cross-edges (≤ 6 Å) carrying messages between the two graphs.

See `docs/superpowers/specs/2026-04-29-fp-gnn-pipeline-design.md` for the design.

## Setup

```bash
uv sync
```

## Train

```bash
# Architecture A (disjoint baseline)�
# Architecture B (merged with cross-edges)
uv run python -m fp_gnn.train --model b --max-epochs 60
```

CSV logs land in `logs/model_<a|b>/version_*/metrics.csv`.

## Test

```bash
uv run pytest -v
```

## Data

- `data/raw/*.pdb` — PDB structures (currently 7ZCT and 4OQW).
- `data/labels.csv` — per-protein scalars and targets, with a `split` column.
- `data/processed/` (gitignored) — cached PyG dataset after first run.
- `data/ccd_cache/` (gitignored) — cached CCD SMILES per chromophore code.
