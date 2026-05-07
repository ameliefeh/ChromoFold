# Fluorescent-Protein GNN Pipeline — Design

**Date:** 2026-04-29
**Status:** Approved (pending user review of this written spec)

## 1. Goal

Predict two scalar fluorescence properties of a fluorescent protein from its 3D
structure:

- **Brightness** (≈ EC × QY)
- **Emission wavelength** (nm)

Inputs: a `.pdb` file plus a per-protein scalar (kDa) from `data/labels.csv`.
Approach: build two graphs per sample (one for the protein backbone, one for
the bound chromophore), encode each with a Message-Passing Neural Network
(MPNN), fuse the two graph-level embeddings with kDa, and run an MLP head to
produce a 2-D prediction.

For this iteration we have **two PDBs** (`7ZCT_mScarlet3.pdb`,
`4OQW_mCardinal.pdb`). The deliverable is a runnable training loop on this
pair, ready to swap in the full ~100-PDB dataset later.

**Two model architectures are implemented for direct comparison:**

- **A — Disjoint two-graph baseline (friend's design).** Protein and
  chromophore are encoded by separate MPNNs; their graph-level embeddings
  are concatenated with kDa and fed to the head. Cannot model
  chromophore-protein interactions directly.
- **B — Merged graph with cross-edges.** Same protein and chromophore
  graphs *plus* a third edge type connecting residues whose Cα is within
  6 Å of a chromophore atom. Messages flow between protein and chromophore
  during message passing, so the model can learn local environment effects
  (H-bonds, π-stacking, electrostatics) that drive emission wavelength.

A switchable config flag selects between A and B. Everything except the
inner message-passing wiring is shared, so the comparison isolates exactly
the architectural difference.

## 2. Non-goals

- Not a production-ready package — class-project / M2 style: plain
  functions returning tuples or arrays, modules organized by purpose but
  not over-modularized, only the dataclass-like structures PyG actually
  needs (the `FPData(Data)` subclass for batching). Mirrors the ipynb in
  spirit.
- No hyperparameter search, no cross-validation, no W&B integration.
- No symmetry-mate handling, no NMR multi-model ensembles.
- No prediction of QY / Stokes shift / EC alone.

## 3. Pipeline overview

```
.pdb ──► pdb_io ──► (residue_names, ca_coords)         ─► protein_graph
                ─► chromophore PDB block                ─► chromophore_graph
                                                           (RDKit Mol via CCD template)
                ─► residue/atom xyz                     ─► cross-edges (residue ↔ atom, ≤ 6 Å)

(protein_graph + chrom_graph + cross_edges + kDa + y) ─► FluorProteinDataset
                                                          (InMemoryDataset, cached)

────────── Architecture A (disjoint baseline) ──────────
              ┌────────────────────┐
   protein  ─►│   ProteinMPNN      │─► pool ┐
              └────────────────────┘        │
                                            ├─► concat ─► [+kDa] ─► MLP ─► [b*, λ*]
              ┌────────────────────┐        │
   chrom    ─►│   ChromMPNN        │─► pool ┘
              └────────────────────┘

────────── Architecture B (merged with cross-edges) ────
                          ┌────────────────────┐
   protein + chrom + ────►│ FPNetMerged        │─► pool(res) ┐
   cross-edges            │ (NNConv per edge   │             ├─► concat ─► [+kDa] ─► MLP ─► [b*, λ*]
                          │  type, incl. cross)│─► pool(atom)┘
                          └────────────────────┘
```

Both architectures share: graph builders, dataset, pooling shape, head, loss,
optimizer. Only the message-passing layout inside differs.

## 4. Repository layout

```
pdb_project_2/
├── pyproject.toml              # uv-managed
├── data/
│   ├── labels.csv              # existing
│   ├── raw/                    # PDB files (existing)
│   ├── processed/              # cached PyG graphs (.pt) — gitignored
│   └── ccd_cache/              # cached CCD SMILES per chromophore code — gitignored
├── src/fp_gnn/
│   ├── __init__.py
│   ├── pdb_io.py               # PDB parsing helpers
│   ├── chromophore_graph.py    # RDKit/CCD → PyG Data + heavy-atom xyz
│   ├── protein_graph.py        # residue contact graph + cross-edges
│   ├── dataset.py              # FluorProteinDataset (InMemoryDataset)
│   ├── model.py                # ProteinMPNN, ChromMPNN, FPNetA, FPNetB
│   └── train.py                # Lightning module + train loop + z-score helpers
├── tests/                      # pytest sanity tests
├── docs/superpowers/specs/     # this design doc
└── README.md
```

## 5. Tooling

- **Environment:** `uv` for venv + dep management. `pyproject.toml` declares
  deps.
- **Core deps:** `torch`, `torch_geometric`, `rdkit`, `ogb`, `pytorch_lightning`,
  `numpy`, `pandas`, `requests`, `pytest`.
- **Logging:** Lightning's `CSVLogger` + TensorBoard. **No Weights & Biases.**
- **Reproducibility:** `torch.manual_seed(0)`, `np.random.seed(0)`,
  `random.seed(0)` at process start.

## 6. Data ingestion (`pdb_io.py`)

Three small functions, no wrapper dataclass — close to the ipynb in style.

```python
def get_chromophore_code(pdb_path: Path) -> str:
    """First residue name in HETATMs that matches CHROMOPHORE_CODES."""

def get_chromophore_pdb_block(pdb_path: Path, residue_code: str) -> str:
    """All HETATM lines for the chromophore in chain A, as a PDB block string."""

def get_protein_residue_ca(pdb_path: Path) -> tuple[list[str], np.ndarray]:
    """
    Parse ATOM records (chain A, altloc ' ' or 'A' only), pick the Cα atom
    of each residue, skip HOH.
    Returns:
        residue_names: list[str], length N
        ca_coords:     np.ndarray of shape (N, 3)
    """
```

**Strict filters** (per user direction):
- Chain A only — no first-chain fallback.
- AltLoc ' ' or 'A' only — discard B/C/etc.
- Skip waters (HOH) entirely.
- Skip ions / buffers (HETATMs that aren't a known chromophore code).

**`CHROMOPHORE_CODES` constant:** lifted verbatim from the friend's ipynb —
the 29-code set `{'NRQ', 'CRQ', 'NRP', 'CH6', 'CRO', '5SQ', '4M9', 'CR2',
'OFM', 'CR8', 'CFY', 'OIM', 'CH7', 'GYS', 'WCR', 'GYC', 'DYG', 'FAD', 'PIA',
'CCY', 'BLR', 'CRF', 'NYG', 'CR7', 'FMN', 'B2H', 'SWG', 'CSH', 'BJF'}`. Lives
in `pdb_io.py`.

**Position convention:** **Cα coordinates** as the residue position, not
center-of-mass. Always present, reproducible across PDBs, matches contact-graph
literature conventions (8 Å Cα–Cα).

**Failure modes:** raise on bad input (no chromophore, no chain A);
warn-and-skip on rare cases (multi-copy chromophore → take first by resseq;
residue missing Cα → skip).

## 7. Graph construction

### 7a. Chromophore graph (`chromophore_graph.py`)

Lifted from the friend's ipynb skeleton, with two principled fixes so it
plugs into OGB's encoders correctly.

```python
def build_chromophore_graph(pdb_path: Path) -> Data:
    code  = get_chromophore_code(pdb_path)
    block = get_chromophore_pdb_block(pdb_path, code)
    smiles = get_ccd_smiles_cached(code)              # caches under data/ccd_cache/
    template = Chem.MolFromSmiles(smiles)
    raw_mol  = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    mol      = AllChem.AssignBondOrdersFromTemplate(template, raw_mol)
    mol      = Chem.RemoveHs(mol)                     # heavy-atom-only graph
    Chem.SanitizeMol(mol)
    return mol_to_graph(mol)
```

**Node features (9-int OGB schema)** — produced by
`ogb.utils.features.atom_to_feature_vector(rdkit_atom)`:

| # | Field |
|---|---|
| 1 | atomic_num |
| 2 | chirality |
| 3 | degree |
| 4 | formal_charge |
| 5 | num_hs |
| 6 | num_radical_electrons |
| 7 | hybridization |
| 8 | is_aromatic |
| 9 | is_in_ring |

Stored as `x: torch.long` of shape `[N, 9]`. The four fields the friend
defined (atom_type, degree, formal_charge, hybridization) are a strict subset
of these. `xyz` is **not** a node feature — 3D info lives on edges as
distances.

**Edge features:**
- `edge_attr_chem: torch.long` of shape `[E, 3]` — `bond_to_feature_vector`
  output (bond_type, stereo, conjugated).
- `edge_attr_dist: torch.float` of shape `[E, 1]` — Euclidean bond distance
  in Å.

Both edges (i→j) and (j→i) are emitted (undirected graph encoded as
bidirectional).

**Heavy-atom positions** are stored in the standard PyG `pos` attribute
(`pos: torch.float [N_atom, 3]`), read from the RDKit conformer. The
cross-edge builder (Section 7c) consumes this `pos` to compute residue–atom
distances.

**CCD SMILES caching:** `get_ccd_smiles_cached(code)` reads/writes
`data/ccd_cache/<code>.json`. Hits the RCSB API once per chromophore type per
machine.

### 7b. Protein graph (`protein_graph.py`) — new code

```python
def build_protein_graph(
    residue_names: list[str],
    ca_coords: np.ndarray,         # (N, 3)
    cutoff: float = 8.0,
) -> Data:
    ...
```

**Nodes:** one per residue (waters already dropped in parsing).
**Node features:** 20-D one-hot of standard amino acids,
`x: torch.float` of shape `[N, 20]`. Non-canonical residues encode as all-zero
+ warning.

**Edges:** all residue pairs `(i, j)` with `‖Cα_i − Cα_j‖ ≤ 8.0 Å`, no
self-loops. Built from the full pairwise distance matrix (`N²`; trivial at
N≈230). Both directions emitted.

**Edge features:** `edge_attr: torch.float` of shape `[E, 1]` — the raw Cα–Cα
distance. (Optional Gaussian-RBF expansion is a future tweak.)

### 7c. Cross-edges (residue ↔ chromophore atom) — for architecture B

A single function in `protein_graph.py` (no separate module). Built once at
dataset construction and stored alongside both graphs. The disjoint
baseline (A) ignores them; the merged model (B) uses them as a third edge
type.

```python
def build_cross_edges(
    ca_coords: np.ndarray,           # (N_res, 3)
    chrom_coords: np.ndarray,        # (N_atom, 3) — heavy atoms of the chromophore
    cutoff: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (cross_edge_index, cross_edge_attr).
    cross_edge_index: long [2, E] — row 0 residue indices, row 1 chrom-atom indices
    cross_edge_attr:  float [E, 1] — Euclidean distance Å
    """
```

**Cutoff: 6.0 Å** between residue Cα and any chromophore heavy atom. Rationale:
≈5 Å covers direct H-bonds and contacts; the extra 1 Å picks up second-shell
residues that still tune the chromophore environment without exploding the
edge count.

**Direction:** stored as residue → atom (row 0 → row 1). Model B emits both
directions internally via two bipartite NNConvs (residue→atom, atom→residue).

**Chromophore atom positions:** read from the RDKit conformer used to build
the chromophore graph (heavy atoms only — same set, same indexing).

## 8. Dataset (`dataset.py`)

`FluorProteinDataset(InMemoryDataset)` follows the friend's `ESOLGraphData`
pattern (cache to `data/processed/data.pt`, build once via `process()`).

**Item type — `FPData(Data)`.** `InMemoryDataset` expects items to be PyG
`Data` (or subclasses), so we use a thin `Data` subclass that stores both
graphs in one object via prefixed attributes:

```python
class FPData(Data):
    """
    Holds protein graph + chromophore graph + cross-edges in one Data object.

    Protein graph (PyG defaults — no prefix):
        x, edge_index, edge_attr           # 20-D one-hot, [2, E_p], [E_p, 1]
    Chromophore graph (chrom_ prefix):
        chrom_x, chrom_edge_index,
        chrom_edge_attr_chem,              # long [E_c, 3] for BondEncoder
        chrom_edge_attr_dist               # float [E_c, 1] for distance MLP
    Cross-edges (residue → chromophore atom, used by Architecture B only):
        cross_edge_index,                  # long [2, E_x]; row 0 residue, row 1 atom
        cross_edge_attr                    # float [E_x, 1] distance
    Scalars:
        kda                                # float scalar
        y                                  # [2]: [brightness, emission] (raw)
        pdb_code                           # str
    """

    def __inc__(self, key, value, *args, **kwargs):
        # Tell PyG how to offset edge indices when batching.
        if key == 'chrom_edge_index':
            return self.chrom_x.size(0)
        if key == 'cross_edge_index':
            # Row 0 indexes residues (offset by N_res = x.size(0));
            # row 1 indexes chromophore atoms (offset by N_atom = chrom_x.size(0)).
            return torch.tensor([[self.x.size(0)],
                                 [self.chrom_x.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)
```

This makes PyG's standard `DataLoader` Just Work — `__inc__` tells it how to
renumber both `chrom_edge_index` and the bipartite `cross_edge_index` per
batch. No custom collate function needed.

**`process()`** iterates rows in `data/labels.csv`, builds both graphs, packs
into `FPData`, calls `self.collate(...)` and saves under `data/processed/`.

**Train/test split:** add a `split` column to `labels.csv`. For now:
`7ZCT → train`, `4OQW → test`. Swap to a `RandomSplitter` (like the friend's)
when the dataset grows.

## 9. Model (`model.py`)

```
ChromMPNN(batch)   -> [B, H]   # reads chrom_x, chrom_edge_index, chrom_edge_attr_*
ProteinMPNN(batch) -> [B, H]   # reads x, edge_index, edge_attr
FPNetA(batch)      -> [B, 2]   # disjoint baseline; reads batch.kda_z attached by LitModule
FPNetB(batch)      -> [B, 2]   # merged graph with cross-edges (residue ↔ atom)
```

`FPNetA` and `FPNetB` are **picked at construction time** by a config flag.
They share the head, hidden size, optimizer, loss, and Lightning wrapper —
the only difference is the inner message-passing wiring.

### 9a. ChromMPNN

Direct adaptation of the friend's `MPNN` skeleton (`AtomEncoder` /
`BondEncoder` / `NNConv` / `GRU` / `global_add_pool`), with:

- `AtomEncoder(emb_dim=H)` over the 9-int atom features. Works as-is once the
  feature schema matches OGB.
- `BondEncoder(emb_dim=H)` over the 3-int bond chemistry features.
- A **distance side channel** for bond distance:
  ```python
  e_chem = self.bond_emb(edge_attr_chem)              # [E, H]
  e_dist = self.dist_proj(edge_attr_dist)             # [E, H], Linear(1, H)
  edge_emb = e_chem + e_dist                          # [E, H]
  ```
- Same `NNConv(in=H, out=H, nn=edge_network, aggr="mean")` + `GRU` update +
  `num_message_steps=3` as the friend.
- `global_add_pool` → `[B, H]`.

### 9b. ProteinMPNN

Same structure, simpler features:
- Node encoder: `nn.Linear(20, H)` over the 20-D residue one-hot.
- Edge encoder: `nn.Linear(1, H)` over the raw Cα–Cα distance.
- Same `NNConv` + `GRU` + 3 message steps + `global_add_pool` → `[B, H]`.

(The two MPNN classes share the message-passing core; `ChromMPNN` and
`ProteinMPNN` are thin wrappers around it differing only in the input
encoders.)

### 9c. FPNetA — disjoint two-graph baseline

The friend's design. Receives a unified batch, runs the two MPNNs
independently, fuses graph-level. The LightningModule (Section 10) is the
single owner of normalization; `FPNetA` is normalization-agnostic.

```python
def forward(self, batch):
    prot_emb  = self.protein_mpnn(batch)            # [B, H], reads x/edge_index/edge_attr
    chrom_emb = self.chrom_mpnn(batch)              # [B, H], reads chrom_*
    fused     = torch.cat([prot_emb, chrom_emb,
                           batch.kda_z.unsqueeze(-1)], dim=-1)  # [B, 2H+1]
    out_z     = self.head(fused)                    # [B, 2]
    return out_z
```

**Head:** 2-layer MLP `(2H+1) → H → 2`, ReLU between.
**Hidden size:** `H = 64` to start (matches the friend's default).

### 9d. FPNetB — merged graph with cross-edges

Same encoders, same head, same pooling, same hidden size as A. The single
difference is **how messages flow during the 3 message-passing steps**:
`FPNetB` adds bipartite NNConv layers over the cross-edges so residues and
chromophore atoms exchange messages.

**Encoders** (identical to A, plus one for the cross-edge):
- Residue node: `nn.Linear(20, H)`
- Atom node: `AtomEncoder(emb_dim=H)`
- Contact edge (residue–residue): `nn.Linear(1, H)` over distance
- Bond edge (atom–atom): `BondEncoder(emb_dim=H)` + `nn.Linear(1, H)` for distance, summed
- **Cross edge (residue–atom):** `nn.Linear(1, H)` over distance — *new*

**Message-passing layers** (used in each of the 3 steps):
- `contact_conv: NNConv(H, H, …)` — residue → residue
- `bond_conv: NNConv(H, H, …)` — atom → atom
- `cross_conv_r2a: NNConv((H, H), H, …)` — residue → atom (bipartite)
- `cross_conv_a2r: NNConv((H, H), H, …)` — atom → residue (bipartite)
- `gru_res: GRU(H, H)`, `gru_atom: GRU(H, H)`

**Per-step update:**
```
msg_r = contact_conv(h_r, contact_edges, e_contact)
      + cross_conv_a2r((h_a, h_r), cross_edges_swapped, e_cross)

msg_a = bond_conv(h_a, bond_edges, e_bond)
      + cross_conv_r2a((h_r, h_a), cross_edges, e_cross)

h_r = gru_res(ReLU(msg_r), h_r)
h_a = gru_atom(ReLU(msg_a), h_a)
```

(`cross_edges_swapped` flips the row order of `cross_edge_index` so the
bipartite NNConv treats atoms as source and residues as destination.)

**Pooling and head: identical to A.** `global_add_pool(h_r, batch.batch)` and
`global_add_pool(h_a, batch.chrom_x_batch)` produce the two `[B, H]` vectors;
they're concatenated with kDa and passed to the same MLP head.

**Result:** A and B differ in *one* place — whether protein and chromophore
exchange messages mid-network. Same input, same output shape, same head, same
optimizer. Clean architectural ablation.

## 10. Training (`train.py`)

**Wrapper:** `FluorLitModule(pl.LightningModule)` owns all normalization.
Constructed with `model: nn.Module` (either an `FPNetA` or an `FPNetB`)
selected by a `--model {a,b}` CLI flag in `train.py`. Four buffers, all
computed from the **train split only** at module construction:
- `target_mean`, `target_std` — shape `[2]`, for `y`
- `kda_mean`, `kda_std` — scalars

**Targets in / out:** `y_z = (y − target_mean) / target_std` (computed in
`{training,validation,test}_step` before the loss). To avoid div-by-zero when
N_train = 1, std is clamped: `target_std = train_std.clamp_min(1.0)`. Same
clamp for `kda_std`.

**kDa flow:** the LitModule attaches `batch.kda_z = (batch.kda - kda_mean) /
kda_std` before calling `FPNet.forward(batch)`. The model receives only
already-normalized scalars; the buffers travel with the LitModule's
state_dict (so checkpoints round-trip cleanly).

**Loss:** `F.mse_loss(pred_z, y_z)` summed across both targets.

**Reported metrics** (per epoch):
- `train_loss` — z-space MSE.
- `val_mse_brightness`, `val_mse_emission` — denormalized to original units.
- `val_mae_brightness`, `val_mae_emission` — denormalized.
- Equivalents for the test split.

**Optimizer:** Adam, `lr=1e-3` (friend's default). No scheduler in v1.

**Trainer:** `pl.Trainer(max_epochs=60, logger=CSVLogger("logs/"))`. Single
`Trainer.fit(model)` call drives training; `Trainer.test(ckpt_path="best")`
runs the test sample.

**Comparison runs.** To compare A vs B, run twice with different `--model`
flags. CSV logs go to `logs/<model>/version_*/metrics.csv`, naturally
separated. With N=1 train both will overfit instantly; meaningful comparison
waits for the full ~100-PDB dataset.

## 11. Tests (`tests/`)

Two minimal pytest files — class-project style, not exhaustive. The bar is
"catches breakage when scaling up."

- `test_graphs.py`: parsing + graph building for 7ZCT (+ a couple of
  assertions for 4OQW). Covers `get_chromophore_code` returning `'NRQ'`,
  protein/chromophore/cross-edge graphs having sane shapes, dtypes, no NaN,
  no self-loops, distances ≤ cutoff.
- `test_smoke_train.py`: end-to-end one training step on the two PDBs **for
  both `FPNetA` and `FPNetB`**; asserts loss is finite and gradients flow.
  Runs in <60 s.

## 12. Open future work (not in v1)

- Gaussian-RBF expansion of edge distances for the protein graph.
- Per-residue features beyond one-hot (Meiler features, conservation).
- Optional inclusion of chromophore-coordinating waters (Q2 option C).
- Mass-weighted COM as alternative residue position.
- Cross-attention fusion (alternative to A's concat and B's NNConv-cross).
- Hyperparameter search once the full ~100-PDB dataset arrives — including
  a sweep of the cross-edge cutoff in B (5, 6, 8 Å).
- Ensemble / k-fold CV; A vs B comparison on full dataset with proper splits.
- More robust chromophore-graph build when PDB heavy-atom counts diverge
  from CCD SMILES. Currently `build_chromophore_graph` trims the terminal
  carboxylate `-OH` from the SMILES template to handle the mScarlet/mCardinal
  case where 7ZCT and 4OQW omit the OXT atom of NRQ. For broader chromophore
  coverage, consider: (a) keeping a per-chromophore template-correction map,
  (b) falling back to `Chem.MolFromPDBBlock(sanitize=True)` with inferred
  bond orders when the template substructure match fails, or (c) using
  OpenBabel as a secondary bond-order assigner.

## 13. Decisions log

| # | Decision | Rationale |
|---|---|---|
| Q1 | Protein graph: 8 Å Cα–Cα contact graph | Field convention; sparse; meaningful spatial inductive bias |
| Q2 | Drop all waters | Crystallography artifacts don't generalize |
| Q3 | Runnable training loop (not just smoke test) | Surfaces wiring bugs the smoke test wouldn't |
| Q4 | Lightning + CSVLogger, no W&B | Avoids login, keeps deps light |
| —  | Cα atom (not COM) for residue position | Reproducible, single-line extraction, field convention |
| —  | OGB 9-feature atom schema for chromophore | Strict superset of friend's 4 fields; uses `AtomEncoder` idiomatically |
| —  | Distance as side-channel on chromophore edges | Preserves friend's good idea; correct plumbing into `BondEncoder` |
| —  | Drop xyz from atom features | Not rotation-invariant; redundant with edge distances |
| —  | Class-project style (no over-engineered dataclasses) | Per user direction — simple, ipynb-like |
| —  | Implement both A (disjoint) and B (merged with cross-edges) | User wants to switch and compare during training; same dataset and head, only message-passing differs |
| —  | Cross-edge cutoff: 6 Å between residue Cα and chromophore atom | ≈5 Å covers direct contacts/H-bonds; +1 Å picks up second-shell residues |
