# Fluorescent-Protein GNN Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable GNN pipeline that predicts brightness and emission wavelength of a fluorescent protein from its PDB structure, with two switchable architectures (disjoint baseline A vs merged-with-cross-edges B) for comparison.

**Architecture:** Read PDB → build a 20-D residue contact graph (Cα–Cα, 8 Å) and a chromophore graph (RDKit + CCD template, OGB atom/bond features) plus residue↔atom cross-edges (6 Å). Both graphs live in one PyG `Data` subclass for batching. Two MPNNs (NNConv + GRU + global_add_pool) encode them; A concatenates the graph-level vectors with kDa, B adds bipartite NNConvs over cross-edges before pooling. Z-scored 2-D output (brightness, emission), Adam, MSE.

**Tech Stack:** Python 3.11+, `uv` for venv/deps, PyTorch, PyTorch Geometric, RDKit, OGB, PyTorch Lightning, pytest.

**Spec:** `docs/superpowers/specs/2026-04-29-fp-gnn-pipeline-design.md`

---

## File Structure

To be created or modified under repo root `/home/l-braz/Documents/git/pdb_project_2`:

| Path | Responsibility |
|---|---|
| `pyproject.toml` | uv-managed deps and project metadata |
| `.gitignore` | exclude data/processed/, data/ccd_cache/, .venv/, etc. |
| `data/labels.csv` | extend with a `split` column (existing file edited) |
| `data/processed/` | dir for cached PyG dataset (gitignored) |
| `data/ccd_cache/` | dir for cached CCD SMILES JSON files (gitignored) |
| `src/fp_gnn/__init__.py` | package marker |
| `src/fp_gnn/pdb_io.py` | `CHROMOPHORE_CODES`, `get_chromophore_code`, `get_chromophore_pdb_block`, `get_protein_residue_ca` |
| `src/fp_gnn/chromophore_graph.py` | `get_ccd_smiles_cached`, `mol_to_graph`, `build_chromophore_graph` |
| `src/fp_gnn/protein_graph.py` | `build_protein_graph`, `build_cross_edges` |
| `src/fp_gnn/dataset.py` | `FPData(Data)`, `FluorProteinDataset(InMemoryDataset)` |
| `src/fp_gnn/model.py` | `ChromMPNN`, `ProteinMPNN`, `FPNetA`, `FPNetB` |
| `src/fp_gnn/train.py` | `FluorLitModule`, z-score helpers, CLI entry point |
| `tests/__init__.py` | empty |
| `tests/test_graphs.py` | parsing + graph-building assertions for both PDBs |
| `tests/test_smoke_train.py` | end-to-end one training step for A and B |

---

## Tasks

### Task 1: Project bootstrap (uv venv, deps, layout, git)

**Files:**
- Create: `pyproject.toml` (via `uv init`)
- Create: `src/fp_gnn/__init__.py` (via `uv init --package`)
- Create: `tests/__init__.py`
- Create: `.gitignore`
- Create: `data/processed/` (empty), `data/ccd_cache/` (empty)
- Delete: `main.py` (unused PyCharm template)

- [ ] **Step 1: Initialize git repo and commit existing assets**

Run from repo root:

```bash
cd /home/l-braz/Documents/git/pdb_project_2
git init
git add data/raw/ data/labels.csv "Ligand_graph_inspired by exercises.ipynb" docs/
git commit -m "chore: initial commit of provided assets"
```

Expected: a commit listing the PDBs, labels.csv, the friend's ipynb, and the design spec.

- [ ] **Step 2: Remove unused main.py and bootstrap uv package**

```bash
rm main.py
uv init --package --name fp-gnn --python 3.11
```

This creates `pyproject.toml` and `src/fp_gnn/__init__.py`.

- [ ] **Step 3: Add core dependencies (CPU build is fine for 2 PDBs)**

```bash
uv add numpy pandas requests rdkit pytorch-lightning torch
uv add --dev pytest
```

If `uv add` is slow or pins to GPU torch, you can add `--index https://download.pytorch.org/whl/cpu` to torch.

- [ ] **Step 4: Add torch_geometric and ogb**

```bash
uv add torch_geometric ogb
```

`torch_geometric`'s optional sparse/cluster extensions are NOT required — `NNConv` and `global_add_pool` work with the pure-Python install.

- [ ] **Step 5: Create remaining directories and tests package marker**

```bash
mkdir -p tests data/processed data/ccd_cache
touch tests/__init__.py
```

- [ ] **Step 6: Write .gitignore**

Create `/home/l-braz/Documents/git/pdb_project_2/.gitignore` with:

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
data/processed/
data/ccd_cache/
logs/
.DS_Store
*.egg-info/
```

- [ ] **Step 7: Smoke-check the environment**

```bash
uv run python -c "import torch, torch_geometric, rdkit, ogb, pytorch_lightning, numpy, pandas, requests; print('OK')"
```

Expected output: `OK`

- [ ] **Step 8: Commit project bootstrap**

```bash
git add pyproject.toml uv.lock src/ tests/ .gitignore
git commit -m "chore: bootstrap uv package layout and dependencies"
```

---

### Task 2: Add `split` column to labels.csv

**Files:**
- Modify: `data/labels.csv`

- [ ] **Step 1: Edit labels.csv to add split column**

Replace the file contents with (preserve the long sequences exactly):

```
pdb_code,protein_name,pdb_path,kDa,brightness,emission,qy,sequence,split
7ZCT,mScarlet3,data/raw/7ZCT_mScarlet3.pdb,25.85,78.00,592,0.75,MDSTEAVIKEFMRFKVHMEGSMNGHEFEIEGEGEGRPYEGTQTAKLRVTKGGPLPFSWDILSPQFMYGSRAFTKHPADIPDYWKQSFPEGFKWERVMNFEDGGAVSVAQDTSLEDGTLIYKVKLRGTNFPPDGPVMQKKTMGWEASTERLYPEDVVLKGDIKMALRLKDGGRYLADFKTTYRAKKPVQMPGAFNIDRKLDITSHNEDYTVVEQYERSVARHSTGGSGGS,train
4OQW,mCardinal,data/raw/4OQW_mCardinal.pdb,27.55,16.53,659,0.19,MVSKGEELIKENMHMKLYMEGTVNNHHFKCTTEGEGKPYEGTQTQRIKVVEGGPLPFAFDILATCFMYGSKTFINHTQGIPDFFKQSFPEGFTWERVTTYEDGGVLTVTQDTSLQDGCLIYNVKLRGVNFPSNGPVMQKKTLGWEATTETLYPADGGLEGRCDMALKLVGGGHLHCNLKTTYRSKKPAKNLKMPGVYFVDRRLERIKEADNETYVEQHEVAVARYCDLPSKLGHKLNGMDELYK,test
```

- [ ] **Step 2: Verify pandas can load it**

```bash
uv run python -c "import pandas as pd; df = pd.read_csv('data/labels.csv'); print(df[['pdb_code','split','brightness','emission']])"
```

Expected: a 2-row dataframe with the split column showing `train` and `test`.

- [ ] **Step 3: Commit**

```bash
git add data/labels.csv
git commit -m "data: add split column (7ZCT=train, 4OQW=test)"
```

---

### Task 3: `pdb_io.py` — `CHROMOPHORE_CODES` and `get_chromophore_code`

**Files:**
- Create: `src/fp_gnn/pdb_io.py`
- Create: `tests/test_graphs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_graphs.py`:

```python
from pathlib import Path

from fp_gnn.pdb_io import CHROMOPHORE_CODES, get_chromophore_code

PDB_7ZCT = Path("data/raw/7ZCT_mScarlet3.pdb")
PDB_4OQW = Path("data/raw/4OQW_mCardinal.pdb")


def test_chromophore_codes_set_size():
    assert len(CHROMOPHORE_CODES) == 29
    assert "NRQ" in CHROMOPHORE_CODES


def test_get_chromophore_code_7zct_is_nrq():
    assert get_chromophore_code(PDB_7ZCT) == "NRQ"


def test_get_chromophore_code_4oqw_is_nrq():
    assert get_chromophore_code(PDB_4OQW) == "NRQ"
```

- [ ] **Step 2: Run the test — it must fail with ImportError**

```bash
uv run pytest tests/test_graphs.py::test_chromophore_codes_set_size -v
```

Expected: ImportError or ModuleNotFoundError on `fp_gnn.pdb_io`.

- [ ] **Step 3: Implement the minimal pdb_io.py**

Create `src/fp_gnn/pdb_io.py`:

```python
from pathlib import Path

CHROMOPHORE_CODES = {
    "NRQ", "CRQ", "NRP", "CH6", "CRO", "5SQ", "4M9", "CR2", "OFM", "CR8",
    "CFY", "OIM", "CH7", "GYS", "WCR", "GYC", "DYG", "FAD", "PIA", "CCY",
    "BLR", "CRF", "NYG", "CR7", "FMN", "B2H", "SWG", "CSH", "BJF",
}


def get_chromophore_code(pdb_path):
    """Return the first HETATM residue name in `pdb_path` that matches
    a known chromophore code, or raise ValueError if none is found."""
    pdb_path = Path(pdb_path)
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            residue_name = line[17:20].strip()
            if residue_name in CHROMOPHORE_CODES:
                return residue_name
    raise ValueError(f"No known chromophore found in {pdb_path}")
```

- [ ] **Step 4: Run the tests — they must pass**

```bash
uv run pytest tests/test_graphs.py -v -k "chromophore_codes or chromophore_code"
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fp_gnn/pdb_io.py tests/test_graphs.py
git commit -m "feat(pdb_io): chromophore code constants and lookup"
```

---

### Task 4: `pdb_io.py` — `get_chromophore_pdb_block` (chain A enforced)

**Files:**
- Modify: `src/fp_gnn/pdb_io.py`
- Modify: `tests/test_graphs.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_graphs.py`:

```python
from fp_gnn.pdb_io import get_chromophore_pdb_block


def test_chromophore_block_7zct_is_chain_a_only():
    block = get_chromophore_pdb_block(PDB_7ZCT, "NRQ")
    lines = [l for l in block.splitlines() if l.startswith("HETATM")]
    assert len(lines) > 0
    # Every line must be HETATM, NRQ, chain A
    for line in lines:
        assert line.startswith("HETATM")
        assert line[17:20].strip() == "NRQ"
        assert line[21] == "A"
    # Block must end with END
    assert block.rstrip().endswith("END")


def test_chromophore_block_raises_when_chain_a_missing():
    # Construct a tiny PDB string with NRQ on chain B only, write to tmp
    import tempfile, os
    fake_block = (
        "HETATM    1  CA1 NRQ B  67       1.000   2.000   3.000  1.00  0.00           C\n"
        "END\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
        f.write(fake_block)
        tmp = f.name
    try:
        import pytest
        with pytest.raises(ValueError):
            get_chromophore_pdb_block(tmp, "NRQ")
    finally:
        os.unlink(tmp)
```

- [ ] **Step 2: Run the new tests — they must fail with AttributeError or similar**

```bash
uv run pytest tests/test_graphs.py -v -k "chromophore_block"
```

Expected: ImportError on `get_chromophore_pdb_block`.

- [ ] **Step 3: Add `get_chromophore_pdb_block` to pdb_io.py**

Append to `src/fp_gnn/pdb_io.py`:

```python
def get_chromophore_pdb_block(pdb_path, residue_code):
    """Return all HETATM lines for `residue_code` in chain A as a PDB
    block string (terminated by 'END\\n'). Raises ValueError if chain A
    is absent.

    If multiple residue copies exist in chain A (e.g. multiple NRQ
    instances), the lowest resseq is used and a warning is printed.
    """
    pdb_path = Path(pdb_path)

    chain_a_resseqs = set()
    matched_lines_per_resseq = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            if line[17:20].strip() != residue_code:
                continue
            if line[21] != "A":
                continue
            resseq = line[22:26].strip()
            chain_a_resseqs.add(resseq)
            matched_lines_per_resseq.setdefault(resseq, []).append(line)

    if not chain_a_resseqs:
        raise ValueError(
            f"Residue '{residue_code}' not found in chain A of {pdb_path.name}"
        )

    if len(chain_a_resseqs) > 1:
        print(
            f"[WARN] Multiple '{residue_code}' copies in chain A "
            f"({sorted(chain_a_resseqs)}); using lowest resseq."
        )

    chosen = sorted(chain_a_resseqs, key=lambda s: int(s))[0]
    return "".join(matched_lines_per_resseq[chosen]) + "END\n"
```

- [ ] **Step 4: Run all pdb_io tests**

```bash
uv run pytest tests/test_graphs.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fp_gnn/pdb_io.py tests/test_graphs.py
git commit -m "feat(pdb_io): get_chromophore_pdb_block with chain A enforcement"
```

---

### Task 5: `pdb_io.py` — `get_protein_residue_ca`

**Files:**
- Modify: `src/fp_gnn/pdb_io.py`
- Modify: `tests/test_graphs.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_graphs.py`:

```python
import numpy as np
from fp_gnn.pdb_io import get_protein_residue_ca


def test_residue_ca_7zct_shapes():
    names, coords = get_protein_residue_ca(PDB_7ZCT)
    assert isinstance(names, list)
    assert isinstance(coords, np.ndarray)
    assert coords.shape == (len(names), 3)
    assert len(names) > 100  # mScarlet3 is ~230 residues
    assert "ALA" in names    # mScarlet3 starts with ALA per the spec example


def test_residue_ca_no_water_no_chromophore():
    names, _ = get_protein_residue_ca(PDB_7ZCT)
    assert "HOH" not in names
    assert "NRQ" not in names
```

- [ ] **Step 2: Run the new tests — they must fail**

```bash
uv run pytest tests/test_graphs.py -v -k "residue_ca"
```

Expected: ImportError on `get_protein_residue_ca`.

- [ ] **Step 3: Implement `get_protein_residue_ca`**

Append to `src/fp_gnn/pdb_io.py`:

```python
import numpy as np


def get_protein_residue_ca(pdb_path):
    """Parse ATOM records from `pdb_path`, keep chain A and altloc ' '
    or 'A', skip waters (HOH), and return one Cα atom per residue.

    Returns:
        residue_names: list[str], length N
        ca_coords:    np.ndarray of shape (N, 3) — x, y, z in Å
    """
    pdb_path = Path(pdb_path)

    # Order-preserving collection: keyed by (chain, resseq, icode)
    seen_residues = []
    ca_per_residue = {}
    name_per_residue = {}

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            chain = line[21]
            if chain != "A":
                continue
            res_name = line[17:20].strip()
            if res_name == "HOH":
                continue
            atom_name = line[12:16].strip()
            resseq = line[22:26].strip()
            icode = line[26]
            res_id = (chain, resseq, icode)

            if res_id not in ca_per_residue:
                seen_residues.append(res_id)
                name_per_residue[res_id] = res_name

            if atom_name == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ca_per_residue[res_id] = (x, y, z)

    residue_names = []
    coords = []
    for res_id in seen_residues:
        if res_id not in ca_per_residue:
            print(
                f"[WARN] residue {res_id} in {pdb_path.name} has no Cα atom; skipping"
            )
            continue
        residue_names.append(name_per_residue[res_id])
        coords.append(ca_per_residue[res_id])

    return residue_names, np.asarray(coords, dtype=np.float32)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_graphs.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fp_gnn/pdb_io.py tests/test_graphs.py
git commit -m "feat(pdb_io): per-residue Cα extraction (chain A, altloc filter, no HOH)"
```

---

### Task 6: `chromophore_graph.py` — `get_ccd_smiles_cached`

**Files:**
- Create: `src/fp_gnn/chromophore_graph.py`
- Modify: `tests/test_graphs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_graphs.py`:

```python
from fp_gnn.chromophore_graph import get_ccd_smiles_cached


def test_ccd_smiles_cache_round_trip(tmp_path, monkeypatch):
    """First call hits the API; second call reads the cache."""
    cache_dir = tmp_path / "ccd_cache"
    cache_dir.mkdir()

    call_count = {"n": 0}

    class FakeResp:
        def json(self):
            return {"rcsb_chem_comp_descriptor": {"smiles": "CCO"}}

    def fake_get(url):
        call_count["n"] += 1
        return FakeResp()

    monkeypatch.setattr("fp_gnn.chromophore_graph.requests.get", fake_get)

    smi1 = get_ccd_smiles_cached("FAKE", cache_dir=cache_dir)
    smi2 = get_ccd_smiles_cached("FAKE", cache_dir=cache_dir)

    assert smi1 == "CCO"
    assert smi2 == "CCO"
    assert call_count["n"] == 1     # second call hit the cache
    assert (cache_dir / "FAKE.json").exists()
```

- [ ] **Step 2: Run the test — it must fail**

```bash
uv run pytest tests/test_graphs.py -v -k "ccd_smiles_cache"
```

Expected: ImportError on `fp_gnn.chromophore_graph`.

- [ ] **Step 3: Implement `get_ccd_smiles_cached`**

Create `src/fp_gnn/chromophore_graph.py`:

```python
import json
from pathlib import Path

import requests

DEFAULT_CCD_CACHE = Path("data/ccd_cache")


def get_ccd_smiles_cached(residue_code, cache_dir=DEFAULT_CCD_CACHE):
    """Return canonical SMILES for `residue_code` from the RCSB Chemical
    Component Dictionary, using a local JSON cache.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{residue_code}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())["smiles"]

    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{residue_code}"
    data = requests.get(url).json()
    smiles = data["rcsb_chem_comp_descriptor"]["smiles"]
    cache_file.write_text(json.dumps({"smiles": smiles}))
    return smiles
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_graphs.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Pre-warm the NRQ cache so later tests don't hit the network**

```bash
uv run python -c "from fp_gnn.chromophore_graph import get_ccd_smiles_cached; print(get_ccd_smiles_cached('NRQ'))"
```

Expected: a SMILES string. Verify `data/ccd_cache/NRQ.json` exists.

- [ ] **Step 6: Commit**

```bash
git add src/fp_gnn/chromophore_graph.py tests/test_graphs.py
git commit -m "feat(chromophore_graph): CCD SMILES fetcher with local JSON cache"
```

---

### Task 7: `chromophore_graph.py` — `mol_to_graph` and `build_chromophore_graph`

**Files:**
- Modify: `src/fp_gnn/chromophore_graph.py`
- Modify: `tests/test_graphs.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_graphs.py`:

```python
import torch
from fp_gnn.chromophore_graph import build_chromophore_graph


def test_chromophore_graph_7zct_shapes():
    data = build_chromophore_graph(PDB_7ZCT)
    # Atom features: long [N, 9] (OGB schema)
    assert data.x.dtype == torch.long
    assert data.x.shape[1] == 9
    N = data.x.shape[0]
    assert 15 <= N <= 30  # NRQ heavy-atom count is in this range

    # Edge index: long [2, E], no self-loops, bidirectional
    assert data.edge_index.dtype == torch.long
    assert data.edge_index.shape[0] == 2
    assert (data.edge_index[0] != data.edge_index[1]).all()

    # Bond chemistry features: long [E, 3]
    assert data.edge_attr_chem.dtype == torch.long
    assert data.edge_attr_chem.shape[1] == 3
    assert data.edge_attr_chem.shape[0] == data.edge_index.shape[1]

    # Bond distance features: float [E, 1]
    assert data.edge_attr_dist.dtype == torch.float
    assert data.edge_attr_dist.shape == (data.edge_index.shape[1], 1)
    assert (data.edge_attr_dist > 0).all()

    # Heavy-atom positions on .pos (PyG convention)
    assert data.pos.shape == (N, 3)
    assert data.pos.dtype == torch.float
```

- [ ] **Step 2: Run — must fail**

```bash
uv run pytest tests/test_graphs.py -v -k "chromophore_graph_7zct_shapes"
```

Expected: ImportError on `build_chromophore_graph`.

- [ ] **Step 3: Implement `mol_to_graph` and `build_chromophore_graph`**

Append to `src/fp_gnn/chromophore_graph.py`:

```python
import numpy as np
import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from fp_gnn.pdb_io import get_chromophore_code, get_chromophore_pdb_block


def mol_to_graph(mol):
    """Convert a sanitized RDKit Mol (heavy atoms only, with a 3D
    conformer) to a PyG Data object using OGB-compatible features."""
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    positions = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(n_atoms)],
        dtype=np.float32,
    )

    atom_feats = [atom_to_feature_vector(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.long)

    src, dst = [], []
    chem_feats = []
    distances = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_to_feature_vector(bond)
        d = float(np.linalg.norm(positions[i] - positions[j]))
        # Both directions
        src.extend([i, j])
        dst.extend([j, i])
        chem_feats.extend([bf, bf])
        distances.extend([d, d])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr_chem = torch.tensor(chem_feats, dtype=torch.long)
    edge_attr_dist = torch.tensor(distances, dtype=torch.float).unsqueeze(-1)
    pos = torch.tensor(positions, dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr_chem=edge_attr_chem,
        edge_attr_dist=edge_attr_dist,
        pos=pos,
    )


def build_chromophore_graph(pdb_path):
    """Read the chromophore from a PDB file, build a sanitized 3D RDKit
    Mol via the CCD SMILES template, and return its PyG Data graph."""
    code = get_chromophore_code(pdb_path)
    block = get_chromophore_pdb_block(pdb_path, code)

    smiles = get_ccd_smiles_cached(code)
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise RuntimeError(f"RDKit could not parse SMILES for {code}: {smiles}")

    raw_mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    if raw_mol is None:
        raise RuntimeError(f"RDKit could not parse PDB block for {code} in {pdb_path}")

    mol = AllChem.AssignBondOrdersFromTemplate(template, raw_mol)
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)

    return mol_to_graph(mol)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_graphs.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fp_gnn/chromophore_graph.py tests/test_graphs.py
git commit -m "feat(chromophore_graph): RDKit+CCD-based PyG graph builder with OGB features"
```

---

### Task 8: `protein_graph.py` — `build_protein_graph`

**Files:**
- Create: `src/fp_gnn/protein_graph.py`
- Modify: `tests/test_graphs.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_graphs.py`:

```python
from fp_gnn.protein_graph import build_protein_graph


def test_protein_graph_7zct():
    names, coords = get_protein_residue_ca(PDB_7ZCT)
    data = build_protein_graph(names, coords, cutoff=8.0)

    N = len(names)
    # Node features: float [N, 20]
    assert data.x.shape == (N, 20)
    assert data.x.dtype == torch.float
    # Each row is a one-hot (or all-zero for non-canonical residues)
    row_sums = data.x.sum(dim=1)
    assert ((row_sums == 1.0) | (row_sums == 0.0)).all()

    # Edges: bidirectional, no self-loops
    assert data.edge_index.shape[0] == 2
    assert (data.edge_index[0] != data.edge_index[1]).all()
    E = data.edge_index.shape[1]
    assert E > 0
    # Distance attr: float [E, 1], all <= cutoff
    assert data.edge_attr.shape == (E, 1)
    assert (data.edge_attr <= 8.0 + 1e-5).all()
    assert (data.edge_attr > 0).all()

    # Edge symmetry: (i,j) appears iff (j,i) appears
    pairs = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    for (i, j) in pairs:
        assert (j, i) in pairs
```

- [ ] **Step 2: Run — must fail**

```bash
uv run pytest tests/test_graphs.py -v -k "protein_graph_7zct"
```

Expected: ImportError on `build_protein_graph`.

- [ ] **Step 3: Implement `build_protein_graph`**

Create `src/fp_gnn/protein_graph.py`:

```python
import numpy as np
import torch
from torch_geometric.data import Data

# 20 standard amino acids in fixed alphabetical order — index = one-hot column
STANDARD_AA = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_INDEX = {name: i for i, name in enumerate(STANDARD_AA)}


def _one_hot_residues(residue_names):
    N = len(residue_names)
    x = torch.zeros((N, 20), dtype=torch.float)
    for i, name in enumerate(residue_names):
        idx = AA_INDEX.get(name)
        if idx is None:
            print(f"[WARN] non-standard residue '{name}' at index {i}; encoding as zero vector")
            continue
        x[i, idx] = 1.0
    return x


def build_protein_graph(residue_names, ca_coords, cutoff=8.0):
    """Residue contact graph. Nodes = residues with 20-D one-hot;
    edges = pairs whose Cα-Cα distance <= cutoff Å, in both directions,
    no self-loops; edge_attr = scalar distance."""
    N = len(residue_names)
    assert ca_coords.shape == (N, 3), f"ca_coords shape {ca_coords.shape} != ({N}, 3)"

    coords = np.asarray(ca_coords, dtype=np.float32)
    # Pairwise Euclidean distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))

    mask = (dist <= cutoff) & (dist > 0.0)  # exclude self-loops
    src, dst = np.where(mask)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(dist[src, dst], dtype=torch.float).unsqueeze(-1)

    x = _one_hot_residues(residue_names)
    pos = torch.tensor(coords, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_graphs.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fp_gnn/protein_graph.py tests/test_graphs.py
git commit -m "feat(protein_graph): 8 Å Cα contact graph with 20-D one-hot residues"
```

---

### Task 9: `protein_graph.py` — `build_cross_edges`

**Files:**
- Modify: `src/fp_gnn/protein_graph.py`
- Modify: `tests/test_graphs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_graphs.py`:

```python
from fp_gnn.protein_graph import build_cross_edges


def test_cross_edges_7zct():
    names, ca = get_protein_residue_ca(PDB_7ZCT)
    chrom = build_chromophore_graph(PDB_7ZCT)
    chrom_xyz = chrom.pos.numpy()

    cross_idx, cross_dist = build_cross_edges(ca, chrom_xyz, cutoff=6.0)

    assert cross_idx.dtype == torch.long
    assert cross_idx.shape[0] == 2
    E = cross_idx.shape[1]
    assert E > 0  # NRQ is buried in mScarlet3 -- expect at least some neighbors
    assert cross_dist.shape == (E, 1)
    assert (cross_dist <= 6.0 + 1e-5).all()
    assert (cross_dist > 0).all()

    # Row 0 must index residues, row 1 must index chrom atoms
    assert cross_idx[0].max().item() < len(names)
    assert cross_idx[1].max().item() < chrom_xyz.shape[0]
```

- [ ] **Step 2: Run — must fail**

```bash
uv run pytest tests/test_graphs.py -v -k "cross_edges_7zct"
```

Expected: ImportError on `build_cross_edges`.

- [ ] **Step 3: Implement `build_cross_edges`**

Append to `src/fp_gnn/protein_graph.py`:

```python
def build_cross_edges(ca_coords, chrom_coords, cutoff=6.0):
    """Bipartite edges between residue Cα and chromophore atoms within
    `cutoff` Å.

    Returns:
        cross_edge_index: long [2, E] -- row 0 = residue idx, row 1 = atom idx
        cross_edge_attr:  float [E, 1] -- Euclidean distance Å
    """
    ca = np.asarray(ca_coords, dtype=np.float32)
    cx = np.asarray(chrom_coords, dtype=np.float32)

    # Pairwise residue-vs-atom distances: (N_res, N_atom)
    diff = ca[:, None, :] - cx[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))

    res_idx, atom_idx = np.where(dist <= cutoff)

    cross_edge_index = torch.tensor(
        np.stack([res_idx, atom_idx], axis=0), dtype=torch.long
    )
    cross_edge_attr = torch.tensor(
        dist[res_idx, atom_idx], dtype=torch.float
    ).unsqueeze(-1)
    return cross_edge_index, cross_edge_attr
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_graphs.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fp_gnn/protein_graph.py tests/test_graphs.py
git commit -m "feat(protein_graph): residue-chromophore cross-edges (6 Å cutoff)"
```

---

### Task 10: `dataset.py` — `FPData(Data)` and `FluorProteinDataset`

**Files:**
- Create: `src/fp_gnn/dataset.py`
- Modify: `tests/test_graphs.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_graphs.py`:

```python
from fp_gnn.dataset import FluorProteinDataset


def test_dataset_builds_two_items(tmp_path):
    # Use a fresh root so we exercise process() (no cache reuse)
    ds = FluorProteinDataset(
        root=str(tmp_path),
        labels_csv="data/labels.csv",
        repo_root=".",
    )
    assert len(ds) == 2

    item = ds[0]
    # Both graphs present
    assert hasattr(item, "x") and item.x.shape[1] == 20            # protein
    assert hasattr(item, "chrom_x") and item.chrom_x.shape[1] == 9 # chromophore
    assert hasattr(item, "cross_edge_index")
    assert item.cross_edge_index.shape[0] == 2
    # Scalars
    assert item.kda.shape == (1,)
    assert item.y.shape == (1, 2)


def test_dataset_batches_correctly(tmp_path):
    from torch_geometric.loader import DataLoader

    ds = FluorProteinDataset(
        root=str(tmp_path), labels_csv="data/labels.csv", repo_root=".",
    )
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))

    # PyG batch attrs: 'batch' for protein nodes, 'chrom_x_batch' for atoms
    assert batch.x.shape[0] == ds[0].x.shape[0] + ds[1].x.shape[0]
    assert batch.chrom_x.shape[0] == ds[0].chrom_x.shape[0] + ds[1].chrom_x.shape[0]
    # Cross-edges keep proper bipartite offsets after batching
    assert batch.cross_edge_index[0].max() < batch.x.shape[0]
    assert batch.cross_edge_index[1].max() < batch.chrom_x.shape[0]
```

- [ ] **Step 2: Run — must fail**

```bash
uv run pytest tests/test_graphs.py -v -k "dataset"
```

Expected: ImportError on `FluorProteinDataset`.

- [ ] **Step 3: Implement `FPData` and `FluorProteinDataset`**

Create `src/fp_gnn/dataset.py`:

```python
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from fp_gnn.chromophore_graph import build_chromophore_graph
from fp_gnn.pdb_io import get_protein_residue_ca
from fp_gnn.protein_graph import build_cross_edges, build_protein_graph


class FPData(Data):
    """Holds protein graph + chromophore graph + cross-edges in one object."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "chrom_edge_index":
            return self.chrom_x.size(0)
        if key == "cross_edge_index":
            # Row 0 = residue indices (offset by N_res = x.size(0))
            # Row 1 = chrom-atom indices (offset by N_atom = chrom_x.size(0))
            return torch.tensor([[self.x.size(0)], [self.chrom_x.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("chrom_edge_index", "cross_edge_index"):
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


class FluorProteinDataset(InMemoryDataset):
    def __init__(self, root, labels_csv="data/labels.csv", repo_root="."):
        self._labels_csv = Path(repo_root) / labels_csv
        self._repo_root = Path(repo_root)
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self._labels_csv)
        data_list = []
        for _, row in df.iterrows():
            pdb_path = self._repo_root / row["pdb_path"]

            res_names, ca = get_protein_residue_ca(pdb_path)
            prot = build_protein_graph(res_names, ca, cutoff=8.0)

            chrom = build_chromophore_graph(pdb_path)

            cross_idx, cross_attr = build_cross_edges(
                ca, chrom.pos.numpy(), cutoff=6.0,
            )

            item = FPData(
                # Protein graph
                x=prot.x,
                edge_index=prot.edge_index,
                edge_attr=prot.edge_attr,
                # Chromophore graph (chrom_ prefix)
                chrom_x=chrom.x,
                chrom_edge_index=chrom.edge_index,
                chrom_edge_attr_chem=chrom.edge_attr_chem,
                chrom_edge_attr_dist=chrom.edge_attr_dist,
                # Cross-edges
                cross_edge_index=cross_idx,
                cross_edge_attr=cross_attr,
                # Scalars
                kda=torch.tensor([row["kDa"]], dtype=torch.float),
                y=torch.tensor([[row["brightness"], row["emission"]]], dtype=torch.float),
                pdb_code=row["pdb_code"],
                split=row["split"],
            )
            data_list.append(item)

        torch.save(self.collate(data_list), self.processed_paths[0])
```

- [ ] **Step 4: Run the dataset tests**

```bash
uv run pytest tests/test_graphs.py -v -k "dataset"
```

Expected: 2 passed.

- [ ] **Step 5: Run the full test file**

```bash
uv run pytest tests/test_graphs.py -v
```

Expected: 13 passed.

- [ ] **Step 6: Commit**

```bash
git add src/fp_gnn/dataset.py tests/test_graphs.py
git commit -m "feat(dataset): FPData subclass + FluorProteinDataset (InMemoryDataset)"
```

---

### Task 11: `model.py` — `ChromMPNN` and `ProteinMPNN`

**Files:**
- Create: `src/fp_gnn/model.py`
- Create: `tests/test_smoke_train.py`

- [ ] **Step 1: Write the failing forward-pass test**

Create `tests/test_smoke_train.py`:

```python
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from fp_gnn.dataset import FluorProteinDataset


def _load_dataset(tmp_path):
    return FluorProteinDataset(
        root=str(tmp_path), labels_csv="data/labels.csv", repo_root=".",
    )


def test_chrom_and_protein_mpnn_forward(tmp_path):
    from fp_gnn.model import ChromMPNN, ProteinMPNN

    ds = _load_dataset(tmp_path)
    batch = next(iter(DataLoader(ds, batch_size=2)))

    chrom = ChromMPNN(node_embedding_dim=64, num_message_steps=3)
    prot = ProteinMPNN(node_embedding_dim=64, num_message_steps=3)

    chrom_emb = chrom(batch)
    prot_emb = prot(batch)

    assert chrom_emb.shape == (2, 64)
    assert prot_emb.shape == (2, 64)
    assert torch.isfinite(chrom_emb).all()
    assert torch.isfinite(prot_emb).all()
```

- [ ] **Step 2: Run — must fail**

```bash
uv run pytest tests/test_smoke_train.py -v
```

Expected: ImportError on `fp_gnn.model`.

- [ ] **Step 3: Implement `ChromMPNN` and `ProteinMPNN`**

Create `src/fp_gnn/model.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import MLP, NNConv, global_add_pool


def _make_edge_network(emb_dim, hidden=None):
    h = hidden or 2 * emb_dim
    return MLP([emb_dim, h, emb_dim * emb_dim])


class ChromMPNN(nn.Module):
    """MPNN over the chromophore graph (atoms + bonds + bond distance)."""

    def __init__(self, node_embedding_dim=64, num_message_steps=3):
        super().__init__()
        H = node_embedding_dim
        self.num_message_steps = num_message_steps

        self.atom_emb = AtomEncoder(emb_dim=H)
        self.bond_chem_emb = BondEncoder(emb_dim=H)
        self.dist_proj = nn.Linear(1, H)

        self.message_layer = NNConv(H, H, nn=_make_edge_network(H), aggr="mean")
        self.gru = nn.GRU(input_size=H, hidden_size=H)

    def forward(self, batch):
        x = self.atom_emb(batch.chrom_x)
        e = self.bond_chem_emb(batch.chrom_edge_attr_chem) + self.dist_proj(
            batch.chrom_edge_attr_dist
        )

        h = x.unsqueeze(0)  # GRU expects [1, N, H]
        node_state = x
        for _ in range(self.num_message_steps):
            m = self.message_layer(node_state, batch.chrom_edge_index, e)
            m = F.relu(m)
            node_state, h = self.gru(m.unsqueeze(0), h)
            node_state = node_state.squeeze(0)

        # PyG attaches a per-node batch index to non-default node attributes
        # via Batch.from_data_list. Default is `<attrname>_batch`.
        chrom_batch = getattr(batch, "chrom_x_batch", None)
        if chrom_batch is None:
            # Single-graph case: all atoms belong to graph 0
            chrom_batch = torch.zeros(node_state.shape[0], dtype=torch.long, device=node_state.device)
        return global_add_pool(node_state, chrom_batch)


class ProteinMPNN(nn.Module):
    """MPNN over the protein residue contact graph."""

    def __init__(self, node_embedding_dim=64, num_message_steps=3, num_residue_types=20):
        super().__init__()
        H = node_embedding_dim
        self.num_message_steps = num_message_steps

        self.node_proj = nn.Linear(num_residue_types, H)
        self.edge_proj = nn.Linear(1, H)

        self.message_layer = NNConv(H, H, nn=_make_edge_network(H), aggr="mean")
        self.gru = nn.GRU(input_size=H, hidden_size=H)

    def forward(self, batch):
        x = self.node_proj(batch.x)
        e = self.edge_proj(batch.edge_attr)

        h = x.unsqueeze(0)
        node_state = x
        for _ in range(self.num_message_steps):
            m = self.message_layer(node_state, batch.edge_index, e)
            m = F.relu(m)
            node_state, h = self.gru(m.unsqueeze(0), h)
            node_state = node_state.squeeze(0)

        prot_batch = getattr(batch, "batch", None)
        if prot_batch is None:
            prot_batch = torch.zeros(node_state.shape[0], dtype=torch.long, device=node_state.device)
        return global_add_pool(node_state, prot_batch)
```

- [ ] **Step 4: Run the test**

```bash
uv run pytest tests/test_smoke_train.py -v -k "chrom_and_protein_mpnn_forward"
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fp_gnn/model.py tests/test_smoke_train.py
git commit -m "feat(model): ChromMPNN and ProteinMPNN with NNConv+GRU+pool"
```

---

### Task 12: `model.py` — `FPNetA` (disjoint baseline)

**Files:**
- Modify: `src/fp_gnn/model.py`
- Modify: `tests/test_smoke_train.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smoke_train.py`:

```python
def test_fpnet_a_forward(tmp_path):
    from fp_gnn.model import FPNetA

    ds = _load_dataset(tmp_path)
    batch = next(iter(DataLoader(ds, batch_size=2)))
    # Pretend the LitModule has already z-scored kda
    batch.kda_z = batch.kda

    model = FPNetA(node_embedding_dim=64, num_message_steps=3)
    out = model(batch)

    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run — must fail**

```bash
uv run pytest tests/test_smoke_train.py -v -k "fpnet_a_forward"
```

Expected: ImportError on `FPNetA`.

- [ ] **Step 3: Implement `FPNetA`**

Append to `src/fp_gnn/model.py`:

```python
class FPNetA(nn.Module):
    """Disjoint two-graph baseline: ProteinMPNN || ChromMPNN || kda -> MLP."""

    def __init__(self, node_embedding_dim=64, num_message_steps=3):
        super().__init__()
        H = node_embedding_dim
        self.protein_mpnn = ProteinMPNN(H, num_message_steps)
        self.chrom_mpnn = ChromMPNN(H, num_message_steps)
        # Head: (2H + 1) -> H -> 2
        self.head = MLP([2 * H + 1, H, 2])

    def forward(self, batch):
        prot_emb = self.protein_mpnn(batch)
        chrom_emb = self.chrom_mpnn(batch)
        fused = torch.cat(
            [prot_emb, chrom_emb, batch.kda_z.view(-1, 1)], dim=-1
        )
        return self.head(fused)
```

- [ ] **Step 4: Run the test**

```bash
uv run pytest tests/test_smoke_train.py -v -k "fpnet_a_forward"
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fp_gnn/model.py tests/test_smoke_train.py
git commit -m "feat(model): FPNetA disjoint two-graph baseline"
```

---

### Task 13: `model.py` — `FPNetB` (merged graph with cross-edges)

**Files:**
- Modify: `src/fp_gnn/model.py`
- Modify: `tests/test_smoke_train.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smoke_train.py`:

```python
def test_fpnet_b_forward(tmp_path):
    from fp_gnn.model import FPNetB

    ds = _load_dataset(tmp_path)
    batch = next(iter(DataLoader(ds, batch_size=2)))
    batch.kda_z = batch.kda

    model = FPNetB(node_embedding_dim=64, num_message_steps=3)
    out = model(batch)

    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run — must fail**

```bash
uv run pytest tests/test_smoke_train.py -v -k "fpnet_b_forward"
```

Expected: ImportError on `FPNetB`.

- [ ] **Step 3: Implement `FPNetB`**

Append to `src/fp_gnn/model.py`:

```python
class FPNetB(nn.Module):
    """Merged graph with bipartite cross-edges between residues and chromophore atoms.

    Same pooling and head as FPNetA; the only architectural difference is
    that residues and atoms exchange messages every message-passing step
    via two bipartite NNConvs over `cross_edge_index`.
    """

    def __init__(self, node_embedding_dim=64, num_message_steps=3, num_residue_types=20):
        super().__init__()
        H = node_embedding_dim
        self.num_message_steps = num_message_steps

        # Encoders (mirrors FPNetA's two MPNNs but used jointly here)
        self.res_proj = nn.Linear(num_residue_types, H)
        self.atom_emb = AtomEncoder(emb_dim=H)

        self.contact_proj = nn.Linear(1, H)
        self.bond_chem_emb = BondEncoder(emb_dim=H)
        self.bond_dist_proj = nn.Linear(1, H)
        self.cross_proj = nn.Linear(1, H)

        # Convs, one per edge type per step (parameters are shared across steps)
        self.contact_conv = NNConv(H, H, nn=_make_edge_network(H), aggr="mean")
        self.bond_conv = NNConv(H, H, nn=_make_edge_network(H), aggr="mean")
        self.cross_conv_r2a = NNConv((H, H), H, nn=_make_edge_network(H), aggr="mean")
        self.cross_conv_a2r = NNConv((H, H), H, nn=_make_edge_network(H), aggr="mean")

        self.gru_res = nn.GRU(H, H)
        self.gru_atom = nn.GRU(H, H)

        self.head = MLP([2 * H + 1, H, 2])

    def forward(self, batch):
        h_r = self.res_proj(batch.x)
        h_a = self.atom_emb(batch.chrom_x)

        e_contact = self.contact_proj(batch.edge_attr)
        e_bond = self.bond_chem_emb(batch.chrom_edge_attr_chem) + self.bond_dist_proj(
            batch.chrom_edge_attr_dist
        )
        e_cross = self.cross_proj(batch.cross_edge_attr)

        # cross_edge_index: row 0 = residue, row 1 = atom
        # For r->a: source residues -> destination atoms (use as-is)
        # For a->r: source atoms    -> destination residues (swap rows)
        cross_r2a = batch.cross_edge_index
        cross_a2r = batch.cross_edge_index.flip(0)

        state_r = h_r.unsqueeze(0)
        state_a = h_a.unsqueeze(0)
        node_r = h_r
        node_a = h_a

        for _ in range(self.num_message_steps):
            msg_r = self.contact_conv(node_r, batch.edge_index, e_contact) + self.cross_conv_a2r(
                (node_a, node_r), cross_a2r, e_cross
            )
            msg_a = self.bond_conv(node_a, batch.chrom_edge_index, e_bond) + self.cross_conv_r2a(
                (node_r, node_a), cross_r2a, e_cross
            )
            msg_r = F.relu(msg_r)
            msg_a = F.relu(msg_a)

            node_r, state_r = self.gru_res(msg_r.unsqueeze(0), state_r)
            node_a, state_a = self.gru_atom(msg_a.unsqueeze(0), state_a)
            node_r = node_r.squeeze(0)
            node_a = node_a.squeeze(0)

        # Pool per node type
        prot_batch = getattr(batch, "batch", None)
        if prot_batch is None:
            prot_batch = torch.zeros(node_r.shape[0], dtype=torch.long, device=node_r.device)
        chrom_batch = getattr(batch, "chrom_x_batch", None)
        if chrom_batch is None:
            chrom_batch = torch.zeros(node_a.shape[0], dtype=torch.long, device=node_a.device)

        prot_emb = global_add_pool(node_r, prot_batch)
        chrom_emb = global_add_pool(node_a, chrom_batch)

        fused = torch.cat([prot_emb, chrom_emb, batch.kda_z.view(-1, 1)], dim=-1)
        return self.head(fused)
```

- [ ] **Step 4: Run the test**

```bash
uv run pytest tests/test_smoke_train.py -v -k "fpnet_b_forward"
```

Expected: 1 passed.

- [ ] **Step 5: Run all model tests**

```bash
uv run pytest tests/test_smoke_train.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/fp_gnn/model.py tests/test_smoke_train.py
git commit -m "feat(model): FPNetB merged graph with bipartite cross-edge NNConvs"
```

---

### Task 14: `train.py` — z-score helpers + `FluorLitModule`

**Files:**
- Create: `src/fp_gnn/train.py`
- Modify: `tests/test_smoke_train.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_smoke_train.py`:

```python
def test_zscore_helpers_train_only():
    from fp_gnn.train import compute_zscore_stats

    ys = torch.tensor([[78.0, 592.0]])
    kdas = torch.tensor([25.85])
    target_mean, target_std, kda_mean, kda_std = compute_zscore_stats(ys, kdas)
    # With N=1 the std would be 0 -> clamp_min(1.0) keeps it usable
    assert (target_std >= 1.0).all()
    assert kda_std.item() >= 1.0
    assert target_mean.shape == (2,)


def test_lit_module_one_training_step_a(tmp_path):
    """End-to-end: one training step on the train sample for FPNetA."""
    import pytorch_lightning as pl
    from fp_gnn.model import FPNetA
    from fp_gnn.train import FluorLitModule

    ds = _load_dataset(tmp_path)
    train_ds = [d for d in ds if d.split == "train"]
    test_ds = [d for d in ds if d.split == "test"]

    model = FluorLitModule(
        net=FPNetA(node_embedding_dim=32, num_message_steps=2),
        train_dataset=train_ds,
        val_dataset=test_ds,
        test_dataset=test_ds,
        batch_size=1,
        lr=1e-3,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(model)
    # Loss should be finite after 1 epoch
    assert torch.isfinite(torch.tensor(trainer.callback_metrics["train_loss"].item()))


def test_lit_module_one_training_step_b(tmp_path):
    """End-to-end: one training step on the train sample for FPNetB."""
    import pytorch_lightning as pl
    from fp_gnn.model import FPNetB
    from fp_gnn.train import FluorLitModule

    ds = _load_dataset(tmp_path)
    train_ds = [d for d in ds if d.split == "train"]
    test_ds = [d for d in ds if d.split == "test"]

    model = FluorLitModule(
        net=FPNetB(node_embedding_dim=32, num_message_steps=2),
        train_dataset=train_ds,
        val_dataset=test_ds,
        test_dataset=test_ds,
        batch_size=1,
        lr=1e-3,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(model)
    assert torch.isfinite(torch.tensor(trainer.callback_metrics["train_loss"].item()))
```

- [ ] **Step 2: Run — must fail**

```bash
uv run pytest tests/test_smoke_train.py -v -k "zscore or lit_module"
```

Expected: ImportError on `fp_gnn.train`.

- [ ] **Step 3: Implement `FluorLitModule` and helpers**

Create `src/fp_gnn/train.py`:

```python
"""Lightning training module + z-score helpers + CLI entry point."""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from fp_gnn.dataset import FluorProteinDataset
from fp_gnn.model import FPNetA, FPNetB


def compute_zscore_stats(targets, kdas):
    """Compute (target_mean, target_std, kda_mean, kda_std) with std
    clamped to >= 1.0 to avoid div-by-zero on tiny train splits.

    Args:
        targets: float tensor [N, 2] -- (brightness, emission) per sample
        kdas:    float tensor [N]    -- kDa per sample
    """
    target_mean = targets.mean(dim=0)
    target_std = targets.std(dim=0, unbiased=False).clamp_min(1.0)
    kda_mean = kdas.mean()
    kda_std = kdas.std(unbiased=False).clamp_min(1.0)
    return target_mean, target_std, kda_mean, kda_std


class FluorLitModule(pl.LightningModule):
    def __init__(
        self,
        net,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=1,
        lr=1e-3,
    ):
        super().__init__()
        self.net = net
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.lr = lr

        # Stats from the train split only
        ys = torch.cat([d.y for d in train_dataset], dim=0)         # [N_train, 2]
        kdas = torch.cat([d.kda for d in train_dataset], dim=0)     # [N_train]
        target_mean, target_std, kda_mean, kda_std = compute_zscore_stats(ys, kdas)

        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.register_buffer("kda_mean", kda_mean)
        self.register_buffer("kda_std", kda_std)

    def _attach_kda_z(self, batch):
        batch.kda_z = (batch.kda - self.kda_mean) / self.kda_std

    def _zscore_targets(self, y):
        return (y - self.target_mean) / self.target_std

    def _denormalize(self, y_z):
        return y_z * self.target_std + self.target_mean

    def training_step(self, batch, batch_idx):
        self._attach_kda_z(batch)
        pred_z = self.net(batch)
        y_z = self._zscore_targets(batch.y)
        loss = F.mse_loss(pred_z, y_z)
        self.log("train_loss", loss, batch_size=self.batch_size)
        return loss

    def _eval_step(self, batch, prefix):
        self._attach_kda_z(batch)
        pred_z = self.net(batch)
        pred = self._denormalize(pred_z)
        # MSE / MAE per target in original units
        diff = pred - batch.y
        mse = (diff ** 2).mean(dim=0)
        mae = diff.abs().mean(dim=0)
        self.log(f"{prefix}_mse_brightness", mse[0], batch_size=self.batch_size)
        self.log(f"{prefix}_mse_emission", mse[1], batch_size=self.batch_size)
        self.log(f"{prefix}_mae_brightness", mae[0], batch_size=self.batch_size)
        self.log(f"{prefix}_mae_emission", mae[1], batch_size=self.batch_size)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
```

- [ ] **Step 4: Run all train tests**

```bash
uv run pytest tests/test_smoke_train.py -v
```

Expected: 6 passed (3 model forward + zscore + 2 lit-module).

- [ ] **Step 5: Commit**

```bash
git add src/fp_gnn/train.py tests/test_smoke_train.py
git commit -m "feat(train): FluorLitModule with z-score buffers and per-target metrics"
```

---

### Task 15: `train.py` — CLI entry point with `--model {a,b}`

**Files:**
- Modify: `src/fp_gnn/train.py`

- [ ] **Step 1: Append CLI entry point to train.py**

Append to `src/fp_gnn/train.py`:

```python
def _build_net(model_name, hidden=64, steps=3):
    if model_name == "a":
        return FPNetA(node_embedding_dim=hidden, num_message_steps=steps)
    if model_name == "b":
        return FPNetB(node_embedding_dim=hidden, num_message_steps=steps)
    raise ValueError(f"Unknown model: {model_name!r} (choose 'a' or 'b')")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["a", "b"], default="a")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--data-root", type=str, default="data/processed")
    parser.add_argument("--labels", type=str, default="data/labels.csv")
    parser.add_argument("--repo-root", type=str, default=".")
    args = parser.parse_args()

    torch.manual_seed(0)
    import numpy as np
    import random
    np.random.seed(0)
    random.seed(0)

    ds = FluorProteinDataset(
        root=args.data_root, labels_csv=args.labels, repo_root=args.repo_root,
    )
    train_ds = [d for d in ds if d.split == "train"]
    test_ds = [d for d in ds if d.split == "test"]

    net = _build_net(args.model, hidden=args.hidden, steps=args.steps)
    lit = FluorLitModule(
        net=net,
        train_dataset=train_ds,
        val_dataset=test_ds,
        test_dataset=test_ds,
        batch_size=1,
        lr=args.lr,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=CSVLogger(save_dir="logs", name=f"model_{args.model}"),
        enable_progress_bar=True,
    )
    trainer.fit(lit)
    trainer.test(lit)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the CLI for model A (3 epochs only, just to confirm wiring)**

```bash
uv run python -m fp_gnn.train --model a --max-epochs 3
```

Expected: Lightning prints "Trainer is using ... Adam", logs are written under `logs/model_a/version_*/metrics.csv`. No exceptions.

- [ ] **Step 3: Smoke-run the CLI for model B**

```bash
uv run python -m fp_gnn.train --model b --max-epochs 3
```

Expected: same as above but under `logs/model_b/...`.

- [ ] **Step 4: Verify CSV logs exist for both architectures**

```bash
ls logs/model_a/ logs/model_b/
```

Expected: at least one `version_*` directory under each.

- [ ] **Step 5: Commit**

```bash
git add src/fp_gnn/train.py
git commit -m "feat(train): CLI entry point with --model {a,b} and CSV logging"
```

---

### Task 16: Final smoke check + README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest -v
```

Expected: all tests pass (~13 graph tests + 6 train tests).

- [ ] **Step 2: Write a minimal README**

Create `/home/l-braz/Documents/git/pdb_project_2/README.md`:

````markdown
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
# Architecture A (disjoint baseline)
uv run python -m fp_gnn.train --model a --max-epochs 60

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
````

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup, train, and test instructions"
```

- [ ] **Step 4: Final summary**

Confirm by running:

```bash
git log --oneline
uv run pytest --tb=line -q
```

Expected:
- Linear commit history of ~16–18 commits.
- All tests pass.
- The pipeline can be invoked end-to-end via `python -m fp_gnn.train --model a` and `--model b`.

---

## Self-Review

**Spec coverage check:**

| Spec section | Implementing tasks |
|---|---|
| §4 Repo layout | Task 1 (bootstrap) |
| §5 Tooling (uv, deps, CSVLogger, no W&B) | Task 1, Task 15 |
| §6 PDB ingestion | Tasks 3, 4, 5 |
| §7a Chromophore graph (OGB features, pos attr, distance side-channel) | Tasks 6, 7 |
| §7b Protein graph (8 Å, 20-D one-hot) | Task 8 |
| §7c Cross-edges (6 Å) | Task 9 |
| §8 Dataset (FPData + InMemoryDataset + __inc__ for both edge index types) | Task 10 |
| §9a-b ChromMPNN, ProteinMPNN | Task 11 |
| §9c FPNetA | Task 12 |
| §9d FPNetB | Task 13 |
| §10 Training (LitModule, z-score, MSE, Adam, denormalized metrics) | Tasks 14, 15 |
| §11 Tests (test_graphs.py + test_smoke_train.py for both archs) | Across all tasks |
| §13 Decisions log | Encoded by code choices throughout |

**Placeholder scan:** None. Every step has a concrete command or full code block.

**Type/name consistency check:**
- `FluorProteinDataset` — used in tasks 10, 11, 12, 13, 14, 15.
- `FPData` — defined in task 10, used implicitly by all later tasks via the dataset.
- `ChromMPNN`, `ProteinMPNN` — defined in task 11, composed inside FPNetA (task 12) and partially mirrored in FPNetB (task 13).
- `FPNetA` / `FPNetB` — defined in tasks 12/13, instantiated in task 14 tests and task 15 CLI.
- `FluorLitModule` — defined in task 14, used in task 14 tests and task 15 CLI.
- Attribute names on `FPData`: `x`, `edge_index`, `edge_attr`, `chrom_x`, `chrom_edge_index`, `chrom_edge_attr_chem`, `chrom_edge_attr_dist`, `cross_edge_index`, `cross_edge_attr`, `kda`, `y`, `pdb_code`, `split` — used consistently across tasks 10, 11, 12, 13, 14.
- `kda_z` — written by `FluorLitModule._attach_kda_z` (task 14) and by tests in tasks 12/13; read by `FPNetA.forward` and `FPNetB.forward`.

**Scope check:** This plan implements one self-contained subsystem (the GNN pipeline). It does not need to be split.

**One known fragility worth flagging at execution time:** `AllChem.AssignBondOrdersFromTemplate` can be picky if the PDB block has unexpected atom ordering. If task 7's tests fail at this line, the pragmatic recovery is to fall back to `Chem.MolFromPDBBlock(block, sanitize=True, removeHs=False)` and accept inferred bond orders. This is not in scope for v1, but worth noting.
