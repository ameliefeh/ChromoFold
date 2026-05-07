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

    # Defensive: exclude any zero-distance pair (should never occur for real
    # PDB data, since Cα and chromophore atoms belong to distinct molecules,
    # but the guard is cheap and prevents a degenerate edge if a parsing bug
    # ever produces colocated coordinates).
    res_idx, atom_idx = np.where((dist <= cutoff) & (dist > 0.0))

    cross_edge_index = torch.tensor(
        np.stack([res_idx, atom_idx], axis=0), dtype=torch.long
    )
    cross_edge_attr = torch.tensor(
        dist[res_idx, atom_idx], dtype=torch.float
    ).unsqueeze(-1)
    return cross_edge_index, cross_edge_attr
