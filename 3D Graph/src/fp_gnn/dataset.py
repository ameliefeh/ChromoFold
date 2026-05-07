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
