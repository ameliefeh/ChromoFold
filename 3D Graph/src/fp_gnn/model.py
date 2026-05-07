import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import MLP, NNConv, global_add_pool


def _make_edge_network(emb_dim, hidden=None):
    h = hidden or 2 * emb_dim
    return MLP([emb_dim, h, emb_dim * emb_dim], norm=None)


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

        # PyG does NOT auto-create chrom_x_batch for non-default node attributes.
        # Reconstruct it from _slice_dict['chrom_x'] when batched, else fall back
        # to single-graph (all atoms belong to graph 0).
        chrom_batch = getattr(batch, "chrom_x_batch", None)
        if chrom_batch is None:
            slice_dict = getattr(batch, "_slice_dict", None)
            if slice_dict is not None and "chrom_x" in slice_dict:
                slices = slice_dict["chrom_x"].to(node_state.device)
                sizes = slices[1:] - slices[:-1]
                num_graphs = len(sizes)
                chrom_batch = torch.repeat_interleave(
                    torch.arange(num_graphs, device=node_state.device), sizes
                )
            else:
                # Single-graph (non-batched) case
                chrom_batch = torch.zeros(
                    node_state.shape[0], dtype=torch.long, device=node_state.device
                )
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


class FPNetA(nn.Module):
    """Disjoint two-graph baseline: ProteinMPNN || ChromMPNN || kda -> MLP."""

    def __init__(self, node_embedding_dim=64, num_message_steps=3):
        super().__init__()
        H = node_embedding_dim
        self.protein_mpnn = ProteinMPNN(H, num_message_steps)
        self.chrom_mpnn = ChromMPNN(H, num_message_steps)
        # Head: (2H + 1) -> H -> 2
        self.head = MLP([2 * H + 1, H, 2], norm=None)

    def forward(self, batch):
        prot_emb = self.protein_mpnn(batch)
        chrom_emb = self.chrom_mpnn(batch)
        fused = torch.cat(
            [prot_emb, chrom_emb, batch.kda_z.view(-1, 1)], dim=-1
        )
        return self.head(fused)


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

        self.head = MLP([2 * H + 1, H, 2], norm=None)

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

        # PyG does NOT auto-create chrom_x_batch for non-default node attributes.
        # Reconstruct it from _slice_dict['chrom_x'] when batched, else fall back
        # to single-graph (all atoms belong to graph 0).
        chrom_batch = getattr(batch, "chrom_x_batch", None)
        if chrom_batch is None:
            slice_dict = getattr(batch, "_slice_dict", None)
            if slice_dict is not None and "chrom_x" in slice_dict:
                slices = slice_dict["chrom_x"].to(node_a.device)
                sizes = slices[1:] - slices[:-1]
                num_graphs = len(sizes)
                chrom_batch = torch.repeat_interleave(
                    torch.arange(num_graphs, device=node_a.device), sizes
                )
            else:
                # Single-graph (non-batched) case
                chrom_batch = torch.zeros(
                    node_a.shape[0], dtype=torch.long, device=node_a.device
                )

        prot_emb = global_add_pool(node_r, prot_batch)
        chrom_emb = global_add_pool(node_a, chrom_batch)

        fused = torch.cat([prot_emb, chrom_emb, batch.kda_z.view(-1, 1)], dim=-1)
        return self.head(fused)
