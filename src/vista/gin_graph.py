from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


HYBRIDIZATIONS = ["SP","SP2","SP3","SP3D","SP3D2","OTHER"]
CHIRAL_TAGS = ["CHI_UNSPECIFIED","CHI_TETRAHEDRAL_CW","CHI_TETRAHEDRAL_CCW","OTHER"]
BOND_TYPES = ["SINGLE","DOUBLE","TRIPLE","AROMATIC","OTHER"]
BOND_STEREO = ["STEREONONE","STEREOZ","STEREOE","OTHER"]
EDGE_DIM = len(BOND_TYPES) + 2 + len(BOND_STEREO)


def atom_features(atom) -> List[float]:
    from rdkit.Chem.rdchem import HybridizationType, ChiralType

    hyb_name = str(atom.GetHybridization())
    if hyb_name not in HYBRIDIZATIONS[:-1]:
        hyb_name = "OTHER"
    hyb = [1.0 if hyb_name == x else 0.0 for x in HYBRIDIZATIONS]

    chiral_name = str(atom.GetChiralTag())
    if chiral_name not in CHIRAL_TAGS[:-1]:
        chiral_name = "OTHER"
    chiral = [1.0 if chiral_name == x else 0.0 for x in CHIRAL_TAGS]

    return [
        atom.GetAtomicNum() / 100.0,
        min(atom.GetDegree(), 8) / 8.0,
        np.clip(atom.GetFormalCharge(), -4, 4) / 4.0,
        min(atom.GetTotalNumHs(), 8) / 8.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        atom.GetMass() / 250.0,
        *hyb,
        *chiral,
    ]


def bond_features(bond) -> List[float]:
    bt = str(bond.GetBondType())
    if bt not in BOND_TYPES[:-1]:
        bt = "OTHER"
    bt_oh = [1.0 if bt == x else 0.0 for x in BOND_TYPES]

    st = str(bond.GetStereo())
    if st not in BOND_STEREO[:-1]:
        st = "OTHER"
    st_oh = [1.0 if st == x else 0.0 for x in BOND_STEREO]

    return [
        *bt_oh,
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        *st_oh,
    ]


def smiles_to_graph(smiles: str):
    from rdkit import Chem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed SMILES: {smiles}")

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)

    edges = []
    edge_attr = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ef = bond_features(b)
        edges.extend([(i,j),(j,i)])
        edge_attr.extend([ef,ef])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,EDGE_DIM), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class ManifestGraphDataset(torch.utils.data.Dataset):
    """Row-level dataset over a frozen manifest.

    Repeated canonical linkers remain repeated rows for fair sample-weighted
    comparison with the image models. Graph templates are cached by canonical SMILES.
    """

    def __init__(self, frame, label_columns):
        self.frame = frame.reset_index(drop=True).copy()
        self.label_columns = list(label_columns)
        self._cache: Dict[str,object] = {}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        from torch_geometric.data import Data
        row = self.frame.iloc[int(idx)]
        smi = str(row["canonical_first_linker_smiles"])
        if smi not in self._cache:
            self._cache[smi] = smiles_to_graph(smi)
        base = self._cache[smi]
        d = Data(x=base.x.clone(), edge_index=base.edge_index.clone(), edge_attr=base.edge_attr.clone())
        d.y = torch.tensor(row[self.label_columns].to_numpy(dtype=np.float32), dtype=torch.float32).view(1,-1)
        d.row_index = torch.tensor([int(idx)], dtype=torch.long)
        return d


class GINMultiLabel(nn.Module):
    """Single GIN-family graph baseline implemented with edge-aware GINEConv.

    Uses rich atom features plus bond type, conjugation, ring flag, and
    stereochemistry. This remains one predeclared GIN family rather than a graph
    model zoo.
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int, dropout: float, n_labels: int = 10):
        super().__init__()
        from torch_geometric.nn import GINEConv

        self.n_labels = int(n_labels)
        self.dropout_p = float(dropout)

        self.input = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(int(n_layers)):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=EDGE_DIM))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(hidden_dim, self.n_labels),
        )

    def forward(self, data):
        from torch_geometric.nn import global_mean_pool
        x = self.input(data.x)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, data.edge_index, data.edge_attr)
            x = norm(x)
            x = torch.relu(x)
            x = torch.dropout(x, self.dropout_p, self.training)
        g = global_mean_pool(x, data.batch)
        return self.head(g)
