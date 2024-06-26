from rdkit.Chem import rdchem
from typing import Any
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from utils import get_smiles, sanitize
from rdkit.Chem import rdmolops
from rdkit import Chem

ATOM = {
    'MUTAG': {
        0: 6,
        1: 7,
        2: 8,
        3: 9,
        4: 53,
        5: 17,
        6: 35,
    },
    
    'PTC_MR': {
        0: 49,
        1: 15,
        2: 8,
        3: 7,
        4: 11,
        5: 6,
        6: 17,
        7: 16,
        8: 35,
        9: 9,
        10: 19,
        11: 29,
        12: 30,
        13: 53,
        14: 56,
        15: 50,
        16: 82,
        17: 20,
    },

    'PTC_FR': {
        0: 49,
        1: 15,
        2: 8,
        3: 7,
        4: 11,
        5: 6,
        6: 17,
        7: 16,
        8: 35,
        9: 9,
        10: 33,
        11: 19,
        12: 29,
        13: 30,
        14: 53,
        15: 50,
        16: 82,
        17: 52,
        18: 20,
    },

    'PTC_MM': {
        0: 49,
        1: 15,
        2: 8,
        3: 7,
        4: 11,
        5: 6,
        6: 17,
        7: 16,
        8: 35,
        9: 9,
        10: 33,
        11: 19,
        12: 5,
        13: 29,
        14: 30,
        15: 53,
        16: 56,
        17: 50,
        18: 82,
        19: 20,
    },

    'PTC_FM': {
        0: 49,
        1: 15,
        2: 6,
        3: 8,
        4: 7,
        5: 17,
        6: 16,
        7: 35,
        8: 11,
        9: 9,
        10: 33,
        11: 19,
        12: 29,
        13: 53,
        14: 56,
        15: 50,
        16: 82,
        17: 20,
    },
    
    'Mutagenicity': {
        0: 6,
        1: 8,
        2: 17,
        3: 1,
        4: 7,
        5: 9,
        6: 35,
        7: 16,
        8: 15,
        9: 53,
        10: 11,
        11: 19,
        12: 3,
        13: 20,
    },
    
    'COX2_MD': {
        0: 6,
        1: 7,
        2: 9,
        3: 16,
        4: 8,
        5: 17,
        6: 35,
    },

    'BZR_MD': {
        0: 6,
        1: 7,
        2: 8,
        3: 9,
        4: 17,
        5: 16,
        6: 15,
        7: 35,
    },

    'DHFR_MD': {
        0: 7,
        1: 6,
        2: 17,
        3: 8,
        4: 9,
        5: 16,
        6: 35,
    },

    'ER_MD': {
        0: 6,
        1: 8,
        2: 7,
        3: 17,
        4: 16,
        5: 9,
        6: 35,
        7: 14,
        8: 53,
        9: 15,
    },
    
}

EDGE = {
    'MUTAG': {
        0: rdchem.BondType.AROMATIC,
        1: rdchem.BondType.SINGLE,
        2: rdchem.BondType.DOUBLE,
        3: rdchem.BondType.TRIPLE,
    },

    'PTC_MR': {
        0: rdchem.BondType.TRIPLE,
        1: rdchem.BondType.DOUBLE,
        2: rdchem.BondType.SINGLE,
        3: rdchem.BondType.AROMATIC,
    },

    'PTC_FR': {
        0: rdchem.BondType.TRIPLE,
        1: rdchem.BondType.DOUBLE,
        2: rdchem.BondType.SINGLE,
        3: rdchem.BondType.AROMATIC,
    },

    'PTC_MM': {
        0: rdchem.BondType.TRIPLE,
        1: rdchem.BondType.DOUBLE,
        2: rdchem.BondType.SINGLE,
        3: rdchem.BondType.AROMATIC,
    },

    'PTC_FM': {
        0: rdchem.BondType.TRIPLE,
        1: rdchem.BondType.SINGLE,
        2: rdchem.BondType.DOUBLE,
        3: rdchem.BondType.AROMATIC,
    },
    
    'Mutagenicity': {
        0: rdchem.BondType.SINGLE,
        1: rdchem.BondType.DOUBLE,
        2: rdchem.BondType.TRIPLE,
    },
    
    'COX2_MD': {
        0: rdchem.BondType.AROMATIC,
        1: None,
        2: rdchem.BondType.SINGLE,
        3: rdchem.BondType.DOUBLE,
        4: rdchem.BondType.TRIPLE,
    },

    'BZR_MD': {
        0: rdchem.BondType.AROMATIC,
        1: None,
        2: rdchem.BondType.SINGLE,
        3: rdchem.BondType.DOUBLE,
        4: rdchem.BondType.TRIPLE,
    },

    'DHFR_MD': {
        0: rdchem.BondType.AROMATIC,
        1: None,
        2: rdchem.BondType.SINGLE,
        3: rdchem.BondType.DOUBLE,
        4: rdchem.BondType.TRIPLE,
    },

    'ER_MD': {
        0: rdchem.BondType.AROMATIC,
        1: None,
        2: rdchem.BondType.SINGLE,
        3: rdchem.BondType.DOUBLE,
        4: rdchem.BondType.TRIPLE,
    },
}

def to_smiles(data: 'torch_geometric.data.Data',
              kekulize: bool = True, data_name: str = 'MUTAG') -> Any:
    """Converts a :class:`torch_geometric.data.Data` instance to a SMILES
    string.

    Args:
        data (torch_geometric.data.Data): The molecular graph.
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
        data_name: The name of dataset
    """
    from rdkit import Chem

    mol = Chem.RWMol()

    for i in range(data.num_nodes):
        # Some dataset does not have 
        if data_name in ["COX2", "BZR", "NCI1"]:
            atom = rdchem.Atom(torch.argmax(data.x[i]).item()+1)
        else:
            atom = rdchem.Atom(ATOM[data_name][torch.argmax(data.x[i]).item()])
        mol.AddAtom(atom)
    edges = [tuple(i) for i in data.edge_index.t().tolist()]
    visited = set()
    deleted = []
    # print(mol.GetNumAtoms())
    # print(f"Data: {data}")
    # print(f"Data attribute: {data.keys}")
    
    for i in range(len(edges)):
        src, dst = edges[i]
        if tuple(sorted(edges[i])) in visited:
            continue
        if "edge_attr" in data.keys():
            # print(stop)
            bond_type = EDGE[data_name][torch.argmax(data.edge_attr[i]).item()]
            if bond_type == None:
                deleted.append(tuple(edges[i]))
            else:
                mol.AddBond(src, dst, bond_type)
        else:
            mol.AddBond(src, dst)

        visited.add(tuple(sorted(edges[i])))

    mol = mol.GetMol()

    # try:
    #     Chem.SanitizeMol(mol)
    # except Chem.rdchem.KekulizeException as e:
    #     print("KekulizeException:", e)
    # if kekulize:
    #     Chem.Kekulize(mol, clearAromaticFlags=True)
    mol = sanitize(mol, False)
    if mol is None:
        # import ipdb; ipdb.set_trace()
        # print(stop)
        return None
    # return get_smiles(mol)
    # Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol)

    return Chem.MolToSmiles(mol, isomericSmiles=True)

def to_tudataset(mol, data_name, label=None):
    if mol == None:
        return None
    if mol.GetNumAtoms() == 0 and mol.GetNumBonds() == 0:
        return None
    rdmolops.AssignStereochemistry(mol)
    # if addH == True:
    #     mol = Chem.AddHs(mol)
    # Extract atom-level features
    atom_features = []
    swapped_feature_map = {value: key for key, value in ATOM[data_name].items()}
    for atom in mol.GetAtoms():
        atom_features.append(swapped_feature_map[atom.GetAtomicNum()])
    

    # Extract bond-level features
    bond_features = []
    swapped_edge_feature_map = {value: key for key, value in EDGE[data_name].items()}
    for bond in mol.GetBonds():
        bond_type = bond.GetBondTypeAsDouble()
        if bond_type == 1.0:
            bond_feat = swapped_edge_feature_map[rdchem.BondType.SINGLE]
        elif bond_type == 1.5:
            bond_feat = swapped_edge_feature_map[rdchem.BondType.AROMATIC]
        elif bond_type == 2.0:
            bond_feat = swapped_edge_feature_map[rdchem.BondType.DOUBLE]
        elif bond_type == 3.0:
            bond_feat = swapped_edge_feature_map[rdchem.BondType.TRIPLE]
        else:
            bond_type = swapped_edge_feature_map[None]

        bond_features.append(bond_feat)
    
    atom_features = torch.tensor(atom_features, dtype=torch.long)
    x = F.one_hot(atom_features, num_classes=len(swapped_feature_map)).float()  # Node feature matrix
    edge_index = []
    for bond in mol.GetBonds():
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Edge connectivity
    if mol.GetNumBonds() == 0:
        edge_index = torch.tensor([[], []], dtype=torch.long)
    bond_features = torch.tensor(bond_features, dtype=torch.long)  # Edge feature matrix
    edge_attr = F.one_hot(bond_features, num_classes=len(swapped_edge_feature_map))
    if not label == None:
        y = torch.tensor([label])
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    else:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data