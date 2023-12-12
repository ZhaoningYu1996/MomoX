import torch
from torch_geometric.datasets import TUDataset
from tu2smiles import to_smiles
from utils import sanitize_smiles, get_mol, sanitize, get_tree, tensorize, get_clique_mol, get_smiles
from model import GCN, GCN2, GAT
from tqdm import tqdm
from mol_tree import MolTree
import pickle

data_name = "Mutagenicity"

motif_list = torch.load("checkpoints/motif_list.pt")
# print(motif_list)
print(motif_list[0])

motif_0 = torch.load("checkpoints/motif_0.pt")
motif_1 = torch.load("checkpoints/motif_1.pt")
# print(len(motif_0))
# print(len(motif_1))
# print(stop)

train_dataset = torch.load("checkpoints/train_dataset_mutagenicity.pt")

PATH = 'checkpoints/best_model_mutagenicity.pt'
target_model = GCN(hidden_channels=64, input_channels=14, output_channels=2)
target_model.load_state_dict(torch.load(PATH))
target_model.eval()

train_list = []
train_smiles = []

for i, data in enumerate(train_dataset):
    smiles = to_smiles(data, False, data_name)
    smiles = sanitize_smiles(smiles)
    if not smiles == None:
        batch = torch.zeros(data.x.size(0), dtype=torch.int64)
        out = target_model(data.x, data.edge_index, batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        if pred == data.y:
            train_list.append(data)
            train_smiles.append(smiles)

print(len(train_list))
if len(train_list) != len(motif_list):
    print("Error!")

count = 0
all_possible_combination_0 = []
all_possible_combination_1 = []
all_train_smiles_0 = []
all_train_smiles_1 = []
for i, data in enumerate(tqdm(train_list)):
    label = data.y.item()
    motifs = motif_list[i]
    check = 0
    motif_nodes = []
    if label == 0:
        for motif in motifs.keys():
            if motif in motif_0:
                motif_nodes.append(motifs[motif])
                check = 1
    if label == 1:
        for motif in motifs.keys():
            if motif in motif_1:
                motif_nodes.append(motifs[motif])
                check = 1
    # ignore data without important motifs
    if check == 0:
        continue
    possible_combination = [[]]

    for j in range(len(motif_nodes)):
        for motif in motif_nodes[j]:
            # print(motif_nodes[j])
            # print(motif)
            check_added = 0
            for po_list in possible_combination:
                check_intersection = 0
                if len(po_list) == 0:
                    po_list.append(motif)
                    check_added = 1
                for prev_motif in po_list:
                    if len(set(motif).intersection(set(prev_motif))) != 0:
                        check_intersection = 1
                if check_intersection == 0:
                    po_list.append(motif)
                    check_added = 1
            if check_added == 0:
                possible_combination.append([motif])
    if label == 0:
        all_train_smiles_0.append(train_smiles[i])
        all_possible_combination_0.append(possible_combination)
    elif label == 1:
        all_train_smiles_1.append(train_smiles[i])
        all_possible_combination_1.append(possible_combination)

all_train_data_0 = []
all_train_data_1 = []
extended_motif_0 = []
extended_motif_1 = []
for i, possible_combination in enumerate(tqdm(all_possible_combination_0)):
    # print(possible_combination)
    smiles = all_train_smiles_0[i]
    mol = get_mol(smiles, False)
    mol = sanitize(mol, False)
    n_atoms = mol.GetNumAtoms()
    ori_edges = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        ori_edges.append([a1,a2])

    for combination in possible_combination:
        clique_set = []
        cliques = []
        cliques.extend(combination)
        for motif in combination:
            clique_set.extend(motif)
        for a1, a2 in ori_edges:
            if a1 not in clique_set or a2 not in clique_set:
                cliques.append([a1, a2])
        cliques, edges = get_tree(cliques, n_atoms)
        for clique in cliques:
            cmol = get_clique_mol(mol, clique)
            node_smiles = sanitize_smiles(get_smiles(cmol))
            if node_smiles not in motif_0:
                extended_motif_0.append(node_smiles)
        mol_tree = MolTree(smiles, cliques, edges, motif_0)
        mol_tree = tensorize(mol_tree)
        all_train_data_0.append(mol_tree)

for i, possible_combination in enumerate(tqdm(all_possible_combination_1)):
    # print(possible_combination)
    smiles = all_train_smiles_1[i]
    mol = get_mol(smiles, False)
    mol = sanitize(mol, False)
    n_atoms = mol.GetNumAtoms()
    ori_edges = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        ori_edges.append([a1,a2])

    for combination in possible_combination:
        clique_set = []
        cliques = []
        cliques.extend(combination)
        for motif in combination:
            clique_set += motif
        for a1, a2 in ori_edges:
            if a1 not in clique_set or a2 not in clique_set:
                cliques.append([a1, a2])
        cliques, edges = get_tree(cliques, n_atoms)
        for clique in cliques:
            cmol = get_clique_mol(mol, clique)
            node_smiles = sanitize_smiles(get_smiles(cmol))
            if node_smiles not in motif_1:
                extended_motif_1.append(node_smiles)
        mol_tree = MolTree(smiles, cliques, edges, motif_1)
        mol_tree = tensorize(mol_tree)
        all_train_data_1.append(mol_tree)

motif_0.extend(list(set(extended_motif_0)))
motif_1.extend(list(set(extended_motif_1)))

file_path_0 = 'vocab_0.txt'
file_path_1 = 'vocab_1.txt'

# Open the file in write mode ('w') or append mode ('a') if you want to keep existing content
with open(file_path_0, 'w') as file:
    for string in motif_0:
        # Write each string to a separate line
        file.write(f"{string}\n")

with open(file_path_1, 'w') as file:
    for string in motif_1:
        # Write each string to a separate line
        file.write(f"{string}\n")

num_splits = 1
print(len(all_train_data_0))
le = int((len(all_train_data_0)) / num_splits)

for split_id in range(num_splits):
    st = int(split_id * le)
    sub_data = all_train_data_0[st : st + le]

    with open('processed_data/Mutagenicity_0/tensors-label-0-%d.pkl' % split_id, 'wb') as f:
        pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
        
num_splits = 1
print(len(all_train_data_1))
le = int((len(all_train_data_1)) / num_splits)

for split_id in range(num_splits):
    st = int(split_id * le)
    sub_data = all_train_data_1[st : st + le]

    with open('processed_data/Mutagenicity_1/tensors-label-1-%d.pkl' % split_id, 'wb') as f:
        pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

# print(len(all_possible_combination))
# print(all_possible_combination[:5])