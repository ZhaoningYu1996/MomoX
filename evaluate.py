import rdkit.Chem as Chem
from model import GCN
import torch
from utils import get_mol, sanitize
from tu2smiles import to_tudataset

def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return (mol is not None)

# file_path = 'mol_samples_0_2.txt'  # Replace with your file path

# with open(file_path, 'r') as file:
#     lines = [line.strip() for line in file]
    
# print(lines[:3])
# smiles_set = set([])
# valid_count = 0

# for smiles in lines:
#     smiles_set.add(smiles)
#     valid_count += is_valid_smiles(smiles)

# print(len(smiles_set)/len(lines))
# print(valid_count/len(lines))
    
PATH = 'checkpoints/best_model_mutagenicity.pt'
target_model = GCN(hidden_channels=64, input_channels=14, output_channels=2).cuda()
target_model.load_state_dict(torch.load(PATH))
target_model.eval()

# dataset = []
# dataset_smiles = []

# for smiles in smiles_set:
#     mol = get_mol(smiles, False)
#     mol = sanitize(mol, False)
#     data = to_tudataset(mol, "Mutagenicity")
#     dataset.append(data)
#     dataset_smiles.append(smiles)

# print(len(dataset))
# softmax = torch.nn.Softmax(dim=1)
# all_logit = 0.0

# count = 0
# for i, data in enumerate(dataset):
#     if data == None:
#         count += 1
#         continue
#     data.cuda()
#     batch = torch.zeros(data.x.size(0), dtype=torch.int64).cuda()
#     cand_pred = target_model(data.x, data.edge_index, batch, return_embedding = False)
#     logit = softmax(cand_pred)[0, 0].item()
#     if logit < 0.9:
#         print(dataset_smiles[i])
#     all_logit += logit

# print(all_logit/(len(dataset)-count))
    
file_path = 'mol_samples_1_0.txt'  # Replace with your file path

with open(file_path, 'r') as file:
    lines = [line.strip() for line in file]

smiles_set = set([])
valid_count = 0

for smiles in lines:
    smiles_set.add(smiles)
    valid_count += is_valid_smiles(smiles)

print(len(smiles_set)/len(lines))
print(valid_count/len(lines))
dataset = []
dataset_smiles = []

for smiles in smiles_set:
    mol = get_mol(smiles, False)
    mol = sanitize(mol, False)
    data = to_tudataset(mol, "Mutagenicity")
    dataset.append(data)
    dataset_smiles.append(smiles)

softmax = torch.nn.Softmax(dim=1)
all_logit = 0.0

count = 0
for i, data in enumerate(dataset):
    if data == None:
        count += 1
        continue
    data.cuda()
    batch = torch.zeros(data.x.size(0), dtype=torch.int64).cuda()
    cand_pred = target_model(data.x, data.edge_index, batch, return_embedding = False)
    logit = softmax(cand_pred)[0, 1].item()
    if logit < 0.9:
        print(dataset_smiles[i])
    all_logit += logit

print(all_logit/(len(dataset)-count))