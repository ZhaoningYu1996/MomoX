import torch
from model import GCN
from tu2smiles import to_tudataset
from utils import get_mol, sanitize

PATH = 'checkpoints/best_model_mutagenicity.pt'
target_model = GCN(hidden_channels=64, input_channels=14, output_channels=2).cuda()
target_model.load_state_dict(torch.load(PATH))
target_model.eval()

lines = []

# Open the file and read lines
with open('vocab_0.txt', 'r') as file:
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    
# print(lines)

mask = []
sm = torch.nn.Softmax(dim=1)
for i, smiles in enumerate(lines):
    mol = get_mol(smiles, False)
    mol = sanitize(mol, False)
    data = to_tudataset(mol, "Mutagenicity")
    data.cuda()
    batch = torch.zeros(data.x.size(0), dtype=torch.int64).cuda()
    out = target_model(data.x, data.edge_index, batch)
    pred = sm(out)
    if pred[0,0] > 0.9:
        mask.append(i)
full_mask = torch.tensor([i for _ in range(len(lines))], dtype=torch.int64)
print(mask)
# mask = [i for i in range(165)]
torch.save(mask, "checkpoints/mask_0.pt")
torch.save(full_mask, "checkpoints/full_mask_0.pt")
print(len(mask))
print(len(full_mask))

lines = []

# Open the file and read lines
with open('vocab_1.txt', 'r') as file:
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    
# print(lines)

mask = []
sm = torch.nn.Softmax(dim=1)
for i, smiles in enumerate(lines):
    mol = get_mol(smiles, False)
    mol = sanitize(mol, False)
    data = to_tudataset(mol, "Mutagenicity")
    data.cuda()
    batch = torch.zeros(data.x.size(0), dtype=torch.int64).cuda()
    out = target_model(data.x, data.edge_index, batch)
    pred = sm(out)
    if pred[0,1] > 0.9:
        print(smiles)
        mask.append(i)
print(mask)
full_mask = torch.tensor([i for _ in range(len(lines))], dtype=torch.int64)
# mask = [i for i in range(113)]
torch.save(mask, "checkpoints/mask_1.pt")
torch.save(full_mask, "checkpoints/full_mask_1.pt")
print(len(mask))
print(len(full_mask))