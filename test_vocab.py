import torch
from model import GCN
from tu2smiles import to_tudataset
from utils import get_mol, sanitize

data_name ="Mutagenicity"
hid_dim=16
input_dim=14
PATH = "checkpoints/best_model_"+data_name+".pt"
target_model = GCN(hidden_channels=hid_dim, input_channels=input_dim, output_channels=2).cuda()
target_model.load_state_dict(torch.load(PATH))
target_model.eval()

lines = []

# Open the file and read lines
with open(data_name+"_vocab_0.txt", 'r') as file:
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    
# print(lines)

mask = []
sm = torch.nn.Softmax(dim=1)
motif_embedding = []
for i, smiles in enumerate(lines):
    mol = get_mol(smiles, False)
    mol = sanitize(mol, False)
    data = to_tudataset(mol, data_name)
    data.cuda()
    batch = torch.zeros(data.x.size(0), dtype=torch.int64).cuda()
    out = target_model(data.x, data.edge_index, batch)
    embedding = target_model(data.x, data.edge_index, batch, return_embedding=True)
    motif_embedding.append(embedding)
    pred = sm(out)
    if pred[0,0] > 0.9:
        mask.append(i)
full_mask = torch.tensor([i for _ in range(len(lines))], dtype=torch.int64)
motif_embedding = torch.stack(motif_embedding).squeeze()
# mask = [i for i in range(10)]
print(mask)
torch.save(motif_embedding, "checkpoints/"+data_name+"_motif_embedding_0.pt")
torch.save(mask, "checkpoints/"+data_name+"_mask_0.pt")
torch.save(full_mask, "checkpoints/"+data_name+"_full_mask_0.pt")
print(len(mask))
print(len(full_mask))

lines = []

# Open the file and read lines
with open(data_name+'_vocab_1.txt', 'r') as file:
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    
# print(lines)

mask = []
sm = torch.nn.Softmax(dim=1)
motif_embedding = []
for i, smiles in enumerate(lines):
    mol = get_mol(smiles, False)
    mol = sanitize(mol, False)
    data = to_tudataset(mol, data_name)
    data.cuda()
    batch = torch.zeros(data.x.size(0), dtype=torch.int64).cuda()
    out = target_model(data.x, data.edge_index, batch)
    embedding = target_model(data.x, data.edge_index, batch, return_embedding=True)
    motif_embedding.append(embedding)
    pred = sm(out)
    # print(pred)
    if pred[0,1] > 0.9:
        print(smiles)
        mask.append(i)
print(mask)
full_mask = torch.tensor([i for _ in range(len(lines))], dtype=torch.int64)
# mask = [i for i in range(30)] + [i for i in range(1339, 1364)]
motif_embedding = torch.stack(motif_embedding).squeeze()
torch.save(motif_embedding, "checkpoints/"+data_name+"_motif_embedding_1.pt")
torch.save(mask, "checkpoints/"+data_name+"_mask_1.pt")
torch.save(full_mask, "checkpoints/"+data_name+"_full_mask_1.pt")
print(len(mask))
print(len(full_mask))