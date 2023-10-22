import torch
from torch_geometric.datasets import TUDataset
from momox import MomoX
from model import GCN
from tu2smiles import to_smiles
from utils import sanitize_smiles

device = torch.device("cuda:0")

data_name = "Mutagenicity"

dataset = TUDataset('data', name=data_name)
print(dataset[0])

torch.manual_seed(12345)

# Clean the dataset
indices = []
for i, data in enumerate(dataset):
    smiles = to_smiles(data, True, data_name)
    smiles = sanitize_smiles(smiles)
    if not smiles == None:
        indices.append(i)
dataset = dataset[indices]

dataset = dataset.shuffle()
train_dataset = dataset[:2450]
test_dataset = dataset[2450:]

torch.save(train_dataset, "checkpoints/train_dataset_mutagenicity.pt")
torch.save(test_dataset, "checkpoints/test_dataset_mutagenicity.pt")

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = GCN(hidden_channels=64, input_channels=dataset.num_features, output_channels=dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

best_test_acc = 0.0

PATH = 'checkpoints/best_model_mutagenicity.pt'

for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), PATH)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


target_model = GCN(hidden_channels=64, input_channels=dataset.num_features, output_channels=dataset.num_classes)
target_model.load_state_dict(torch.load(PATH))
target_model.eval()

train_dataset = torch.load("checkpoints/train_dataset_mutagenicity.pt")
test_dataset = torch.load("checkpoints/test_dataset_mutagenicity.pt")

train_smiles_list = []
train_list = []
test_smiles_list = []

for i, data in enumerate(train_dataset):
    smiles = to_smiles(data, True, data_name)
    smiles = sanitize_smiles(smiles)
    if not smiles == None:
        train_smiles_list.append(smiles)
        train_list.append(data)

for i, data in enumerate(test_dataset):
    smiles = to_smiles(data, True, data_name)
    smiles = sanitize_smiles(smiles)
    if not smiles == None:
        test_smiles_list.append(smiles)

print(len(train_smiles_list))
motif_explainer = MomoX(train_smiles_list, target_model)

torch.save(motif_explainer.motif_explanation, "checkpoints/mutagenicity_1_cubic.pt")