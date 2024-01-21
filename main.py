import torch
import random
from torch_geometric.datasets import TUDataset
from model import GCN, GCN2, GAT
from tu2smiles import to_smiles, to_tudataset
from utils import sanitize_smiles, get_mol, sanitize
from heterdataset import HeterDataset
from split import split_list

from tqdm import tqdm

device = torch.device("cuda:0")

data_name = "Mutagenicity"

dataset = TUDataset('data', name=data_name)
print(dataset[0])
# print(stop)

torch.manual_seed(12345)

# # Clean the dataset
# indices = []
# cleaned_dataset = []
# for i, data in enumerate(dataset):
#     smiles = to_smiles(data, True, data_name)
#     smiles = sanitize_smiles(smiles)
#     if not smiles == None:
#         indices.append(i)
#         mol = sanitize(get_mol(smiles, False), False)
#         data = to_tudataset(mol, data_name, data.y)
#         cleaned_dataset.append(data)
        
# # dataset = dataset[indices]
# random.shuffle(cleaned_dataset)
# print(len(cleaned_dataset))
# # dataset = dataset.shuffle()
# train_dataset = dataset[:2841]
# test_dataset = dataset[2841:]

# torch.save(train_dataset, "checkpoints/train_dataset_mutagenicity.pt")
# torch.save(test_dataset, "checkpoints/test_dataset_mutagenicity.pt")

# print(f'Number of training graphs: {len(train_dataset)}')
# print(f'Number of test graphs: {len(test_dataset)}')

# from torch_geometric.loader import DataLoader

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# model = GCN(hidden_channels=64, input_channels=dataset.num_features, output_channels=dataset.num_classes)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.CrossEntropyLoss()

# def train():
#     model.train()

#     for data in train_loader:  # Iterate in batches over the training dataset.
#          out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
#          loss = criterion(out, data.y)  # Compute the loss.
#          loss.backward()  # Derive gradients.
#          optimizer.step()  # Update parameters based on gradients.
#          optimizer.zero_grad()  # Clear gradients.

# def test(loader):
#      model.eval()

#      correct = 0
#      for data in loader:  # Iterate in batches over the training/test dataset.
#          out = model(data.x, data.edge_index, data.batch)  
#          pred = out.argmax(dim=1)  # Use the class with highest probability.
#          correct += int((pred == data.y).sum())  # Check against ground-truth labels.
#      return correct / len(loader.dataset)  # Derive ratio of correct predictions.

# best_test_acc = 0.0

PATH = 'checkpoints/best_model_mutagenicity.pt'

# for epoch in range(1, 171):
#     train()
#     train_acc = test(train_loader)
#     test_acc = test(test_loader)
#     if test_acc > best_test_acc:
#         best_test_acc = test_acc
#         torch.save(model.state_dict(), PATH)
#     print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


target_model = GCN(hidden_channels=64, input_channels=dataset.num_features, output_channels=dataset.num_classes)
target_model.load_state_dict(torch.load(PATH))
target_model.eval()

train_dataset = torch.load("checkpoints/train_dataset_mutagenicity.pt")
test_dataset = torch.load("checkpoints/test_dataset_mutagenicity.pt")

train_smiles_list = []
train_list = []
test_smiles_list = []

for i, data in enumerate(train_dataset):
    smiles = to_smiles(data, False, data_name)
    smiles = sanitize_smiles(smiles)
    if not smiles == None:
        batch = torch.zeros(data.x.size(0), dtype=torch.int64)
        out = target_model(data.x, data.edge_index, batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        if pred == data.y:
            train_smiles_list.append(smiles)
            train_list.append(data)

for i, data in enumerate(test_dataset):
    smiles = to_smiles(data, True, data_name)
    smiles = sanitize_smiles(smiles)
    if not smiles == None:
        test_smiles_list.append(smiles)


print(len(train_smiles_list))
# print(stop)
heter_dataset = HeterDataset("heter_data/Mutagenicity", train_list, data_name, train_smiles_list, target_model)
heter_data = heter_dataset[0].to(device)
print(heter_data.x.size())
# label_list = heter_data.label_list
#
positive_list = heter_data.label_0.tolist()
negative_list = heter_data.label_1.tolist()
length_pos = len(positive_list)
length_neg = len(negative_list)
intersect = set(positive_list).intersection(set(negative_list))
print(positive_list)
print(len(positive_list))
print(len(negative_list))

if length_pos > length_neg:
    pos = random.sample(positive_list, length_neg)
    neg = negative_list
else:
    neg = random.sample(negative_list, length_pos)
    pos = positive_list


pos_1, pos_2 = split_list(pos)
neg_1, neg_2 = split_list(neg)


pos_1 = torch.tensor(pos_1)
pos_2 = torch.tensor(pos_2)
neg_1 = torch.tensor(neg_1)
neg_2 = torch.tensor(neg_2)

# start_1 = torch.zeros_like(pos_1)
# start_2 = torch.ones_like(neg_1)
# start_3 = torch.ones_like(pos_2)
# start_4 = torch.zeros_like(neg_2)

# start_5 = torch.zeros_like(pos_2)
# start_6 = torch.ones_like(neg_2)
# start_7 = torch.ones_like(pos_1)
# start_8 = torch.zeros_like(neg_1)

# training_indices = torch.cat((pos_1, neg_1, pos_2, neg_2), dim=0).to(device)
# testing_indices = torch.cat((pos_2, neg_2, pos_1, neg_1), dim=0).to(device)
# training_start_node_indices = torch.cat((start_1, start_2, start_3, start_4), dim=0).to(device)
# testing_start_node_indices = torch.cat((start_5, start_6, start_7, start_8), dim=0).to(device)

# training_label = torch.cat((torch.zeros(pos_1.size(0)+neg_1.size(0), dtype=torch.int64), torch.ones(pos_2.size(0)+neg_2.size(0), dtype=torch.int64)), dim=0).to(device)
# testing_label = torch.cat((torch.zeros(pos_2.size(0)+neg_2.size(0), dtype=torch.int64), torch.ones(pos_1.size(0)+neg_1.size(0), dtype=torch.int64)), dim=0).to(device)

training_indices = torch.cat((pos_1, neg_1), dim=0).to(device)
testing_indices = torch.cat((pos_2, neg_2), dim=0).to(device)
training_label = torch.cat((torch.zeros(pos_1.size(0), dtype=torch.int64), torch.ones(neg_1.size(0), dtype=torch.int64)), dim=0).to(device)
testing_label = torch.cat((torch.zeros(pos_2.size(0), dtype=torch.int64), torch.ones(neg_2.size(0), dtype=torch.int64)), dim=0).to(device)

# model = GCN(hidden_channels=64, input_channels=64, output_channels=dataset.num_classes).to(device)
model = GAT(hidden_channels=64, input_channels=64, output_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
linear = torch.nn.Linear(64, 2).to(device)
optimizer_linear = torch.optim.Adam(linear.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

def train_heter():
    model.train()
    linear.train()
    feature_embedding, _ = model(heter_data.x, heter_data.edge_index)

    input = feature_embedding[training_indices]
    logit = linear(input)
    loss = criterion(logit, training_label)

    optimizer.zero_grad()
    optimizer_linear.zero_grad()

    loss.backward()

    optimizer.step()
    optimizer_linear.step()


def test_heter(indices, label):
    model.eval()
    linear.eval()

    with torch.no_grad():
        feature_embedding, edge_tuple = model(heter_data.x, heter_data.edge_index)

        input = feature_embedding[indices]
        logit = linear(input)
        pred = logit.argmax(dim=1)
        correct = int((pred == label).sum())
        accuracy = correct / indices.size(0)
    
    return accuracy, edge_tuple, logit

best_test_acc = 0

for epoch in range(1, 5000):
    train_heter()
    train_acc, _, train_logit= test_heter(training_indices, training_label)
    test_acc, edge_tuple, test_logit = test_heter(testing_indices, testing_label)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_edge_tuple = edge_tuple
        best_train_logit = train_logit
        best_test_logit = test_logit

    print(f"Epoch {epoch}: train acc: {train_acc}, test acc: {test_acc}.")

print(f"Best test accuracy: {best_test_acc}")
edge_index, attention_weights = best_edge_tuple

print(attention_weights.size())
edge_list = edge_index.t()
print(edge_list.size())
print(f"number of data: {len(train_list)}")
num_motifs = heter_data.x.size(0) - len(train_list)
print(f"number of motifs: {num_motifs}")
M_mot_mol = torch.zeros(num_motifs, len(train_list))
M_mol_label = torch.zeros(len(train_list), dataset.num_classes)
M_mot_dom = torch.ones((num_motifs, 2), dtype=torch.float)

for i, edge in enumerate(tqdm(edge_list)):
    if edge[0] > edge[1]:
        continue
    motif_id = edge[0].item()
    M_mot_mol[motif_id][edge[1]-num_motifs] = attention_weights[i]
    M_mot_dom[motif_id][0] += 1
    M_mot_dom[motif_id][1] += 1
    

m = torch.nn.Softmax(dim=1)
sig = torch.nn.Sigmoid()
train_pred = m(best_train_logit)
test_pred = m(best_test_logit)
print(pred.size())

M_mol_label[training_indices.cpu()-num_motifs] = train_pred.cpu()
M_mol_label[testing_indices.cpu()-num_motifs] = test_pred.cpu()

print(M_mot_mol)
print(M_mol_label)

ini_score = torch.matmul(M_mot_mol, M_mol_label)
ini_score = torch.div(ini_score, M_mot_dom)

# print(ini_score.size())
# print(stop)
norm_score = m(ini_score)
sig_score = sig(ini_score)
# print(sig_score.size())
# print(ini_score.size())

# score = norm_score * sig_score
score = sig_score

motif_list = list(heter_data.motif_vocab.keys())

label_0 = sorted(zip(score[:, 0], motif_list), key=lambda x: x[0], reverse=True)
label_0_score, sorted_label_0 = zip(*label_0)
label_1 = sorted(zip(score[:, 1], motif_list), key=lambda x: x[0], reverse=True)
label_1_score, sorted_label_1 = zip(*label_1)

# print(label_0_score[:20])
# print(label_1_score[:20])

# print(sorted_label_0[:20])
# print(sorted_label_1[:20])

final_motif_0 = []
final_motif_1 = []

for i, score in enumerate(label_0_score):
    if score >= 0.0:
        final_motif_0.append(sorted_label_0[i])
for i, score in enumerate(label_1_score):
    if score >= 0.0:
        final_motif_1.append(sorted_label_1[i])

print(len(label_0_score))
print(len(final_motif_0))
print(len(label_1_score))
print(len(final_motif_1))

torch.save(final_motif_0, "checkpoints/motif_0.pt")
torch.save(final_motif_1, "checkpoints/motif_1.pt")