import torch
from motifpiece import MotifPiece
from torch_geometric.data import InMemoryDataset, Data
from tu2smiles import to_tudataset
from utils import get_mol, sanitize
from tqdm import tqdm

class HeterDataset(InMemoryDataset):
    def __init__(self, root, dataset, data_name, data_smiles, model, transform=None, pre_transform=None, pre_filter=None) -> None:
        self.dataset = dataset
        self.dataname = data_name
        self.data_smiles = data_smiles
        self.model = model
        if len(self.dataset) != len(self.data_smiles):
            print("Error!")
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        
        
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        motif_piece = MotifPiece(self.data_smiles, self.dataname)
        # print(motif_piece.motif_list)
        motif_list = motif_piece.motif_list
        motif_mapping = motif_piece.motif_mapping
        motif_explanation = motif_piece.motif_explanation
        motif_vocab = motif_piece.motif_vocab
        print(len(motif_explanation))
        x = []
        edge_index = []
        num_motif = len(motif_vocab)
        for motif in motif_vocab.keys():
            mol = get_mol(motif, False)
            mol = sanitize(mol, False)
            data = to_tudataset(mol, self.dataname)
            batch = torch.zeros(data.x.size(0), dtype=torch.int64)
            embedding = self.model(data.x, data.edge_index, batch, return_embedding=True)
            x.append(embedding)
            
        label_0 = []
        label_1 = []
            
        for i, data in enumerate(tqdm(self.data_smiles)):
            motifs = motif_list[i].keys()
            mol = get_mol(data, False)
            mol = sanitize(mol, False)
            data = to_tudataset(mol, self.dataname)
            batch = torch.zeros(data.x.size(0), dtype=torch.int64)
            embedding = self.model(data.x, data.edge_index, batch, return_embedding=True)
            logit = self.model(embedding, classifier=True)

            if logit[0].argmax() == 0:
                label_0.append(i+num_motif)
            elif logit[0].argmax() == 1:
                label_1.append(i+num_motif)
                
            x.append(embedding)
            for motif in motifs:
                id = motif_vocab[motif]
                edge_index.append((id, i+num_motif))
                # edge_index.append((i+num_motif, id))
        print(len(x))
        x = torch.stack(x)
        x = x.squeeze(dim=1)
        edge_index = torch.tensor(edge_index).t()
        print(len(label_0), len(label_1))
        label_0 = torch.tensor(label_0)
        label_1 = torch.tensor(label_1)
        heter_data = Data(x, edge_index, label_0=label_0, label_1=label_1, motif_vocab=motif_vocab)
        data_list.append(heter_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        