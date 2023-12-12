import torch
import torch.nn as nn

import math, random, sys
import argparse
from vocab import Vocab
from jtnn_vae import JTNNVAE
import rdkit
from tqdm import tqdm
from model import GCN

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--nsample', type=int, required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--mask', required=True)
parser.add_argument('--graph_label', required=True)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

PATH = 'checkpoints/best_model_mutagenicity.pt'
target_model = GCN(hidden_channels=64, input_channels=14, output_channels=2).to(device)
target_model.load_state_dict(torch.load(PATH))
target_model.eval()

# mask = torch.tensor([i for i in range(10)])
mask = torch.load(args.mask)
# mask = torch.tensor(mask).cuda()

model = JTNNVAE(vocab, mask, args.hidden_size, args.latent_size, args.depthT, args.depthG, target_model, int(args.graph_label), device).to(device)
model.load_state_dict(torch.load(args.model))

torch.manual_seed(0)
smiles_set = set([])
for i in tqdm(range(args.nsample)):
    print(model.sample_prior()) 

# while len(smiles_set) < args.nsample:
#     smiles = model.sample_prior()
#     if smiles not in smiles_set:
#         print(smiles)
    