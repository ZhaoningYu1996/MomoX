import torch
import torch.nn as nn
import torch.nn.functional as F

from mol_tree import MolTree
from vocab import Vocab
from nnutils import create_var, flatten_tensor, avg_pool
from jtnn_enc import JTNNEncoder
from jtnn_dec import JTNNDecoder
from mpn import MPN
from jtmpn import JTMPN
from datautils import tensorize

from utils import enum_assemble, set_atommap, copy_edit_mol, attach_mols
import rdkit
import rdkit.Chem as Chem
import copy, math
loss_fn = torch.nn.CrossEntropyLoss()

class JTNNVAE(nn.Module):

    def __init__(self, vocab, mask, hidden_size, latent_size, depthT, depthG, target_model, graph_label, motif_embedding, device):
        super(JTNNVAE, self).__init__()
        self.vocab = vocab
        self.mask = mask
        self.device = device
        self.hidden_size = hidden_size
        self.latent_size = latent_size = int(latent_size / 2) #Tree and Mol has two vectors

        # self.jtnn = JTNNEncoder(hidden_size, depthT, nn.Embedding(vocab.size(), hidden_size), self.device)
        self.jtnn = JTNNEncoder(hidden_size, depthT, motif_embedding, self.device)
        # print("-------->")
        # print(latent_size)
        # print(hidden_size)
        # print(vocab.size())
        # self.decoder = JTNNDecoder(vocab, hidden_size, latent_size, nn.Embedding(vocab.size(), hidden_size), self.mask, self.device)
        self.decoder = JTNNDecoder(vocab, hidden_size, latent_size, motif_embedding, self.mask, self.device)

        self.jtmpn = JTMPN(hidden_size, depthG, self.device)
        self.mpn = MPN(hidden_size, depthG, self.device)
        self.target_model = target_model
        self.graph_label = graph_label
        self.softmax = nn.Softmax(dim=0).to(device)
        self.softmax_decode = nn.Softmax(dim=1).to(device)

        self.A_assm = nn.Linear(latent_size, hidden_size, bias=False)
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)
        self.pred_loss = torch.nn.CrossEntropyLoss()

        self.T_mean = nn.Linear(hidden_size, latent_size)
        self.T_var = nn.Linear(hidden_size, latent_size)
        self.G_mean = nn.Linear(hidden_size, latent_size)
        self.G_var = nn.Linear(hidden_size, latent_size)

    def encode(self, jtenc_holder, mpn_holder):
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        # mol_vecs = self.mpn(*mpn_holder)
        fatoms, fbonds, agraph, bgraph, scope, all_bonds = mpn_holder
        batch_vector = torch.cat([torch.full((t[1],), i) for i, t in enumerate(scope)]).to(self.device)
        edge_index = torch.tensor(all_bonds, dtype=torch.long)
        edge_index = edge_index.t().contiguous().to(self.device)
        mol_vecs = self.target_model(fatoms.to(self.device), edge_index, batch_vector, return_embedding = True)
        mol_pred = self.target_model(fatoms.to(self.device), edge_index, batch_vector, return_embedding = False)
        y = mol_pred.argmax(dim=1)
        # print(y)
        # print(stop)
        # for i in y:
        #     if i == 0:
        #         print(y)
        #         print("ERROR!")
        #         print(stop)
        return tree_vecs, tree_mess, mol_vecs, y
    
    def encode_from_smiles(self, smiles_list):
        tree_batch = [MolTree(s) for s in smiles_list]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)
        return torch.cat([tree_vecs, mol_vecs], dim=-1)

    # def encode_latent(self, jtenc_holder, mpn_holder):
    #     tree_vecs, _ = self.jtnn(*jtenc_holder)
    #     # mol_vecs = self.mpn(*mpn_holder)
    #     fatoms, fbonds, agraph, bgraph, scope, all_bonds = mpn_holder
    #     batch_vector = torch.cat([torch.full((t[1],), i) for i, t in enumerate(scope)]).to(self.device)
    #     edge_index = torch.tensor(all_bonds, dtype=torch.long)
    #     edge_index = edge_index.t().contiguous().to(self.device)
    #     mol_vecs = self.target_model(fatoms.to(self.device), edge_index, batch_vector, return_embedding = True)
    #     tree_mean = self.T_mean(tree_vecs)
    #     mol_mean = self.G_mean(mol_vecs)
    #     tree_var = -torch.abs(self.T_var(tree_vecs))
    #     mol_var = -torch.abs(self.G_var(mol_vecs))
    #     return torch.cat([tree_mean, mol_mean], dim=1), torch.cat([tree_var, mol_var], dim=1)

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = create_var(torch.randn_like(z_mean), device=self.device)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def sample_prior(self, prob_decode=False):
        z_tree = torch.randn(1, self.latent_size).to(self.device)
        z_mol = torch.randn(1, self.latent_size).to(self.device)
        smiles = self.decode(z_tree, z_mol, prob_decode)
        if smiles == None:
            return self.sample_prior()
        else:
            return smiles

    def forward(self, x_batch, beta):
        
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch
        x_tree_vecs, x_tree_mess, x_mol_vecs, y = self.encode(x_jtenc_holder, x_mpn_holder)
        tree_pred = self.target_model(x_tree_vecs, classifier=True)
        tree_loss = loss_fn(tree_pred, y)
        tree_acc = sum(tree_pred.argmax(dim=1)==y) / tree_pred.size(0)
        # print(x_tree_vecs.size())
        # print(stop)
        
        z_tree_vecs,tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs,mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)
        
        kl_div = tree_kl + mol_kl
        
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_batch, z_tree_vecs)
        
        assm_loss, assm_acc, pred_acc = self.assm(x_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess)
        # print(f"pred acc: {pred_acc}")

        return tree_loss + word_loss + topo_loss + assm_loss + beta * kl_div, kl_div.item(), word_acc, topo_acc, assm_acc, pred_acc, tree_acc.cpu()
        # return assm_loss + beta * kl_div, kl_div.item(), word_acc, topo_acc, assm_acc, pred_acc

    def assm(self, mol_batch, jtmpn_holder, x_mol_vecs, x_tree_mess):
        # print(f"x_mol_vecs: {x_mol_vecs.size()}")
        jtmpn_holder,batch_idx = jtmpn_holder
        # print(f"jtmpn_holder: {jtmpn_holder}")
        fatoms,fbonds,agraph,bgraph,scope,all_bonds = jtmpn_holder

        batch_idx = create_var(batch_idx, device=self.device)
        # print(batch_idx)
        
        # node_indices = [t[1] for t in scope]
        batch_vector = torch.cat([torch.full((t[1],), i) for i, t in enumerate(scope)]).to(self.device)
        # cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, x_tree_mess)
        edge_index = torch.tensor(all_bonds, dtype=torch.long)
        edge_index = edge_index.t().contiguous().to(self.device)
        cand_vecs = self.target_model(fatoms.to(self.device), edge_index, batch_vector, return_embedding = True)
        cand_pred = self.target_model(fatoms.to(self.device), edge_index, batch_vector, return_embedding = False)
        x_mol_vecs = x_mol_vecs.index_select(0, batch_idx)
        x_mol_vecs = self.A_assm(x_mol_vecs) #bilinear
        
        scores = torch.bmm(
                x_mol_vecs.unsqueeze(1),
                cand_vecs.unsqueeze(-1)
        ).squeeze()
        # print(scores.size())
        
        cnt,tot,acc,pred_acc = 0,0,0,0
        all_loss = []
        
        for i,mol_tree in enumerate(mol_batch):
            # print(i)
            comp_nodes = [node for node in mol_tree.nodes if len(node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            # print(comp_nodes)
            
            for node in comp_nodes:
                # print(f"node.cands: {node.cands}")
                label = node.cands.index(node.label)
                # print(f"label: {label}")
                ncand = len(node.cands)
                # print(f"ncand: {ncand}")
                
                cur_score = scores.narrow(0, tot, ncand)
                # print(f"cur_score: {cur_score}")
                cur_pred = cand_pred.narrow(0, tot, ncand)
                # print(cur_pred.size())
                
                tot += ncand

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1
                # print("---->")
                # print(cur_pred[label].size())
                pred_acc += self.softmax(cur_pred[label])[self.graph_label].item()
                

                # print(f"pred loss: {self.assm_loss(cur_pred[label].unsqueeze(0), torch.tensor([self.graph_label]).to(self.device))}")
                # print(f"pred: {self.softmax(cur_pred[label])[self.graph_label]}")
                

                label = create_var(torch.LongTensor([label]), device=self.device)
                # print("---->")
                # print(f"assm loss: {self.assm_loss(cur_score.view(1,-1), label)}")
                all_loss.append(self.assm_loss(cur_score.view(1,-1), label))
                # print(cur_pred[label].size())
                
                all_loss.append(self.assm_loss(cur_pred[label], torch.tensor([self.graph_label]).to(self.device)))
        # print(tot)
        # print(stop)
        
        all_loss = sum(all_loss) / (2*len(mol_batch))
        return all_loss, acc * 1.0 / cnt, pred_acc * 1.0 / cnt

    def decode(self, x_tree_vecs, x_mol_vecs, prob_decode):
        #currently do not support batch decoding
        assert x_tree_vecs.size(0) == 1 and x_mol_vecs.size(0) == 1

        pred_root,pred_nodes = self.decoder.decode(x_tree_vecs, prob_decode, self.mask)
        if len(pred_nodes) == 0: return None
        elif len(pred_nodes) == 1: return pred_root.smiles
        # elif len(pred_nodes) == 1: return None

        #Mark nid & is_leaf & atommap
        for i,node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder,mess_dict = JTNNEncoder.tensorize_nodes(pred_nodes, scope)
        _,tree_mess = self.jtnn(*jtenc_holder)
        tree_mess = (tree_mess, mess_dict) #Important: tree_mess is a matrix, mess_dict is a python dict

        x_mol_vecs = self.A_assm(x_mol_vecs).squeeze() #bilinear

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol,_ = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, check_aroma=True)
        if cur_mol is None: 
            cur_mol = copy_edit_mol(pred_root.mol)
            global_amap = [{}] + [{} for node in pred_nodes]
            global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}
            cur_mol,pre_mol = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, check_aroma=False)
            if cur_mol is None: cur_mol = pre_mol

        if cur_mol is None: 
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None
        
    def dfs_assemble(self, y_tree_mess, x_mol_vecs, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node, prob_decode, check_aroma):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid,a2,a1) for nid,a1,a2 in fa_amap if nid == cur_node.nid]
        cands,aroma_score = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            return None, cur_mol

        cand_smiles,cand_amap = zip(*cands)
        aroma_score = torch.Tensor(aroma_score).to(self.device)
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        if len(cands) > 1:
            jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            fatoms,fbonds,agraph,bgraph,scope,all_bonds = jtmpn_holder
            # cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])
            batch_vector = torch.cat([torch.full((t[1],), i) for i, t in enumerate(scope)]).to(self.device)
            edge_index = torch.tensor(all_bonds, dtype=torch.long)
            edge_index = edge_index.t().contiguous().to(self.device)
            # cand_vecs = self.target_model(fatoms.to(self.device), edge_index, batch_vector, return_embedding = True)
            # scores = torch.mv(cand_vecs, x_mol_vecs) + aroma_score
            cand_pred = self.target_model(fatoms.to(self.device), edge_index, batch_vector, return_embedding = False)
            scores = self.softmax_decode(cand_pred)[:, self.graph_label]
            # print(batch_vector)
            
            # print(stop)
        else:
            scores = torch.Tensor([1.0])
            # # print(cands)
            # jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            # fatoms,fbonds,agraph,bgraph,scope,all_bonds = jtmpn_holder
            # # cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])
            # batch_vector = torch.cat([torch.full((t[1],), i) for i, t in enumerate(scope)]).to(self.device)
            # edge_index = torch.tensor(all_bonds, dtype=torch.long)
            # edge_index = edge_index.t().contiguous().to(self.device)
            # # cand_vecs = self.target_model(fatoms.to(self.device), edge_index, batch_vector, return_embedding = True)
            # # scores = torch.mv(cand_vecs, x_mol_vecs) + aroma_score
            # cand_pred = self.target_model(fatoms.to(self.device), edge_index, batch_vector, return_embedding = False)
            # scores = self.softmax_decode(cand_pred)[:, self.graph_label]
            

        if prob_decode:
            probs = F.softmax(scores.view(1,-1), dim=1).squeeze() + 1e-7 #prevent prob = 0
            cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _,cand_idx = torch.sort(scores, descending=True)
            
        # if scores[0] < 0.9:
        #     return None, None

        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol
        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id,ctr_atom,nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap) #father is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None: continue
            
            has_error = False
            for nei_node in children:
                if nei_node.is_leaf: continue
                tmp_mol, tmp_mol2 = self.dfs_assemble(y_tree_mess, x_mol_vecs, all_nodes, cur_mol, new_global_amap, pred_amap, nei_node, cur_node, prob_decode, check_aroma)
                if tmp_mol is None: 
                    has_error = True
                    if i == 0: pre_mol = tmp_mol2
                    break
                cur_mol = tmp_mol

            if not has_error: return cur_mol, cur_mol

        return None, pre_mol
