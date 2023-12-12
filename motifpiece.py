from collections import defaultdict
from tqdm import tqdm
import math
import torch
from torch_geometric.data import Data
import rdkit.Chem as Chem
from utils import get_mol, sanitize, get_smiles, sanitize_smiles, get_fragment_mol, get_complement_fragment_mol
device = torch.device("cuda:0")
softmax = torch.nn.Softmax(dim=1).to(device)
sig = torch.nn.Sigmoid().to(device)
import numpy as np

class TreeNode:
    def __init__(self, value, smiles):
        self.value = value
        self.smiles = smiles
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.value) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

class MotifPiece:
    def __init__(self, smiles_list) -> None:
        self.smiles_list = smiles_list
        self.motif_explanation = defaultdict(int)
        self.motifs_mapping = defaultdict(self.create_int_defaultdict)
        self.process()

    def create_int_defaultdict(self):
        return defaultdict(int)
    
    def smiles_from_atomic_number(self, atomic_number):
        # Create a molecule with a single atom using its atomic number
        atom = Chem.Atom(atomic_number)
        mol = Chem.RWMol()
        mol.AddAtom(atom)
        # Convert the molecule to a SMILES string
        smiles = Chem.MolToSmiles(mol)
        return smiles

    def initialize_node(self, mol):
        """
        An initialization function for node merge motif vocabulary generation

        Parameters
        ---------
        mol: RDKit object
            A mol that will be process for the next process
        
        Output
        -----
        v_dict: A dictionary
            A dictionary with format subgraph_id: constituent origianl node indices of the subgraph
        e_dict: A dictionary
            A dictionary with format edge_id: (start_node_id, end_node_id)
        """
        s_dict = defaultdict(list)
        v_dict = defaultdict(list)
        e_dict = defaultdict(tuple)

        for bond in mol.GetBonds():
            id = bond.GetIdx()
            startid = bond.GetBeginAtomIdx()
            endid = bond.GetEndAtomIdx()
            e_dict[id] = (startid, endid)
            s_dict[startid].append(id)
            s_dict[endid].append(id)
        
        motif_list = []
        for atom in mol.GetAtoms():
            id = atom.GetIdx()
            motif_list.append(self.smiles_from_atomic_number(atom.GetAtomicNum()))
            v_dict[id] = [id]
            if id not in s_dict:
                s_dict[id] = []

        tree_nodes = []
        for i, motif in enumerate(motif_list):
            tree_nodes.append(TreeNode(i, motif))

        return s_dict, v_dict, e_dict, len(v_dict), len(e_dict), tree_nodes

    def process(self):
        iteration = 0
        s_dict_list = []
        v_dict_list = []
        e_dict_list = []

        mol_list = []
        max_node_list = []
        max_edge_list = []
        tree_list = []
        self.motif_list = [defaultdict(list) for x in self.smiles_list]
        while True:
            motif_count = defaultdict(int)
            motif_indices = {}
            for i, smiles in enumerate(tqdm(self.smiles_list)):
                if iteration == 0:
                    mol = get_mol(smiles, False)
                    mol = sanitize(mol, False)
                    if mol == None:
                        print("None mol")
                        continue
                    s_dict, v_dict, e_dict, max_node, max_edge, tree_nodes = self.initialize_node(mol)
                    s_dict_list.append(s_dict)
                    v_dict_list.append(v_dict)
                    e_dict_list.append(e_dict)
                    tree_list.append(tree_nodes)
                    max_node_list.append(max_node)
                    max_edge_list.append(max_edge)
                    mol_list.append(mol)
                else:
                    mol, s_dict, v_dict, e_dict, max_node, max_edge, tree_nodes = mol_list[i], s_dict_list[i], v_dict_list[i], e_dict_list[i], max_node_list[i], max_edge_list[i], tree_list[i]
                
                ### Check all possible motif candidate which is a pair of node
                for e_id, node_pair in e_dict.items():
                    node_1 = node_pair[0]
                    node_2 = node_pair[1]
                    ori_node_1 = v_dict[node_1]
                    ori_node_2 = v_dict[node_2]
                    m = list(set(ori_node_1+ori_node_2))
                    m_mol = get_fragment_mol(mol, m)
                    m_smiles = get_smiles(m_mol)
                    m_smiles = sanitize_smiles(m_smiles)
                    motif_count[m_smiles] += 1
                    # a_i = 0
                    if m_smiles not in motif_indices:
                        motif_indices[m_smiles] = defaultdict(list)
                        motif_indices[m_smiles][i] = [e_id]
                    else:
                        motif_indices[m_smiles][i].append(e_id)

            ### Select the best motif candidate
            selected_motif = None
            max_score = -1

            # print(motif_count)
            # print(stop)
            for motif, count in motif_count.items():
                if count < 2:
                    continue
                if mol == None:
                    print("ERROR!!!")
                    print(stop)

                score = count
                # score = math.sqrt(freq*affect)
                if score > max_score:
                    max_score = score
                    selected_motif = motif
            
            print(f"max score: {max_score}")
            print(f"Selected Motif: {selected_motif}")

            if selected_motif == None:
                break
            merge_indices = motif_indices[selected_motif]
            self.motif_explanation[selected_motif] += len(merge_indices)
            
            
            ### Merge selected motif in the graph
            for id, edge_list in merge_indices.items():
                count_merged_motif = 0
                # print(a_i_list)
                # print(sorted_a_i_list)
                # print(edge_list)
                # print(sorted_edge_list)
                # print(stop)
                
                merged_set = set()
                s_dict, v_dict, e_dict, max_node, max_edge, tree_nodes = s_dict_list[id], v_dict_list[id], e_dict_list[id], max_node_list[id], max_edge_list[id], tree_list[id]
                node_list_edge = []
                
                for e in edge_list:
                    node_pair = e_dict[e]
                    node_1 = node_pair[0]
                    node_2 = node_pair[1]
                    node_list_edge.append((node_1, node_2))
                    if node_1 not in v_dict or node_2 not in v_dict:
                        print("Huge Error!")
                        print(stop)
                    if node_1 not in s_dict or node_2 not in s_dict:
                        print("Huge Error!")
                        print(stop)
                for i, e in enumerate(edge_list):
                    node_pair = node_list_edge[i]
                    node_1 = node_pair[0]
                    node_2 = node_pair[1]
                    if node_1 in merged_set or node_2 in merged_set:
                        continue
                    count_merged_motif += 1
                    merged_set.add(node_1)
                    merged_set.add(node_2)
                    merged_node_list = list(set(v_dict[node_1]+v_dict[node_2]))

                    v_dict[max_node] = merged_node_list
                    self.motif_list[id][selected_motif].append(merged_node_list)

                    # Create a new tree node for max node
                    tree_node_1 = tree_nodes[node_1]
                    tree_node_2 = tree_nodes[node_2]
                    new_tree_node = TreeNode(max_node, selected_motif)
                    new_tree_node.add_child(tree_node_1)
                    new_tree_node.add_child(tree_node_2)
                    tree_nodes.append(new_tree_node)

                    edge_node_1 = s_dict[node_1]
                    edge_node_2 = s_dict[node_2]
                    need_to_del_edge = []
                    need_to_change_node = []
                    for other_e in list(set(edge_node_1+edge_node_2)):
                        first_node = e_dict[other_e][0]
                        second_node = e_dict[other_e][1]
                        need_to_del_edge.append(other_e)
                        if first_node in [node_1, node_2] and second_node not in [node_1, node_2]:
                            need_to_change_node.append(second_node)
                        elif first_node not in [node_1, node_2] and second_node in [node_1, node_2]:
                            need_to_change_node.append(first_node)
                        elif first_node in [node_1, node_2] and second_node in [node_1, node_2]:
                            pass
                        else:
                            print("Wrong merge!!!")
                            print(stop)
                    
                    count_added_edge = 0
                    for node in need_to_change_node:
                        s_dict[node] = [x for x in s_dict[node] if x not in need_to_del_edge]
                        e_dict[max_edge+count_added_edge] = (node, max_node)
                        s_dict[max_node].append(max_edge+count_added_edge)
                        s_dict[node].append(max_edge+count_added_edge)
                        count_added_edge += 1

                    for item in need_to_del_edge:
                        del e_dict[item]
                    
                    del v_dict[node_1], v_dict[node_2], s_dict[node_1], s_dict[node_2]
                    max_node += 1
                    max_edge += count_added_edge
                self.motifs_mapping[id][selected_motif] += count_merged_motif
                s_dict_list[id], v_dict_list[id], e_dict_list[id], max_node_list[id], max_edge_list[id], tree_list[id] = s_dict, v_dict, e_dict, max_node, max_edge, tree_nodes

            iteration += 1
        print(f"The final motif explanation: {self.motif_explanation}")
        self.s_dict_list = s_dict_list
        self.v_dict_list = v_dict_list
        self.e_dict_list = e_dict_list
        self.tree_root_list = []
        for i, smiles in enumerate(tqdm(self.smiles_list)):
            v_dict, tree_nodes, max_node = v_dict_list[i], tree_list[i], max_node_list[i]
            if len(v_dict) == 1:
                self.tree_root_list.append(tree_nodes[-1])
            new_tree_node = TreeNode(max_node, smiles)
            for key, _ in v_dict.items():
                new_tree_node.add_child(tree_nodes[key])
            self.tree_root_list.append(new_tree_node)

    def inference(self):
        all_motif_smiles = []
        all_edge_list = []
        print("hhh")
        for i in tqdm(range(len(self.smiles_list))):
            s_dict, v_dict, e_dict, smiles = self.s_dict_list[i], self.v_dict_list[i], self.e_dict_list[i], self.smiles_list[i]
            mol = get_mol(smiles, False)
            mol = sanitize(mol, False)
            motif_smiles_list = []
            s_id_list = []

            for subgraph_id, atom_list in v_dict.items():
                if len(atom_list) == 1:
                    m_mol = get_fragment_mol(mol, atom_list)
                    m_smiles = get_smiles(m_mol)
                    m_smiles = sanitize_smiles(m_smiles)
                    motif_smiles_list.append(m_smiles)
                    s_id_list.append(subgraph_id)
            
            edge_list = []

            # Miss one situation that the edge motif is original edge, and no edges for this motif
            for edge_id, edge in e_dict.items():
                if edge[0] in s_id_list and edge[1] in s_id_list:
                    edge_list.append((s_id_list.index(edge[0]), s_id_list.index(edge[1])))
            all_motif_smiles.append(motif_smiles_list)
            all_edge_list.append(edge_list)
        # print(all_motif_smiles)
        # print(all_edge_list)
        return all_motif_smiles, all_edge_list