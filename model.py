import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GATConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, input_channels, output_channels):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(input_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index=None, batch=None, return_embedding=False, classifier=False):
        if classifier:
            return self.lin(x)
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        if batch != None:
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        if return_embedding:
            return x
        x = self.lin(x)
        
        return x


class GCN2(torch.nn.Module):
    def __init__(self, hidden_channels, input_channels, output_channels):
        super(GCN2, self).__init__()
        self.conv1 = GraphConv(input_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)
        self.conv5 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index, batch=None, return_embedding=False):
        # 1. Obtain node embeddings 
        x = x + self.conv1(x, edge_index)
        x = x.relu()
        x = x + self.conv2(x, edge_index)
        x = x.relu()
        x = x + self.conv3(x, edge_index)
        x = x.relu()
        x = x + self.conv4(x, edge_index)
        x = x.relu()
        x = x + self.conv5(x, edge_index)

        # 2. Readout layer
        if batch != None:
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        if return_embedding:
            return x
        x = self.lin(x)
        
        return x
    


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, input_channels, output_channels, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_channels, hidden_channels, heads=heads, add_self_loops=False)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.batch_norms1 = nn.BatchNorm1d(hidden_channels)
        self.batch_norms2 = nn.BatchNorm1d(hidden_channels)
        self.batch_norms3 = nn.BatchNorm1d(hidden_channels)
        # self.lin = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index, return_weight=False):

        x, edge_tuple = self.conv1(x, edge_index, return_attention_weights=return_weight)
        # x = self.batch_norms1(x)
        # x = x.relu()
        # x = self.conv2(x, edge_index)
        # x = self.batch_norms2(x)
        # x = x.relu()
        # x = self.conv3(x, edge_index)
        # x = self.batch_norms3(x)
        # x = global_mean_pool(x, batch)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin(x)
        return x, edge_tuple
