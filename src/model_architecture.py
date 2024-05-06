import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class MeshPairClassifier(torch.nn.Module):
    pass

class GraphPairClassifier(torch.nn.Module):
    def __init__(self, num_input_features=3):
        torch.manual_seed(12345)
        super(GraphPairClassifier, self).__init__()
        
        self.encoder = GCNEncoder(num_input_features, 512)
        self.classifier = MLPBinaryClassifier(1024)

    def forward(self, data):

        # Encoding
        x_1 = self.encoder(data.x_1, data.edge_index_1, data.x_1_batch)
        x_2 = self.encoder(data.x_2, data.edge_index_2, data.x_2_batch)
        
        # Create feature embedding of pair graph
        x = torch.cat((x_1, x_2), dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return torch.sigmoid(x)

class TabularClassifier(torch.nn.Module):
    def __init__(self, num_input_features = 24):
        torch.manual_seed(12345)
        super(TabularClassifier, self).__init__()
        
        self.classifier = MLPBinaryClassifier(num_input_features)

    def forward(self, data):
        x = self.classifier(data)
        return torch.sigmoid(x)
    
####################
## Helper Classes ##
####################
        
class GCNEncoder(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(GCNEncoder, self).__init__()

        self.conv1 = GCNConv(num_input_features, 2**9)
        self.conv2 = GCNConv(2**9, 2**9)
        self.conv3 = GCNConv(2**9, 2**9)
        self.conv4 = GCNConv(2**9, 2**9)
        self.conv5 = GCNConv(2**9, num_output_features)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))

        x = global_mean_pool(x, batch)  # Get whole graph embeddings
        return x
 
class MLPBinaryClassifier(torch.nn.Module):
    def __init__(self, num_input_features):
        super(MLPBinaryClassifier, self).__init__()
        self.layer_1 = torch.nn.Linear(num_input_features, 2**16)
        self.layer_2 = torch.nn.Linear(2**16, 2**8)
        self.layer_3 = torch.nn.Linear(2**8, 2**8)
        self.layer_4 = torch.nn.Linear(2**8, 2**4)
        self.layer_5 = torch.nn.Linear(2**4, 2**2)
        self.layer_6 = torch.nn.Linear(2**2, 1)
        
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        x = F.relu(self.layer_5(x))
        x = self.layer_6(x) # Linear output
        return x

class SpiralEncoder(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features, spiral_indices):
        super(SpiralEncoder, self).__init__()

        # Define the number of features for each SpiralConv layer
        self.conv1 = SpiralConv(num_input_features, 2**10, spiral_indices)
        self.conv2 = SpiralConv(2**10, 2**9, spiral_indices)
        self.conv3 = SpiralConv(2**9, 2**8, spiral_indices)
        self.conv4 = SpiralConv(2**8, num_output_features, spiral_indices)

    def forward(self, x, batch):
        # Forward pass for each SpiralConv layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Global mean pooling to get the whole graph embedding
        x = global_mean_pool(x, batch)
        return x

class SpiralConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.layer = torch.nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)
