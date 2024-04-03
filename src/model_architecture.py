import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class GraphPairClassifier(torch.nn.Module):
    def __init__(self, num_input_features):
        torch.manual_seed(12345)
        super(GraphPairClassifier, self).__init__()
        
        self.encoder = GCNEncoder(num_input_features, 128)
        self.classifier = MLPClassifier(256, 1)

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
    def __init__(self, num_input_features):
        torch.manual_seed(12345)
        super(TabularClassifier, self).__init__()
        
        self.classifier = MLPClassifier(num_input_features, 1)

    def forward(self, data):
        x = self.classifier(data)
        return torch.sigmoid(x)
    
####################
## Helper Classes ##
####################
        
class GCNEncoder(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(num_input_features, num_output_features)
        self.conv2 = GCNConv(num_output_features, num_output_features)
        self.conv3 = GCNConv(num_output_features, num_output_features)
        self.conv4 = GCNConv(num_output_features, num_output_features)
        self.conv5 = GCNConv(num_output_features, num_output_features)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = global_mean_pool(x, batch)  # Get whole graph embeddings
        return x
 
class MLPClassifier(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(MLPClassifier, self).__init__()
        self.layer_1 = torch.nn.Linear(num_input_features, 128)
        self.layer_2 = torch.nn.Linear(128, 64)
        self.layer_3 = torch.nn.Linear(64, 32)
        self.layer_4 = torch.nn.Linear(32, 16)
        self.layer_5 = torch.nn.Linear(16, num_output_features)
        
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        x = self.layer_5(x) # Linear output
        return x

        

