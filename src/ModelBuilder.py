import torch
import torch.nn.functional as F



class TabularClassifier(torch.nn.Module):
    def __init__(self, num_input_features):
        torch.manual_seed(12345)
        super(TabularClassifier, self).__init__()
        
        self.classifier = MLPClassifier(num_input_features, 1)

    def forward(self, data):
        x = self.classifier(data)
        return torch.sigmoid(x)
        
 
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

        

