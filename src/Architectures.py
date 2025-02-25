import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, dropout_rate, task):
        super().__init__()
        self.task = task
        self.activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'leakyrelu': nn.LeakyReLU,
            'elu': nn.ELU
        }
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                self.activation_map[activation](),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim

        self.shared_net = nn.Sequential(*layers)

        self.classifier_branch = nn.Sequential(
            nn.Linear(current_dim, current_dim // 2),
            self.activation_map[activation](),
            nn.Linear(current_dim // 2, 1)
        )

        self.regressor_branch = nn.Sequential(
            nn.Linear(current_dim, current_dim ),
            self.activation_map[activation](),
            nn.Linear(current_dim, 1)
        )

    def forward(self, x):
        shared_out = self.shared_net(x)
        if self.task == 'classification_and_regression':
            cls_out = torch.sigmoid(self.classifier_branch(shared_out))
            reg_out = self.regressor_branch(shared_out)
            return torch.cat([cls_out, reg_out], dim=1)
        elif self.task == 'binary_classification':
            return torch.sigmoid(self.classifier_branch(shared_out))
        elif self.task == 'regression':
            return self.regressor_branch(shared_out)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self(x)
            if self.task == 'binary_classification':
                return (logits > 0.5).int().squeeze()
            elif self.task == 'regression':
                return logits.squeeze()
            elif self.task == 'classification_and_regression':
                cls_prediction = (logits[:, 0] > 0.5).int().squeeze()
                reg_prediction = logits[:, 1].squeeze()
                return torch.cat([cls_prediction, reg_prediction], dim=1)

class DeepSet(nn.Module):
    """
    Implementation of the DeepSet architecture for permutation-invariant processing.
    """
    def __init__(self, point_dim, hidden_dims, output_dim, activation, dropout_rate, task):
        super().__init__()
        # Phi network processes individual points
        self.phi_net = MLP(
            input_dim=point_dim,
            hidden_dims=hidden_dims[:-1],
            output_dim=hidden_dims[-1],
            activation=activation,
            dropout_rate=dropout_rate,
            task=task
        )

        # Rho network processes the aggregated representation
        self.rho_net = MLP(
            input_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            output_dim=output_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            task=task  # Task-specific output applied here
        )
        self.task = task

    def forward(self, x):
        batch_size = x.size(0)
        points = x.view(batch_size, -1, 3)  # Reshape to (batch_size, num_points, 3)

        # Apply phi network to each point
        phi_output = self.phi_net(points)

        # Sum across points (permutation-invariant operation)
        summed = torch.sum(phi_output, dim=1)

        # Apply rho network to the summed representation
        x = self.rho_net(summed)

        # Apply task-specific output
        return task_specific_output(x, self.task)


class TNet(nn.Module):
    """Transformation network for input transformation."""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        
        # Compact version of T-Net
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, k*k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global feature aggregation
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        
        # FC layers to predict transformation matrix
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Add identity matrix to ensure the transformation is close to orthogonal
        identity = torch.eye(self.k, device=x.device).view(1, self.k*self.k).repeat(batch_size, 1)
        x = x + identity
        x = x.view(batch_size, self.k, self.k)
        
        return x

class CompactPointNet(nn.Module):

    def __init__(self, input_dim=3, hidden_dims=[64, 128, 64], activation='relu', dropout_rate=0.3, task='binary_classification'):
        super().__init__()
        self.task = task
        self.activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'leakyrelu': nn.LeakyReLU,
            'elu': nn.ELU
        }
        self.activation = self.activation_map[activation]

        # Input transformation network
        self.input_transform = TNet(k=input_dim)
        
        # Point feature extraction network
        self.point_feature_net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims[0], 1),
            nn.BatchNorm1d(hidden_dims[0]),
            self.activation(),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(hidden_dims[0], hidden_dims[1], 1),
            nn.BatchNorm1d(hidden_dims[1]),
            self.activation(),
            nn.Dropout(dropout_rate)
        )

        # Global feature aggregation (max pooling)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Task-specific branches (similar to your MLP model)
        last_hidden_dim = hidden_dims[1]
        
        self.classifier_branch = nn.Sequential(
            nn.Linear(last_hidden_dim, hidden_dims[2]),
            self.activation(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[2], 1)
        )
        
        self.regressor_branch = nn.Sequential(
            nn.Linear(last_hidden_dim, hidden_dims[2]),
            self.activation(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[2], 1)
        )

    def forward(self, x):

        batch_size = x.size(0)
        
        # Transpose for Conv1d operations
        x = x.transpose(1, 2)  # (batch_size, input_dim, num_points)
        
        # Input transformation
        trans_input = self.input_transform(x)
        x = torch.bmm(trans_input, x)  # Apply transformation
        
        # Extract point features
        point_features = self.point_feature_net(x)
        
        # Global feature aggregation (permutation-invariant)
        global_features = self.global_pool(point_features)
        global_features = global_features.view(batch_size, -1)
        
        # Task-specific outputs
        if self.task == 'classification_and_regression':
            cls_out = torch.sigmoid(self.classifier_branch(global_features))
            reg_out = self.regressor_branch(global_features)
            return torch.cat([cls_out, reg_out], dim=1)
        elif self.task == 'binary_classification':
            return torch.sigmoid(self.classifier_branch(global_features))
        elif self.task == 'regression':
            return self.regressor_branch(global_features)
    
    def predict(self, x):
        """
        Prediction method similar to your MLP model.
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            if self.task == 'binary_classification':
                return (logits > 0.5).int().squeeze()
            elif self.task == 'regression':
                return logits.squeeze()
            elif self.task == 'classification_and_regression':
                cls_prediction = (logits[:, 0] > 0.5).int()
                reg_prediction = logits[:, 1]
                return torch.cat([cls_prediction.unsqueeze(1), reg_prediction.unsqueeze(1)], dim=1)