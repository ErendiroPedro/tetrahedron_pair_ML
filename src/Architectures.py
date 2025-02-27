import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNet(nn.Module):
    """Base network with common functionality for all models"""
    def __init__(self, activation, task):
        super().__init__()
        self.task = task
        self.activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'leakyrelu': nn.LeakyReLU,
            'elu': nn.ELU
        }
        self.activation = self.activation_map[activation]()
        self.classifier_branch = None
        self.regressor_branch = None

    def _build_branch(self, layer_dims):
        """Helper to create sequential branches"""
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:  # No activation after last layer
                layers.append(self.activation)
        return nn.Sequential(*layers)

    def forward(self, x):
        """Common forward logic for all networks"""
        shared_out = self._forward_shared(x)
        
        if self.task == 'classification_and_regression':
            return torch.cat([
                self.classifier_branch(shared_out), # Logits for classification
                self.regressor_branch(shared_out) # Raw regression output
            ], dim=1)
        elif self.task == 'binary_classification':
            return self.classifier_branch(shared_out)
        elif self.task == 'regression':
            return self.regressor_branch(shared_out)
        raise ValueError(f"Unknown task: {self.task}")

    def predict(self, x):
        """Common prediction logic for all networks"""
        self.eval()
        with torch.no_grad():
            output = self(x)

            if self.task == 'binary_classification':
                return (torch.sigmoid(output) > 0.5).int().squeeze() 

            elif self.task == 'regression':
                return output.squeeze()

            elif self.task == 'classification_and_regression':
                cls_prediction = (torch.sigmoid(output[:, 0]) > 0.5).int().squeeze()
                reg_prediction = output[:, 1].squeeze()
                return torch.stack([cls_prediction, reg_prediction], dim=1)
            
            raise ValueError(f"Unknown task: {self.task}")

class MLP(BaseNet):
    """Multi-Layer Perceptron with shared base"""
    def __init__(self, input_dim, shared_layers, activation, dropout_rate, task):
        super().__init__(activation, task)
        
        # Build shared layers
        layers = []
        current_dim = input_dim
        for hidden_dim in shared_layers:
            layers += [
                nn.Linear(current_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ]
            current_dim = hidden_dim
        self.shared_net = nn.Sequential(*layers)

        # Initialize task branches
        self._init_branches(current_dim)

    def _init_branches(self, current_dim):
        """Initialize task-specific branches"""
        self.classifier_branch = self._build_branch([
            current_dim, 
            current_dim // 2,
            current_dim // 4,
            current_dim // 8,   
            1
        ])
        self.regressor_branch = self._build_branch([
            current_dim, 
            current_dim, 
            1
        ])

    def _forward_shared(self, x):
        """Shared network forward pass"""
        return self.shared_net(x)

class ResidualBlock(nn.Module):
    """Residual block (unchanged from original)"""
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.ln1 = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        return out + residual

class DeepSet(BaseNet):
    """DeepSet architecture with shared base"""
    def __init__(self, input_dim, activation, dropout_rate, task):
        super().__init__(activation, task)

        # Encoder components
        self.tetra_residual = ResidualBlock(input_dim=3, output_dim=128, dropout_rate=dropout_rate)
        self.post_concat_residual = ResidualBlock(input_dim=256, output_dim=64, dropout_rate=dropout_rate)
        self.final_fc = nn.Linear(64, 64)
        self.final_activation = nn.ReLU()

        # Initialize task branches
        self._init_branches()

    def _init_branches(self):
        """Initialize task-specific branches"""
        self.classifier_branch = self._build_branch([64, 32, 16, 1])
        self.regressor_branch = self._build_branch([64, 64, 1])

    def _forward_shared(self, x):
        """Shared DeepSet processing"""
        x = x.view(-1, 2, 4, 3)
        
        # Process tetrahedrons
        tet1 = self.tetra_residual(x[:, 0, :, :])
        tet1 = torch.max(tet1, dim=1)[0]
        
        tet2 = self.tetra_residual(x[:, 1, :, :])
        tet2 = torch.max(tet2, dim=1)[0]
        
        x = torch.cat([tet1, tet2], dim=1)
        x = self.post_concat_residual(x)
        return self.final_activation(self.final_fc(x))

# class GeometricAffineModule(nn.Module):
#     """Implements the geometric normalization with learnable affine parameters."""
#     def __init__(self, num_features, epsilon=1e-6):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.ones(num_features))
#         self.beta = nn.Parameter(torch.zeros(num_features))
#         self.epsilon = epsilon

#     def forward(self, x):
#         # x shape: (batch, num_groups, group_size, features)
#         mean = x.mean(dim=2, keepdim=True)  # Mean over group points
#         std = x.std(dim=2, keepdim=True, unbiased=False)  # Std over group points
#         x_norm = (x - mean) / (std + self.epsilon)
#         return self.alpha * x_norm + self.beta  # Learnable affine transformation

# class ResidualBlock(nn.Module):
#     """Residual block with optional geometric affine module."""
#     def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_affine=True):
#         super().__init__()
#         self.use_affine = use_affine
        
#         if self.use_affine:
#             self.affine = GeometricAffineModule(input_dim)
            
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, output_dim),
#             nn.LayerNorm(output_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(output_dim, output_dim),
#             nn.LayerNorm(output_dim)
#         )
#         self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

#     def forward(self, x):
#         if self.use_affine:
#             x = self.affine(x)
            
#         # Flatten for processing
#         batch, groups, points, feat = x.shape
#         x_flat = x.view(-1, feat)
#         out = self.net(x_flat)
#         out += self.skip(x_flat)
#         return out.view(batch, groups, points, -1)

# class TetrahedraPointNet(nn.Module):
#     """PointNet architecture with geometric affine residual blocks."""
#     def __init__(self, task='classification', dropout_rate=0.1):
#         super().__init__()
#         self.task = task
        
#         # Tetrahedron processing blocks
#         self.tetra_processor = nn.Sequential(
#             ResidualBlock(3, 256, dropout_rate, use_affine=True),
#             ResidualBlock(256, 256, dropout_rate, use_affine=True)
#         )
        
#         # Post-concatenation processing
#         self.post_process = nn.Sequential(
#             ResidualBlock(512, 128, dropout_rate, use_affine=False),
#             nn.Linear(128, 128),
#             nn.ReLU()
#         )
        
#         # Task heads
#         self.classifier = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )
#         self.regressor = nn.Linear(128, 1)

#     def forward(self, x):
#         # Reshape input: (batch, 24) -> (batch, 2, 4, 3)
#         x = x.view(-1, 2, 4, 3)
        
#         # Process both tetrahedrons
#         features = []
#         for i in range(2):
#             tetra = x[:, i:i+1, :, :]  # (batch, 1, 4, 3)
#             processed = self.tetra_processor(tetra)  # (batch, 1, 4, 256)
#             pooled = torch.max(processed, dim=2)[0]  # (batch, 1, 256)
#             features.append(pooled.squeeze(1))  # (batch, 256)
            
#         # Combine and process
#         x = torch.cat(features, dim=1)  # (batch, 512)
#         x = x.unsqueeze(1).unsqueeze(2)  # Add dummy dimensions
#         x = self.post_process(x).squeeze()  # (batch, 128)
        
#         # Task-specific outputs
#         if self.task == 'classification_and_regression':
#             cls = torch.sigmoid(self.classifier(x))
#             reg = self.regressor(x)
#             return torch.cat([cls, reg], dim=1)
#         elif self.task == 'binary_classification':
#             return torch.sigmoid(self.classifier(x))
#         elif self.task == 'regression':
#             return self.regressor(x)

#     def predict(self, x):
#         self.eval()
#         with torch.no_grad():
#             out = self(x)
#             if self.task == 'binary_classification':
#                 return (out > 0.5).int().squeeze()
#             elif self.task == 'regression':
#                 return out.squeeze()
#             elif self.task == 'classification_and_regression':
#                 return (out[:, 0] > 0.5).int().squeeze(), out[:, 1].squeeze()

