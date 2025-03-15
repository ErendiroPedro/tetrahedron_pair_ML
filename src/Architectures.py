import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseNet(nn.Module, ABC):
    """Base network with common functionality for all models"""
    def __init__(self, activation, task, regression_scale_factor=1):
        super().__init__()
        self.task = task
        self.activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'leakyrelu': nn.LeakyReLU,
            'elu': nn.ELU
        }
        self.activation = self.activation_map[activation]()
        self.regression_scale_factor = nn.Parameter(
            torch.tensor(regression_scale_factor), requires_grad=False
        )
        self.classifier_branch = None
        self.regressor_branch = None

    @abstractmethod
    def _forward_shared(self, x):
        """To be implemented by child classes"""
        pass

    def _build_branch(self, layer_dims, regressor = False):
        """Helper to create sequential branches"""
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:  # No activation after last layer
                layers.append(self.activation)
        if regressor:
            layers.append(nn.Softplus())
        return nn.Sequential(*layers)

    def forward(self, x):
        """Common forward logic for all networks"""
        shared_out = self._forward_shared(x)
        
        if self.task == 'classification_and_regression':
            cls_out = self.classifier_branch(shared_out)
            reg_out = self.regressor_branch(shared_out) 

            if self.training:
                reg_out =reg_out * self.regression_scale_factor
            else:
                reg_out = reg_out / self.regression_scale_factor

            return torch.cat([
                cls_out,
                reg_out
            ], dim=1)
        
        elif self.task == 'binary_classification':
            return self.classifier_branch(shared_out)

        elif self.task == 'regression':
            reg_out = self.regressor_branch(shared_out) 

            if self.training:
                reg_out = reg_out * self.regression_scale_factor
            else:
                reg_out = reg_out / self.regression_scale_factor

            return reg_out
        raise ValueError(f"Unknown task: {self.task}")

    def predict(self, x):
        """Common prediction logic for all networks"""
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
                return torch.stack([cls_prediction, reg_prediction], dim=1)         
            raise ValueError(f"Unknown task: {self.task}")

class MLP(BaseNet):
    def __init__(self, input_dim, shared_layers, classification_head, regression_head, 
                 activation, dropout_rate, task, regression_scale_factor=1):
        super().__init__(activation, task, regression_scale_factor)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Build shared layers
        shared_dims = [input_dim] + shared_layers
        self.shared_layers = nn.ModuleList()
        for i in range(len(shared_dims)-1):
            self.shared_layers.append(nn.Linear(shared_dims[i], shared_dims[i+1]))
        
        # Initialize branches
        self.classifier_branch = self._build_branch([shared_dims[-1]] + classification_head)
        self.regressor_branch = self._build_branch([shared_dims[-1]] + regression_head, regressor = True)

    def _forward_shared(self, x):
        """MLP-specific shared forward pass"""
        for layer in self.shared_layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Main path components
        self.norm1 = nn.LayerNorm(input_dim)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, output_dim)
        
        self.norm2 = nn.LayerNorm(output_dim)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)
        
        # Skip connection (identity or projection)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x):
        # Skip connection
        residual = self.skip(x)
        
        # Main path with pre-activation pattern
        # First block: norm -> activation -> weight
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.fc1(out)
        
        # Second block: norm -> activation -> weight
        # out = self.norm2(out)
        # out = self.relu2(out)
        # out = self.fc2(out)
        
        # Add skip connection to output
        return out + residual
class DeepSet(BaseNet):
    """DeepSet architecture with shared base"""
    def __init__(self, input_dim, activation, dropout_rate, task):
        super().__init__(activation, task)
        
        # Determine number of tetrahedrons based on input_dim.
        # 24 means two tetrahedrons (2*4*3), 12 means one tetrahedron (1*4*3).
        if input_dim == 24:
            self.num_tets = 2
            # For two tetrahedrons, the concatenated feature vector is 2*128 = 256.
            self.post_concat_residual = ResidualBlock(input_dim=32, output_dim=16, dropout_rate=dropout_rate)
        elif input_dim == 12:
            self.num_tets = 1
            # For a single tetrahedron, the feature vector is 128.
            self.post_concat_residual = ResidualBlock(input_dim=128, output_dim=64, dropout_rate=dropout_rate)
        else:
            raise ValueError("input_dim must be either 12 or 24")

        self.tetra_residual = ResidualBlock(input_dim=3, output_dim=128, dropout_rate=dropout_rate)
        self.final_fc = nn.Linear(64, 64)
        self.final_activation = nn.ReLU()

        # Initialize task branches
        self._init_branches()

    def _init_branches(self):
        """Initialize task-specific branches"""
        self.classifier_branch = self._build_branch([64, 32, 16, 1])
        self.regressor_branch = self._build_branch([64, 64, 1])

    def _forward_shared(self, x):
        """Shared DeepSet processing with dynamic tetrahedron count"""
        if self.num_tets == 2:
            # Reshape to (batch, 2, 4, 3) for two tetrahedrons.
            x = x.view(-1, 2, 4, 3)
            tet1 = self.tetra_residual(x[:, 0, :, :])
            tet1 = torch.max(tet1, dim=1)[0]
            tet2 = self.tetra_residual(x[:, 1, :, :])
            tet2 = torch.max(tet2, dim=1)[0]
            x = torch.cat([tet1, tet2], dim=1)
        else:  # self.num_tets == 1
            # Reshape to (batch, 4, 3) for a single tetrahedron.
            x = x.view(-1, 4, 3)
            tet1 = self.tetra_residual(x)
            tet1 = torch.max(tet1, dim=1)[0]
            x = tet1

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

