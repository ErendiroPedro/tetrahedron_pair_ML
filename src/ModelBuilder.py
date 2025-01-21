import torch
import torch.nn as nn

class ModelBuilder:
    def __init__(self, config):
        self.config = config
        
        self.activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leakyrelu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
    
    def _build_mlp(self, input_dim, hidden_dims, output_dim):
        """
        Builds a multi-layer perceptron with specified dimensions.
        
        Args:
            input_dim (int): Input dimension
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Output dimension
        """
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                self.activation_map[self.config['activation_function']],
                nn.Dropout(self.config['dropout_rate'])
            ])
            current_dim = hidden_dim
        
        # Add final linear layer without activation
        layers.append(nn.Linear(current_dim, output_dim))
        
        # Add task-specific output activation
        layers.append(TaskSpecificOutput(self.config['task']))
        
        return nn.Sequential(*layers)
    
    def _build_deep_set(self, point_dim, hidden_dims, output_dim):
        """
        Builds a DeepSet architecture for processing sets of points.
        
        Args:
            point_dim (int): Dimension of each point
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Output dimension
        """
        # Phi network processes each point independently
        phi_net = self._build_mlp(point_dim, hidden_dims[:-1], hidden_dims[-1])
        
        # Remove the TaskSpecificOutput from phi_net as it's only needed at the end
        phi_net = nn.Sequential(*list(phi_net.children())[:-1])
        
        # Rho network processes the sum of phi outputs
        rho_net = self._build_mlp(hidden_dims[-1], hidden_dims[:-1], output_dim)
        
        return DeepSetNetwork(phi_net, rho_net)
    
    def build(self):
        print("Building Architecture")
        """
        Builds the complete model architecture based on configuration.
        
        Returns:
            nn.Module: The constructed neural network
        """
        input_dim = 24  # 4 points × 3 coordinates × 2 tetrahedra
        hidden_dims = [128, 128, 128]  # Example dimensions, adjust as needed
        
        if self.config['task'] in ['binary_classification', 'Regression']:
            output_dim = 1
        else:  # Classification&Regression
            output_dim = 2  # One for classification, one for regression
        
        # Build the specified architecture
        if self.config['architecture'] == 'mlp':
            model = self._build_mlp(input_dim, hidden_dims, output_dim)
        elif self.config['architecture'] == 'deep_set':
            point_dim = 3  # Each point has x, y, z coordinates
            model = self._build_deep_set(point_dim, hidden_dims, output_dim)
        else:
            raise ValueError(f"Unsupported architecture: {self.config['architecture']}")
        
        return model

class DeepSetNetwork(nn.Module):
    """
    Implementation of DeepSet architecture for permutation-invariant processing.
    """
    def __init__(self, phi_net, rho_net):
        super().__init__()
        self.phi_net = phi_net
        self.rho_net = rho_net
    
    def forward(self, x):
        # Reshape input to process each point separately
        batch_size = x.size(0)
        points = x.view(batch_size, -1, 3)  # Reshape to (batch_size, num_points, 3)
        
        # Apply phi network to each point
        phi_output = self.phi_net(points)
        
        # Sum across points (permutation-invariant operation)
        summed = torch.sum(phi_output, dim=1)
        
        # Apply rho network to the sum
        return self.rho_net(summed)

class TaskSpecificOutput(nn.Module):
    """
    Module that handles different output activations based on the task.
    Supports binary classification (sigmoid), regression (no activation),
    and joint classification & regression tasks.
    """
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        if self.task == 'binary_classification':
            return self.sigmoid(x)
        elif self.task == 'Regression':
            return x
        else:  # Classification&Regression
            # Split the output into classification and regression heads
            cls_output = self.sigmoid(x[:, 0].unsqueeze(1))  # First output for classification
            reg_output = x[:, 1].unsqueeze(1)  # Second output for regression
            return torch.cat([cls_output, reg_output], dim=1)