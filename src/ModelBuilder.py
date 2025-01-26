import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelBuilder:
    def __init__(self, config):
        """
        Initialize the model builder with the provided configuration.
        """
        self.config = config
        self.task = config['task']

    def build(self):
        """
        Build and return the specified model architecture.
        
        Returns:
            nn.Module: Constructed neural network
        """
        if self.config["skip_building"]:
            print("-- Skipped building architecture --")
            return
        
        print("-- Building Architecture --")
        model = None
        if self.config['architecture'] == 'mlp':
            model = MLP(
                input_dim=24,  # 4 points × 3 coordinates × 2 tetrahedra
                hidden_dims=[128, 128, 128],
                output_dim=self._determine_output_dim(),
                activation=self.config['activation_function'],
                dropout_rate=self.config['dropout_rate'],
                task=self.task
            )
        elif self.config['architecture'] == 'deep_set':
            model = DeepSet(
                point_dim=3,  # Each point has x, y, z coordinates
                hidden_dims=[128, 128, 128],
                output_dim=self._determine_output_dim(),
                activation=self.config['activation_function'],
                dropout_rate=self.config['dropout_rate'],
                task=self.task
            )
        else:
            raise ValueError(f"Unsupported architecture: {self.config['architecture']}")
        
        print("---- Architecture Built ----")
        return model

    def _determine_output_dim(self):
        """
        Determine the output dimension based on the task.
        """
        if self.task in ['binary_classification', 'regression']:
            return 1
        elif self.task == 'classification_and_regression':
            return 2
        else:
            raise ValueError(f"Unsupported task: {self.task}")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation, dropout_rate, task):
        super().__init__()
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

        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.task = task

    def forward(self, x):
        x = self.network(x)
        return task_specific_output(x, self.task)
    
    def predict(self, x):
        """
        Generate predictions for classification, regression, or combined tasks.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Predictions.
        """
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            logits = self(x)
            if self.task == 'binary_classification':
                return (logits > 0.5).int()
            elif self.task == 'regression':
                return logits
            elif self.task == 'classification_and_regression':
                cls_prediction = (logits[:, 0] > 0.5).int().unsqueeze(1)
                reg_prediction = logits[:, 1].unsqueeze(1) 
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

def task_specific_output(output, task):
    if task == 'binary_classification':
        return torch.sigmoid(output)
    elif task == 'regression':
        return output
    elif task == 'classification_and_regression':
        # Split the output into classification and regression heads
        cls_output = torch.sigmoid(output[:, 0].unsqueeze(1))  # Classification
        reg_output = output[:, 1].unsqueeze(1)  # Regression
        return torch.cat([cls_output, reg_output], dim=1)
    else:
        raise ValueError(f"Unsupported task: {task}")
