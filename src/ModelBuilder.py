import torch.nn.functional as F
from src.Architectures import MLP, DeepSet, PointNet

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
        
        print("-- Building Architecture --")
        model = None
        if self.config['architecture'] == 'mlp':
            model = MLP(
                input_dim=24,  # 4 points × 3 coordinates × 2 tetrahedra
                hidden_dims=[128],
                activation=self.config['activation_function'],
                dropout_rate=self.config['dropout_rate'],
                task=self.task
            )
        elif self.config['architecture'] == 'deep_set':
            model = DeepSet(
                point_dim=3,  # Each point has x, y, z coordinates
                hidden_dims=[128, 128, 128],
                activation=self.config['activation_function'],
                dropout_rate=self.config['dropout_rate'],
                task=self.task
            )
        elif self.config['architecture'] == 'pointnet':
            model = PointNet(
                point_dim=3,  # Each point has x, y, z coordinates
                hidden_dims=[64, 128, 256],  # Minimal hidden dimensions
                activation=self.config['activation_function'],
                dropout_rate=self.config['dropout_rate'],
                task=self.task
            )
        else:
            raise ValueError(f"Unsupported architecture: {self.config['architecture']}")
        
        print("---- Architecture Built ----")
        return model