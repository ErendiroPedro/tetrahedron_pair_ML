import torch.nn.functional as F
from src.Architectures import MLP, DeepSet
import pandas as pd
import os

class ModelBuilder:
    def __init__(self, config):
        """
        Initialize the model builder with the provided configuration.
        """
        self.config = config
        self.task = config['task']

    def _infer_input_shape(self):
        """
        Load dataset from the path and infer input shape.
        """
        data_path = os.path.join(self.config.get('processed_data_path'), "train", "train_data.csv")
        if not data_path:
            raise ValueError("Processed data path is not specified in the config.")

        try:
            sample_data = pd.read_csv(data_path, nrows=2)
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")

        example_input = sample_data.iloc[:, :-2]  # Exclude labels
        return example_input.shape[1]  # Number of input features

    def build(self):
        """
        Build and return the specified model architecture.
        
        Returns:
            nn.Module: Constructed neural network
        """
        
        print("-- Building Architecture --")

        model = None
        input_shape = self._infer_input_shape()

        if self.config['architecture'] == 'mlp':
            model = MLP(
                input_dim=input_shape,
                shared_layers=[128, 64],
                activation=self.config['activation_function'],
                dropout_rate=self.config['dropout_rate'],
                task=self.task
            )
        elif self.config['architecture'] == 'deepset':
            model = DeepSet(
                input_dim=input_shape,
                activation=self.config['activation_function'],
                dropout_rate=self.config['dropout_rate'],
                task=self.task
            )
        # elif self.config['architecture'] == 'pointnet':
        #     model = TetrahedraPointNet(
        #         dropout_rate=self.config['dropout_rate'],
        #         task=self.task
        #     )
        else:
            raise ValueError(f"Unsupported architecture: {self.config['architecture']}")
        
        print("---- Architecture Built ----")
        return model