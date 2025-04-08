import pandas as pd
import os

class CModelBuilder:
    def __init__(self, config):
        """
        Initialize the model builder with the provided configuration.
        """
        self.config = config
        self.task = config['task']
        self.dropout_rate = config['dropout_rate']

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
        from src.CArchitectureManager import CArchitectureManager
        
        print("-- Building Architecture --")
        
        # Get common parameters
        input_shape = self._infer_input_shape()
        common_params = {
            'input_dim': input_shape,
            'activation': self.config['activation_function'],
            'task': self.task,
            'volume_scale_factor': self.config['volume_scale_factor'],
        }
        
        # Define architecture builder functions
        architecture_builders = {
            'mlp': self._build_mlp,
            'tpnet': self._build_tpnet,
            'deepset': self._build_deepset
        }
        
        # Get selected architecture
        architecture_use = self.config['architecture'].get('use', '').lower()
        if not architecture_use:
            raise ValueError("Architecture 'use' not specified in config.")
        
        # Build the model
        if architecture_use in architecture_builders:
            model = architecture_builders[architecture_use](CArchitectureManager, common_params)
        else:
            raise ValueError(f"Unsupported architecture: {architecture_use}")
        
        print("---- Architecture Built ----")
        return model

    def _build_mlp(self, manager, common_params):
        """Build an MLP model with the specified parameters."""
        mlp_config = self.config['architecture'].get('mlp')
        if not mlp_config:
            raise ValueError("MLP configuration missing in architecture settings.")
        
        return manager.MLP(
            **common_params,
            shared_layers=mlp_config['shared_layers'],
            classification_head=mlp_config['classification_head'],
            regression_head=mlp_config['regression_head'],
        )

    def _build_tpnet(self, manager, common_params):
        """Build a TPNet model with the specified parameters."""
        tpnet_config = self.config['architecture'].get('tpnet')
        if not tpnet_config:
            raise ValueError("TPNet configuration missing in architecture settings.")
        
        return manager.TPNet(
            **common_params,
            per_tet_layers=tpnet_config['per_tet_layers'],
            shared_layers=tpnet_config['shared_layers'],
            classification_head=tpnet_config['classification_head'],
            regression_head=tpnet_config['regression_head'],
        )

    def _build_deepset(self, manager, common_params):
        """Build a DeepSet model with the specified parameters."""
        deepset_config = self.config['architecture'].get('deepset')
        if not deepset_config:
            raise ValueError("DeepSet configuration missing in architecture settings.")
        
        # Package DeepSet parameters into a dictionary
        deep_set_params = {
            'hidden_dim': deepset_config.get('hidden_dim', 64),
            'output_dim': deepset_config.get('output_dim', 32),
            'num_blocks': deepset_config.get('num_blocks', 1),
            'dropout_rate': self.dropout_rate
        }
        
        return manager.DeepSet(
            **common_params,
            deep_set_params=deep_set_params,
            classification_head=deepset_config['classification_head'],
            regression_head=deepset_config['regression_head']
        )