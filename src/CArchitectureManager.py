import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class CArchitectureManager:
    """
    Manages the architecture of the neural networks.
    """
    activation_map = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'leakyrelu': nn.LeakyReLU,
        'elu': nn.ELU
    }

    def __init__(self, config):
        self.config = config
        self.model_map = {
            'mlp': self.MLP,
            'deepset': self.DeepSet,
            'tpnet': self.TPNet
        }
    
    def _get_input_dim(self):
        import pandas as pd
        import os
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

    def get_model(self):
        """Returns the model based on the configuration"""
        arch_type = self.config['architecture']['use_model'].lower()
        model_class = self.model_map.get(arch_type)
        
        if not model_class:
            raise ValueError(f"Unknown architecture: {arch_type}")

        arch_config = self.config['architecture'].get(arch_type)
        if not arch_config:
            raise ValueError(f"{arch_type} configuration missing in architecture settings")

        common_params = {
            'input_dim' : self._get_input_dim(),
            'activation': self.config['common_paramers']['activation_function'],
            'task': self.config['common_paramers']['task'],
            'volume_scale_factor': self.config['common_paramers']['volume_scale_factor'],
        }
        
        return model_class(common_params, arch_config)
    
    class BaseNet(nn.Module, ABC):
        """Base network with common functionality for all models"""
        def __init__(self, common_params, mlp_params):
            super().__init__()
            self.input_dim = common_params['input_dim']
            self.task = common_params['task']
            self.activation = CArchitectureManager.activation_map[common_params['activation']]()
            self.dropout = common_params.get('dropout', 0.0)
            self.volume_scale_factor = common_params['volume_scale_factor']
            self.classifier_head= self._build_task_head([mlp_params['shared_layers'][-1]] + mlp_params['classification_head'])
            self.regressor_head= self._build_task_head([mlp_params['shared_layers'][-1]] + mlp_params['regression_head'], is_regression=True)

        @abstractmethod
        def _forward(self, x):
            """To be implemented by child classes"""
            pass

        def _build_task_head(self, hidden_layers, is_regression=False):

            if len(hidden_layers) == 1:
                return nn.Identity()
            elif not hidden_layers:
                raise ValueError("Hidden layers must be specified")
            
            layers = []
            for i in range(len(hidden_layers) - 1):
                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                # Add activation after all layers except the last one
                if i < len(hidden_layers) - 2:
                    layers.append(self.activation)

            # For volume prediction, ensure positive outputs with ReLU
            if is_regression:
                layers.append(nn.ReLU())  # Ensure positive volume predictions

            # For classification, binary_cross_entropy_with_logits handles the sigmoid
    
            return nn.Sequential(*layers)

        def build_shared_network(self, input_dim, layer_sizes, add_final_activation=False):
            """
            Builds a shared network with the specified layer dimensions.
            
            Args:
                input_dim: Input dimension
                layer_sizes: List of hidden layer dimensions 
                add_final_activation: Whether to add activation after the final layer (default: False)
                
            Returns:
                nn.Sequential: Neural network module for the shared network
            """
            # Handle empty layer_sizes
            if not layer_sizes:
                return nn.Identity()
            
            dimensions = [input_dim] + layer_sizes
            layers = []
            # Pair consecutive dimensions (input_dim, layer1), (layer1, layer2)...
            for i, (in_dim, out_dim) in enumerate(zip(dimensions, dimensions[1:])):

                layers.append(nn.Linear(in_dim, out_dim))
                
                is_last_layer = (i == len(dimensions) - 2)
                if not is_last_layer or add_final_activation:
                    layers.append(self.activation)
                    
                if self.dropout > 0 and not is_last_layer:
                    layers.append(nn.Dropout(self.dropout))
            
            return nn.Sequential(*layers)

        def forward(self, x):
            """Common forward logic for all networks"""
            shared_out = self._forward(x)
            shared_out = self.shared_layers(shared_out)

            if self.task == 'classification_and_regression':
                return torch.cat([
                    self.classifier_head(shared_out),
                    self.regressor_head(shared_out)
                ], dim=1)
            
            elif self.task == 'binary_classification':
                return self.classifier_head(shared_out)

            elif self.task == 'regression':
                output = self.regressor_head(shared_out)
                return output

            raise ValueError(f"Unknown task: {self.task}")

        def predict(self, x):
            """Common prediction logic for all networks"""
            self.eval()
            with torch.no_grad():
                processed_embeddings = self(x) 

                if self.task == 'binary_classification':
                    return (processed_embeddings > 0.5).int().squeeze()
                elif self.task == 'regression':
                    return processed_embeddings.squeeze() / self.volume_scale_factor
                elif self.task == 'classification_and_regression':
                    cls_prediction = (processed_embeddings[:, 0] > 0.5).int().squeeze()
                    reg_prediction = processed_embeddings[:, 1].squeeze() / self.volume_scale_factor
                    return torch.stack([cls_prediction, reg_prediction], dim=1)   
                      
                raise ValueError(f"Unknown task: {self.task}")

    class MLP(BaseNet):
        def __init__(self, common_params, mlp_params):     
            super().__init__(common_params=common_params, mlp_params=mlp_params)

            self.shared_layers = self.build_shared_network(common_params['input_dim'], mlp_params['shared_layers'])

        def _forward(self, x):
            return x # BaseNet is mlp
    
    class TPNet(BaseNet):
        def __init__(self, common_params, tpnet_params):     
            mlp_params = {
                'shared_layers': tpnet_params['shared_layers'],
                'classification_head': tpnet_params['classification_head'],
                'regression_head': tpnet_params['regression_head'],
            }
            super().__init__(common_params=common_params, mlp_params=mlp_params)
            self.shared_layers = self.build_shared_network(12, mlp_params['shared_layers'])
            self.per_tet_layers = self._build_pointwise_tet_network(tpnet_params['per_tet_layers'])
            
            # Global residual projection
            self.global_residual_proj = nn.Linear(common_params['input_dim'], mlp_params['shared_layers'][0]) if common_params['input_dim'] != mlp_params['shared_layers'][0] else nn.Identity()
        
        def _build_tetrahedronwise_tet_network(self, layer_sizes):
            """Build network that processes entire tetrahedra at once with residual connections"""
            class TetrahedronwiseNetwork(nn.Module):
                def __init__(self, layer_sizes, activation):
                    super().__init__()
                    self.activation = activation
                    tet_dims = [12] + layer_sizes
                    
                    # Main processing layers for whole tetrahedron
                    layers = []
                    for i in range(len(tet_dims) - 1):
                        layers.append(nn.Linear(tet_dims[i], tet_dims[i+1]))
                        layers.append(activation)
                    self.tet_layers = nn.Sequential(*layers)
                    
                    # Final output projection
                    self.output_proj = nn.Sequential(
                        nn.Linear(layer_sizes[-1], 12),
                        activation
                    )
                    
                    # Input-to-output residual connection
                    self.input_res_proj = nn.Identity()

                def forward(self, x):
                    # Original input for final residual (batch_size, 12)
                    input_residual = self.input_res_proj(x)
                    
                    # Process entire tetrahedron at once
                    processed = self.tet_layers(x)  # [batch, layer_size]
                    
                    # Project to output dimension
                    output = self.output_proj(processed)  # [batch, 12]
                    
                    # Final residual connection
                    return self.activation(output + input_residual)  # [batch, 12]

            return TetrahedronwiseNetwork(layer_sizes, self.activation)

        def _build_pointwise_tet_network(self, layer_sizes):
            """Build network with internal max pooling and residual connections"""
            class PointwiseTetNetwork(nn.Module):
                def __init__(self, layer_sizes, activation):
                    super().__init__()
                    # Per-point processing
                    per_point_dims = [3] + layer_sizes
                    self.point_layers = nn.Sequential(
                        *sum([[nn.Linear(in_d, out_d), activation] 
                            for in_d, out_d in zip(per_point_dims[:-1], per_point_dims[1:])], [])
                    )
                    
                    # Residual projection for each point
                    self.point_res_proj = (
                        nn.Linear(3, layer_sizes[-1]) 
                        if layer_sizes[-1] != 3 
                        else nn.Identity()
                    )
                    
                    # Post-pool processing
                    self.post_pool = nn.Sequential(
                        nn.Linear(layer_sizes[-1], 24),
                        activation,
                        nn.Linear(24, 12)
                    )
                    
                    # Input-to-output residual projection
                    self.input_res_proj = nn.Linear(12, 12)
                    
                    # Final activation
                    self.final_activation = activation

                def forward(self, x):
                    # Original input for final residual
                    input_residual = self.input_res_proj(x)
                    
                    # Process points
                    points = x.view(-1, 4, 3)
                    processed = self.point_layers(points)
                    
                    # Point-wise residual
                    residual = self.point_res_proj(points)  # [batch, 4, layer_size]
                    processed = processed + residual  # [batch, 4, layer_size]
                    
                    # Internal max pooling - aggregate across points within tetrahedron
                    pooled = torch.max(processed, dim=1)[0]  # [batch, layer_size]
                    output = self.post_pool(pooled)  # [batch, 12]
                    
                    # Final residual connection
                    return self.final_activation(output + input_residual)  # [batch, 12]

            return PointwiseTetNetwork(layer_sizes, self.activation)

        def process_tetrahedron(self, x):
            """
            Process a single tetrahedron with internal max pooling.
            
            Args:
                x: Tensor of shape [batch_size, 12] (4 points × 3 coordinates)
            
            Returns:
                Tensor of shape [batch_size, 12]
            """
            
            return self.per_tet_layers(x)
        
        def process_combined_features(self, x):
            """
            Process combined features through shared layers.
            
            Args:
                x: Tensor of combined features
                
            Returns:
                Processed features
            """
            return self.shared_layers(x)
        
        def _forward(self, x):
            """
            Improved TPNet forward pass with internal max pooling and global residual.
            
            Expected input format: batch of flat tensors with shape [batch_size, input_dim]
            Where input_dim is either:
            - 12 (single tetrahedron: 4 points × 3 coordinates)
            - 24 (two tetrahedra: 8 points × 3 coordinates)
            """
            
            # Store original inputs for global residual connection
            original_inputs = x.clone()
            
            # Determine if we're processing one or two tetrahedra based on input dimension
            if self.input_dim == 12:  # Single tetrahedron case

                tet_features = self.process_tetrahedron(x)
                
                pooled_features = tet_features # No need for external max pooling in single tetrahedron case
                
            elif self.input_dim == 24:  # Two tetrahedra case
  
                t1 = x[:, :12]
                t2 = x[:, 12:]
                
                t1_features = self.process_tetrahedron(t1)
                t2_features = self.process_tetrahedron(t2)

                pooled_features = torch.max(t1_features, t2_features) # permutation invariant pooling
                
            else:
                raise ValueError(f"Input dimension must be either 12 or 24, got {self.input_dim}")
            
            # Process combined features
            combined_embeddings = self.process_combined_features(pooled_features)
            
            # Apply global residual connection
            global_residual = self.global_residual_proj(original_inputs)
            
            # Final embeddings: combined embeddings + global residual
            final_embeddings = combined_embeddings + global_residual
            
            return final_embeddings

    class DeepSet(BaseNet):
        def __init__(self, input_dim, deep_set_params, classification_head, regression_head, 
                    activation, task, volume_scale_factor=1):
            """
            deep_set_params is a dict containing:
            - 'hidden_dim': number of hidden units used in residual blocks.
            - 'output_dim': desired output dimension of the deep set encoder.
            - Optional: 'dropout_rate' (default 0.1)
            """
            self.input_dim = input_dim
            output_dim = deep_set_params['output_dim']
            super().__init__(activation, task, output_dim, classification_head, regression_head,
                            volume_scale_factor=volume_scale_factor)
            hidden_dim = deep_set_params['hidden_dim']
            dropout_rate = deep_set_params.get('dropout_rate', 0.1)
            self.deep_set_encoder = CArchitectureManager._DeepSetEncoder(input_dim, hidden_dim, output_dim, self.activation, dropout_rate)

        def _forward(self, x):
            return self.deep_set_encoder(x)


    # Helper classes #
    class _DeepSetEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, activation, dropout_rate=0.1):
            """
            Parameters:
            - input_dim: 12 for a single tetrahedron or 24 for two tetrahedra.
            - hidden_dim: number of units used in the residual blocks.
            - output_dim: final embedding dimension (will be used as shared_dim).
            - activation: an activation module (e.g. nn.ReLU()).
            - dropout_rate: dropout rate in residual blocks.
            """
            super().__init__()
            self.input_dim = input_dim
            self.activation = activation
            
            # Initialize with dummy modules
            self.res_block1 = nn.Identity()
            self.res_block2 = nn.Identity()
            self.res_block_pool = nn.Identity()
            self.res_block3 = nn.Identity()
            self.final_fc = nn.Linear(1, output_dim)  # Minimum valid Linear layer
                   
            # Create the blocks based on input dimension
            if input_dim == 12:
                # Single tetrahedron case
                self.res_block1 = CArchitectureManager._ResidualBlock(3, hidden_dim, activation, dropout_rate)
                self.res_block2 = CArchitectureManager._ResidualBlock(hidden_dim, hidden_dim, activation, dropout_rate)
                self.final_fc = nn.Linear(hidden_dim, output_dim)
            elif input_dim == 24:
                # Two tetrahedra case
                self.res_block1 = CArchitectureManager._ResidualBlock(3, hidden_dim, activation, dropout_rate)
                self.res_block_pool = CArchitectureManager._ResidualBlock(hidden_dim, hidden_dim, activation, dropout_rate)
                self.res_block3 = CArchitectureManager._ResidualBlock(2 * hidden_dim, hidden_dim, activation, dropout_rate)
                self.final_fc = nn.Linear(hidden_dim, output_dim)
            else:
                raise ValueError("input_dim must be either 12 (single tet) or 24 (two tets)")

        def forward(self, x):
            if self.input_dim == 12:
                # Single tetrahedron processing
                batch = x.size(0)
                x = x.view(batch, 4, 3)
                x_vertex = x.view(-1, 3)
                x_processed = self.res_block1(x_vertex)
                x_processed = x_processed.view(batch, 4, -1)
                x_pool, _ = torch.max(x_processed, dim=1)
                x_pool = self.res_block2(x_pool)
                out = self.final_fc(x_pool)
                return self.activation(out)
            elif self.input_dim == 24:
                # Two tetrahedra processing
                batch = x.size(0)
                x = x.reshape(batch, 2, 4, 3)
                x = x.reshape(batch * 2, 4, 3)
                x_vertex = x.reshape(-1, 3)
                x_processed = self.res_block1(x_vertex)
                x_processed = x_processed.view(batch * 2, 4, -1)
                x_pool, _ = torch.max(x_processed, dim=1)
                x_pool = self.res_block_pool(x_pool)
                x_pool = x_pool.view(batch, 2, -1)
                x_cat = x_pool.view(batch, -1)
                x_out = self.res_block3(x_cat)
                out = self.final_fc(x_out)
                return self.activation(out)
            else:
                raise ValueError(f"Unexpected input_dim: {self.input_dim}")
    
    class _ProcessingBlock(nn.Module):
        def __init__(self, in_dim, out_dim, activation):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.ln = nn.LayerNorm(out_dim)
            self.activation = activation

        def forward(self, x):
            out = self.linear(x)
            out = self.ln(out)
            out = self.activation(out)
            out = self.linear(x)
            return out
           
    class _ResidualBlock(nn.Module):
        def __init__(self, in_dim, out_dim, activation, dropout_rate=0.1):
            super().__init__()
            self.linear1 = nn.Linear(in_dim, out_dim)
            self.ln1 = nn.LayerNorm(out_dim)
            self.activation = activation
            self.dropout = nn.Dropout(dropout_rate)
            self.linear2 = nn.Linear(out_dim, out_dim)
            self.ln2 = nn.LayerNorm(out_dim)
            self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        def forward(self, x):
            out = self.linear1(x)
            out = self.ln1(out)
            out = self.activation(out)
            out = self.dropout(out)
            out = self.linear2(out)
            out = self.ln2(out)
            skip = self.proj(x)
            return self.activation(out + skip)
           
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
