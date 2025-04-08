import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class CArchitectureManager:
    """
    Manages the architecture of the neural networks.
    """
    def __init__(self, config):
        self.config = config

    def get_model(self):
        """Returns the model based on the configuration"""
        model_type = self.config['model_type']
        if model_type == 'MLP':
            return self.MLP(**self.config['MLP'])
        elif model_type == 'DeepSet':
            return self.DeepSet(**self.config['DeepSet'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    class BaseNet(nn.Module, ABC):
        """Base network with common functionality for all models"""
        def __init__(self, activation, task, shared_dim, classification_head, regression_head, volume_scale_factor=1):
            super().__init__()
            self.task = task
            self.activation_map = {
                'relu': nn.ReLU,
                'tanh': nn.Tanh,
                'leakyrelu': nn.LeakyReLU,
                'elu': nn.ELU
            }
            self.activation = self.activation_map[activation]()
            self.volume_scale_factor = volume_scale_factor

            self.classifier_head= self._build_task_head([shared_dim] + classification_head)
            self.regressor_head= self._build_task_head([shared_dim] + regression_head, is_regression=True)

        @abstractmethod
        def _forward(self, x):
            """To be implemented by child classes"""
            pass

        def _build_task_head(self, hidden_layers, is_regression=False):
            """Helper to create sequential branches"""
            layers = []
            for i in range(len(hidden_layers) - 1):
                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                if i < len(hidden_layers) - 2:  # No activation after last layer
                    layers.append(self.activation)

            if is_regression:
                layers.append(nn.ReLU())  # Ensure positive volume predictions

            return nn.Sequential(*layers)

        def forward(self, x):
            """Common forward logic for all networks"""
            shared_out = self._forward(x)

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
                    return (processed_embeddings > 0.5).int().squeeze() # Using binary cross-entropy with logits, sgimoid is applied in the loss function
                elif self.task == 'regression':
                    return processed_embeddings.squeeze() / self.volume_scale_factor
                elif self.task == 'classification_and_regression':
                    cls_prediction = (processed_embeddings[:, 0] > 0.5).int().squeeze() # Using binary cross-entropy with logits, sgimoid is applied in the loss function
                    reg_prediction = processed_embeddings[:, 1].squeeze() / self.volume_scale_factor
                    return torch.stack([cls_prediction, reg_prediction], dim=1)         
                raise ValueError(f"Unknown task: {self.task}")
    
    class MLP(BaseNet):
        def __init__(self, input_dim, shared_layers, classification_head, regression_head, 
                    activation, task, volume_scale_factor=1):
            super().__init__(activation, task, shared_layers[-1], classification_head, regression_head, volume_scale_factor=volume_scale_factor)
            
            self.shared_layers = self._build_shared_network(input_dim, shared_layers)

        def _build_shared_network(self, input_dim, layer_sizes):
            dimensions = [input_dim] + layer_sizes
            layers = []
            # Pair consecutive dimensions (input_dim, layer1), (layer1, layer2)...
            for i, (in_dim, out_dim) in enumerate(zip(dimensions, dimensions[1:])):
                layers.append(nn.Linear(in_dim, out_dim))
                # Only add activation if it's NOT the last layer
                # if i < len(layer_sizes) - 1:
                #     layers.append(self.activation)
            return nn.Sequential(*layers)
        def _forward(self, x):
            return self.shared_layers(x)
   
    class TPNet(BaseNet):
        def __init__(self, input_dim, per_tet_layers, shared_layers, classification_head, regression_head, activation, task, volume_scale_factor=1):
            super().__init__(activation, task, shared_layers[-1], classification_head, regression_head, volume_scale_factor=volume_scale_factor)
            
            self.input_dim = input_dim
            self.per_tet_layers = self._build_tetrahedronwise_tet_network(per_tet_layers)
            self.shared_layers = self._build_shared_network(shared_layers[0], shared_layers)
            self.tuple_network = nn.Linear(input_dim, shared_layers[-1])
        
        def _build_tetrahedronwise_tet_network(self, layer_sizes):
            """Single residual block covering whole tetrahedron processing"""
            class TetrahedronwiseNetwork(nn.Module):
                def __init__(self, layer_sizes, activation):
                    super().__init__()
                    # Ensure final output matches input dimensions (12)
                    processing_dims = [12] + layer_sizes + [12]
                    
                    # Main processing sequence
                    self.processor = nn.Sequential(
                        *sum([[nn.Linear(in_d, out_d), activation] 
                            for in_d, out_d in zip(processing_dims[:-2], processing_dims[1:-1])], []),
                        nn.Linear(processing_dims[-2], processing_dims[-1])
                    )
                    
                    # Final activation after residual
                    self.final_activation = activation

                def forward(self, x):
                    # Original input for residual
                    identity = x
                    
                    # Process entire tetrahedron
                    processed = self.processor(x)
                    
                    # Add residual and apply final activation
                    return self.final_activation(processed + identity)

            return TetrahedronwiseNetwork(layer_sizes, self.activation)


        def _build_pointwise_tet_network(self, layer_sizes):
            """Build network with correct residual projections"""
            class PointwiseTetNetwork(nn.Module):
                def __init__(self, layer_sizes, activation):
                    super().__init__()
                    # Per-point processing
                    per_point_dims = [3] + layer_sizes
                    self.point_layers = nn.Sequential(
                        *sum([[nn.Linear(in_d, out_d), activation] 
                            for in_d, out_d in zip(per_point_dims[:-1], per_point_dims[1:])], [])
                    )
                    
                    # CORRECTED RESIDUAL PROJECTION
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

                def forward(self, x):
                    # Original input for final residual
                    input_residual = x
                    
                    # Process points
                    points = x.view(-1, 4, 3)
                    processed = self.point_layers(points)
                    
                    residual = self.point_res_proj(points)  # [batch, 4, layer_size]
                    processed = processed + residual  # [batch, 4, layer_size]
                    
                    # Aggregate and process
                    pooled = torch.max(processed, dim=1)[0]  # [batch, layer_size]
                    output = self.post_pool(pooled)  # [batch, 12]
                    
                    # Final residual connection
                    return output + input_residual  # [batch, 12]

            return PointwiseTetNetwork(layer_sizes, self.activation)

        def _build_shared_network(self, input_dim, layer_sizes):
            """Shared network with residual connections"""

            class ResidualBlock(nn.Module):
                def __init__(self, in_dim, out_dim, activation):
                    super().__init__()
                    self.linear = nn.Linear(in_dim, out_dim)
                    self.activation = activation
                    self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
                    
                def forward(self, x):
                    return self.activation(self.linear(x) + self.res_proj(x))
            
            layers = []
            current_dim = input_dim
            
            for out_dim in layer_sizes:
                layers.append(ResidualBlock(current_dim, out_dim, self.activation))
                current_dim = out_dim
            
            return nn.Sequential(*layers)


        
        def process_tetrahedron(self, x):
            """
            Process a single tetrahedron.
            
            Args:
                x: Tensor of shape [batch_size, 12] (4 points × 3 coordinates)
            
            Returns:
                Tensor of shape [batch_size, per_tet_layers[-1]]
            """
            # Process through the per-tetrahedron network
            features = self.per_tet_layers(x)
            
            return features
        
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
            TPNet forward pass.
            
            Expected input format: batch of flat tensors with shape [batch_size, input_dim]
            Where input_dim is either:
            - 12 (single tetrahedron: 4 points × 3 coordinates)
            - 24 (two tetrahedra: 8 points × 3 coordinates)
            """
            
            # Store original inputs for tuple branch
            original_inputs = x.clone()
            
            # Determine if we're processing one or two tetrahedra based on input dimension
            if self.input_dim == 12:  # Single tetrahedron case
                # Process single tetrahedron
                tet_features = self.process_tetrahedron(x)
                
                # No need for max pooling in single tetrahedron case
                pooled_features = tet_features
                
            elif self.input_dim == 24:  # Two tetrahedra case
                # Split into two tetrahedra
                t1 = x[:, :12]  # First tetrahedron: first 12 values (4 points × 3 coordinates)
                t2 = x[:, 12:]  # Second tetrahedron: last 12 values (4 points × 3 coordinates)
                
                # Process each tetrahedron independently
                t1_features = self.process_tetrahedron(t1)
                t2_features = self.process_tetrahedron(t2)
                
                # Max pooling operation (element-wise max)
                pooled_features = torch.max(t1_features, t2_features)
                
            else:
                raise ValueError(f"Input dimension must be either 12 or 24, got {self.input_dim}")
            
            # Process combined features
            combined_embeddings = self.process_combined_features(pooled_features)
            
            # Process original inputs through tuple branch
            tuple_embeddings = self.tuple_network(original_inputs)
            
            # Final embeddings: combined embeddings + tuple embeddings
            final_embeddings = combined_embeddings + tuple_embeddings
            
            return final_embeddings

    # Helper class
    class ResidualBlock(nn.Module):
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

    class DeepSetEncoder(nn.Module):
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
                self.res_block1 = CArchitectureManager.ResidualBlock(3, hidden_dim, activation, dropout_rate)
                self.res_block2 = CArchitectureManager.ResidualBlock(hidden_dim, hidden_dim, activation, dropout_rate)
                self.final_fc = nn.Linear(hidden_dim, output_dim)
            elif input_dim == 24:
                # Two tetrahedra case
                self.res_block1 = CArchitectureManager.ResidualBlock(3, hidden_dim, activation, dropout_rate)
                self.res_block_pool = CArchitectureManager.ResidualBlock(hidden_dim, hidden_dim, activation, dropout_rate)
                self.res_block3 = CArchitectureManager.ResidualBlock(2 * hidden_dim, hidden_dim, activation, dropout_rate)
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
                x = x.view(batch, 2, 4, 3)
                x = x.view(batch * 2, 4, 3)
                x_vertex = x.view(-1, 3)
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

    # DeepSet model using the DeepSetEncoder inside the BaseNet framework
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
            self.deep_set_encoder = CArchitectureManager.DeepSetEncoder(input_dim, hidden_dim, output_dim, self.activation, dropout_rate)

        def _forward(self, x):
            return self.deep_set_encoder(x)
            
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
