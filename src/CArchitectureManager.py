import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
from abc import ABC, abstractmethod

torch.set_default_dtype(torch.float64)

# ============================================================================
# ACTIVATION FUNCTIONS AND UTILITIES
# ============================================================================

class ActivationRegistry:
    """Central registry for activation functions"""
    
    ACTIVATION_MAP = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'leakyrelu': nn.LeakyReLU,
        'elu': nn.ELU,
        'gelu': nn.GELU,
        'swish': nn.SiLU,
        'mish': nn.Mish
    }
    
    @classmethod
    def get_activation(cls, activation_name):
        """Get activation function by name"""
        activation_class = cls.ACTIVATION_MAP.get(activation_name.lower())
        if not activation_class:
            raise ValueError(f"Unknown activation function: {activation_name}")
        return activation_class()

class AggregationFunctions:
    """Centralized aggregation operations for pooling"""
    
    @staticmethod
    def aggregate_tensors(tensor_stack, method='max'):
        """
        Aggregate stacked tensors using specified method
        
        Args:
            tensor_stack: Tensor of shape [batch_size, num_items, features]
            method: Aggregation method ('max', 'mean', 'sum', 'min', 'hadamard_prod')
        
        Returns:
            Aggregated tensor of shape [batch_size, features]
        """
        if method == 'max':
            result, _ = torch.max(tensor_stack, dim=1)
        elif method == 'mean':
            result = torch.mean(tensor_stack, dim=1)
        elif method == 'sum':
            result = torch.sum(tensor_stack, dim=1)
        elif method == 'min':
            result, _ = torch.min(tensor_stack, dim=1)
        elif method == 'hadamard_prod':
            # Use ReLU to handle negative values before product
            result = torch.prod(F.relu(tensor_stack), dim=1)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return result

# ============================================================================
# NETWORK BUILDING COMPONENTS
# ============================================================================

class NetworkBuilder:
    """Factory for building common network components"""
    
    @staticmethod
    def build_mlp_layers(layer_dimensions, activation_fn, dropout_rate=0.0, 
                        add_final_activation=False, add_final_relu=False):
        """
        Build MLP layers with consistent architecture
        
        Args:
            layer_dimensions: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
            activation_fn: Activation function instance
            dropout_rate: Dropout probability
            add_final_activation: Whether to add activation after final layer
            add_final_relu: Whether to add ReLU after final layer (for positive outputs)
        
        Returns:
            nn.Sequential: Built MLP layers
        """
        if len(layer_dimensions) < 2:
            raise ValueError("Need at least input and output dimensions")
        
        layers = []
        
        for i in range(len(layer_dimensions) - 1):
            layers.append(nn.Linear(layer_dimensions[i], layer_dimensions[i + 1]))
            
            is_final_layer = (i == len(layer_dimensions) - 2)
            
            # Add activation (except for final layer unless specified)
            if not is_final_layer or add_final_activation:
                layers.append(activation_fn)
            
            # Add dropout (except for final layer)
            if dropout_rate > 0.0 and not is_final_layer:
                layers.append(nn.Dropout(dropout_rate))
        
        # Add final ReLU for positive outputs (e.g., volume prediction)
        if add_final_relu:
            layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)

    @staticmethod
    def build_task_head(input_dim, head_layers, activation_fn, is_regression=False):
        """Build task-specific head (classification or regression)"""
        print(f"   Building task head: input_dim={input_dim}, head_layers={head_layers}, is_regression={is_regression}")
        
        if not head_layers:
            # Direct mapping from input to single output
            layers = [nn.Linear(input_dim, 1)]
            print(f"   Created direct mapping: {input_dim} -> 1")
            return nn.Sequential(*layers)
        
        # Check if head_layers already includes final dimension
        if head_layers[-1] == 1:
            # Configuration already includes final layer size
            dimensions = [input_dim] + head_layers
            print(f"   Config includes final layer: {dimensions}")
        else:
            # Configuration doesn't include final layer, add it
            dimensions = [input_dim] + head_layers + [1]
            print(f"   Adding final layer: {dimensions}")
        
        mlp = NetworkBuilder.build_mlp_layers(
            dimensions, activation_fn, add_final_relu=False
        )
        
        # Count actual layers
        linear_count = sum(1 for layer in mlp if isinstance(layer, nn.Linear))
        print(f"   Created MLP with {linear_count} linear layers")
        
        return mlp

# ============================================================================
# DATA UTILITIES
# ============================================================================

class DatasetInspector:
    """Utility for inspecting dataset properties"""
    
    @staticmethod
    def get_input_dimensions(processed_data_path):
        """
        Infer input dimensions from training data
        
        Args:
            processed_data_path: Path to processed data directory
            
        Returns:
            int: Number of input features
        """
        train_data_path = os.path.join(processed_data_path, "train", "train_data.csv")
        
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(f"Training data not found at: {train_data_path}")
        
        try:
            # Load only a few rows to inspect structure
            sample_data = pd.read_csv(train_data_path, nrows=5)
            
            # Exclude last 2 columns (IntersectionVolume, HasIntersection)
            feature_columns = sample_data.columns[:-2]
            input_dim = len(feature_columns)
            
            print(f"Dataset inspection: {input_dim} input features detected")
            return input_dim
            
        except Exception as e:
            raise ValueError(f"Error inspecting dataset: {e}")

# ============================================================================
# BASE NETWORK ARCHITECTURE
# ============================================================================

class BaseNet(nn.Module, ABC):
    """Base class for all neural network architectures"""
    
    def __init__(self, common_params, mlp_params):
        super().__init__()
        
        # Core parameters
        self.input_dim = common_params['input_dim']
        self.task = common_params['task']
        self.volume_scale_factor = common_params['volume_scale_factor']
        self.dropout_rate = common_params.get('dropout_rate', 0.0)
        
        # Activation function
        self.activation = ActivationRegistry.get_activation(common_params['activation'])
        
        # Build shared backbone
        self.shared_layers = self._build_shared_backbone(mlp_params['shared_layers'])
        
        # Build task-specific heads
        shared_output_dim = mlp_params['shared_layers'][-1] if mlp_params['shared_layers'] else self.input_dim
        
        self.classification_head = NetworkBuilder.build_task_head(
            shared_output_dim, mlp_params['classification_head'], 
            self.activation, is_regression=False
        )
        
        self.regression_head = NetworkBuilder.build_task_head(
            shared_output_dim, mlp_params['regression_head'], 
            self.activation, is_regression=True
        )
        
        # Ensure all parameters are float64
        self.to(torch.float64)
        self.double()

        # Verify conversion worked
        for name, param in self.named_parameters():
            if param.dtype != torch.float64:
                print(f"Warning: Parameter {name} is {param.dtype}, converting to float64")
                param.data = param.data.double()
    
    def _build_shared_backbone(self, shared_layer_sizes):
        """Build the shared feature extraction backbone"""
        if not shared_layer_sizes:
            return nn.Identity()
        
        # Get the input dimension for shared layers (depends on architecture)
        backbone_input_dim = self._get_backbone_input_dim()
        
        dimensions = [backbone_input_dim] + shared_layer_sizes
        return NetworkBuilder.build_mlp_layers(
            dimensions, self.activation, self.dropout_rate
        )
    
    @abstractmethod
    def _get_backbone_input_dim(self):
        """Get the input dimension for the shared backbone (architecture-specific)"""
        pass
    
    @abstractmethod
    def _extract_features(self, x):
        """Extract architecture-specific features from input"""
        pass

    def _debug_task_heads(self, shared_features):
        """Debug task head processing"""
        print(f"🔍 TASK HEAD DEBUG:")
        
        # Check shared features diversity
        print(f"   Shared features shape: {shared_features.shape}")
        print(f"   Shared sample 0: {shared_features[0, :3].tolist()}")
        print(f"   Shared sample 1: {shared_features[1, :3].tolist()}")
        print(f"   Shared features different: {not torch.allclose(shared_features[0], shared_features[1])}")
        
        # Debug classification head step by step
        print(f"   === CLASSIFICATION HEAD ===")
        cls_current = shared_features
        for i, layer in enumerate(self.classification_head):
            if isinstance(layer, nn.Linear):
                cls_current = layer(cls_current)
                print(f"   Cls layer {i}: shape = {cls_current.shape}")
                
                # Handle different output sizes
                if cls_current.shape[1] == 1:  # Final layer (scalar output)
                    print(f"   Cls layer {i}: sample 0 = {cls_current[0].item():.8f}")
                    print(f"   Cls layer {i}: sample 1 = {cls_current[1].item():.8f}")
                else:  # Intermediate layer (vector output)
                    print(f"   Cls layer {i}: sample 0 = {cls_current[0, :3].tolist()}")
                    print(f"   Cls layer {i}: sample 1 = {cls_current[1, :3].tolist()}")
                
                print(f"   Cls layer {i} different: {not torch.allclose(cls_current[0], cls_current[1])}")
                
                if torch.allclose(cls_current[0], cls_current[1]):
                    print(f"   🚨 CLASSIFICATION LAYER {i} IS MAKING OUTPUTS IDENTICAL!")
                    print(f"   Layer {i} weight std: {layer.weight.std():.8f}")
                    if layer.bias is not None:
                        print(f"   Layer {i} bias std: {layer.bias.std():.8f}")
                    return False
            elif hasattr(layer, 'forward'):  # Activation
                cls_current = layer(cls_current)
        
        # Debug regression head step by step
        print(f"   === REGRESSION HEAD ===")
        reg_current = shared_features
        for i, layer in enumerate(self.regression_head):
            if isinstance(layer, nn.Linear):
                reg_current = layer(reg_current)
                print(f"   Reg layer {i}: shape = {reg_current.shape}")
                
                # Handle different output sizes
                if reg_current.shape[1] == 1:  # Final layer (scalar output)
                    print(f"   Reg layer {i}: sample 0 = {reg_current[0].item():.8f}")
                    print(f"   Reg layer {i}: sample 1 = {reg_current[1].item():.8f}")
                else:  # Intermediate layer (vector output)
                    print(f"   Reg layer {i}: sample 0 = {reg_current[0, :3].tolist()}")
                    print(f"   Reg layer {i}: sample 1 = {reg_current[1, :3].tolist()}")
                
                print(f"   Reg layer {i} different: {not torch.allclose(reg_current[0], reg_current[1])}")
                
                if torch.allclose(reg_current[0], reg_current[1]):
                    print(f"   🚨 REGRESSION LAYER {i} IS MAKING OUTPUTS IDENTICAL!")
                    print(f"   Layer {i} weight std: {layer.weight.std():.8f}")
                    if layer.bias is not None:
                        print(f"   Layer {i} bias std: {layer.bias.std():.8f}")
                    return False
            elif hasattr(layer, 'forward'):  # Activation
                reg_current = layer(reg_current)
        
        return True

    def forward(self, x):
        """Standard forward pass for all architectures"""
    
        features = self._extract_features(x)
        shared_features = self.shared_layers(features)

        # # Debug task heads
        # if not self._debug_task_heads(shared_features):
        #     raise RuntimeError("Task heads are broken!")
        
        
        # Route to appropriate task heads
        if self.task == 'IntersectionStatus_IntersectionVolume':
            intersection_status_logits = self.classification_head(shared_features)
            regression_raw = self.regression_head(shared_features)
            regression_output = torch.relu(regression_raw)
            return torch.cat([intersection_status_logits, regression_output], dim=1)
        
        elif self.task == 'IntersectionStatus':
            return self.classification_head(shared_features)
        
        elif self.task == 'IntersectionVolume':
            return torch.relu(self.regression_head(shared_features))
        
        else:
            raise ValueError(f"Unknown task type: {self.task}")
    
    def predict(self, x):
        """Generate predictions with appropriate post-processing"""
        self.eval()
        with torch.no_grad():
            raw_output = self(x)
            
            if self.task == 'IntersectionStatus':
                return (raw_output > 0.5).int().squeeze() # model outputs logits
            
            elif self.task == 'IntersectionVolume':
                return raw_output.squeeze() / self.volume_scale_factor
            
            elif self.task == 'IntersectionStatus_IntersectionVolume':
                classification_pred = (raw_output[:, 0:1] > 0.5).int().squeeze() # model outputs logits
                regression_pred = raw_output[:, 1:2].squeeze() / self.volume_scale_factor
                return torch.stack([classification_pred.double(), regression_pred.double()], dim=1)
            
            else:
                raise ValueError(f"Unknown task for prediction: {self.task}")

# ============================================================================
# SPECIFIC ARCHITECTURES
# ============================================================================

class TetrahedronPairNet(BaseNet):
    """Specialized architecture for processing tetrahedron pairs"""
    
    def __init__(self, common_params, mlp_params, architecture_config):
        # Extract architecture-specific parameters
        self.per_vertex_layers = architecture_config.get("per_vertex_layers", [12, 12])
        self.per_tetrahedron_layers = architecture_config.get("per_tetrahedron_layers", [12, 12])
        self.per_two_tetrahedra_layers = architecture_config.get("per_two_tetrahedra_layers", [12, 12])
        
        self.vertex_aggregation_method = architecture_config.get("vertices_aggregation_function", "max")
        self.tetrahedron_aggregation_method = architecture_config.get("tetrahedra_aggregation_function", "max")
        
        # Initialize base class
        super().__init__(common_params, mlp_params)
        
        # Build vertex processing networks (8 vertices total)
        self._build_vertex_processors()
        
        # Build tetrahedron processing networks
        self._build_tetrahedron_processors()
        
        # Build final feature combination network
        self._build_feature_combiner()
    
    def _get_backbone_input_dim(self):
        """Input to shared backbone is the output of TetrahedronPairNet feature extraction"""
        return self.per_two_tetrahedra_layers[-1]
    
    def _build_vertex_processors(self):
        """Build 8 independent vertex processors (4 per tetrahedron) with residuals"""
        activation = ActivationRegistry.get_activation('relu')
        
        self.vertex_processors = nn.ModuleList([
            NetworkBuilder.build_mlp_layers([3] + self.per_vertex_layers, activation)
            for _ in range(8)
        ])
        
        # Residual connections for dimension matching at vertex level
        self.vertex_residuals = nn.ModuleList([
            nn.Linear(3, self.per_vertex_layers[-1])
            for _ in range(8)
        ])
    
    def _build_tetrahedron_processors(self):
        """Build networks that process aggregated vertex features for each tetrahedron"""
        self.tetrahedron_processor_1 = NetworkBuilder.build_mlp_layers(
            [self.per_vertex_layers[-1]] + self.per_tetrahedron_layers, self.activation
        )
        self.tetrahedron_processor_2 = NetworkBuilder.build_mlp_layers(
            [self.per_vertex_layers[-1]] + self.per_tetrahedron_layers, self.activation
        )
        
        # Residual: original tetrahedron coords [12] → tetrahedron output
        self.tetrahedron_residual_1 = nn.Linear(12, self.per_tetrahedron_layers[-1])
        self.tetrahedron_residual_2 = nn.Linear(12, self.per_tetrahedron_layers[-1])
    
    def _build_feature_combiner(self):
        """Build network that combines both tetrahedra features after aggregation"""
        self.feature_combiner = NetworkBuilder.build_mlp_layers(
            [self.per_tetrahedron_layers[-1]] + self.per_two_tetrahedra_layers, self.activation
        )
        
        # Global residual: raw input → final output (skip all processing)
        self.global_residual = nn.Linear(self.input_dim, self.per_two_tetrahedra_layers[-1])
    
    def _extract_features(self, x):
        """Extract features using TetrahedronPairNet architecture"""
        batch_size = x.size(0)
        input_dim = x.size(1)
        
        if input_dim not in [12, 24]:
            raise ValueError(f"Input dimension must be 12 or 24, got {input_dim}")
        
        # Process first tetrahedron (always present)
        tetrahedron_1_features = self._process_tetrahedron(x[:, :12], 0)
        
        if input_dim == 12:
            combined_features = tetrahedron_1_features
        else:
            # Two tetrahedra case
            tetrahedron_2_features = self._process_tetrahedron(x[:, 12:24], 4)
            
            # Aggregate tetrahedron features (DeepSets: permutation invariance)
            stacked_features = torch.stack([tetrahedron_1_features, tetrahedron_2_features], dim=1)
            aggregated_features = AggregationFunctions.aggregate_tensors(
                stacked_features, self.tetrahedron_aggregation_method
            )
            
            combined_features = self.feature_combiner(aggregated_features)
        
        # Global residual: raw input → final output
        return combined_features + self.global_residual(x)
    
    def _process_tetrahedron(self, tetrahedron_coords, vertex_offset):
        """
        Process a single tetrahedron: vertex MLPs → aggregation → tetrahedron MLP
        
        Args:
            tetrahedron_coords: [batch_size, 12] flattened vertex coordinates
            vertex_offset: 0 for first tetrahedron, 4 for second
        """
        batch_size = tetrahedron_coords.size(0)
        vertices = tetrahedron_coords.view(batch_size, 4, 3)
        
        # Process each vertex independently with residual connections
        vertex_features = []
        for i in range(4):
            vertex_idx = vertex_offset + i
            vertex_coords = vertices[:, i, :]
            
            processed = self.vertex_processors[vertex_idx](vertex_coords)
            residual = self.vertex_residuals[vertex_idx](vertex_coords)
            vertex_features.append(processed + residual)
        
        # Aggregate vertices (DeepSets: permutation invariance within tetrahedron)
        stacked_vertices = torch.stack(vertex_features, dim=1)
        aggregated_vertices = AggregationFunctions.aggregate_tensors(
            stacked_vertices, self.vertex_aggregation_method
        )
        
        # Process aggregated features + residual from original tetrahedron coords
        if vertex_offset == 0:
            return self.tetrahedron_processor_1(aggregated_vertices) + \
                   self.tetrahedron_residual_1(tetrahedron_coords)
        else:
            return self.tetrahedron_processor_2(aggregated_vertices) + \
                   self.tetrahedron_residual_2(tetrahedron_coords)

class SimpleMLP(BaseNet):
    """Simple Multi-Layer Perceptron baseline"""
    
    def __init__(self, common_params, mlp_params, architecture_config=None):
        super().__init__(common_params, mlp_params)
    
    def _get_backbone_input_dim(self):
        """For MLP, backbone input is the raw input dimension"""
        return self.input_dim
    
    def _extract_features(self, x):
        """For MLP, no feature extraction - pass through input directly"""
        return x

# ============================================================================
# ARCHITECTURE MANAGER
# ============================================================================

class CArchitectureManager:
    """Main manager for neural network architectures"""
    
    def __init__(self, config):
        self.config = config
        
        # Registry of available architectures
        self.architecture_registry = {
            'tetrahedronpairnet': TetrahedronPairNet,
            'mlp': SimpleMLP,
        }
    
    def get_model(self):
        """
        Create and return the configured model
        
        Returns:
            nn.Module: Configured neural network model
        """
        # Get architecture configuration
        architecture_name = self.config['architecture']['use_model'].lower()
        architecture_config = self.config['architecture'].get(architecture_name, {})
        
        # Validate architecture exists
        if architecture_name not in self.architecture_registry:
            available = list(self.architecture_registry.keys())
            raise ValueError(f"Unknown architecture '{architecture_name}'. Available: {available}")
        
        # Prepare common parameters
        common_params = self._build_common_parameters()
        
        # Prepare MLP parameters
        mlp_params = self._extract_mlp_parameters(architecture_config)
        
        # Create model
        model_class = self.architecture_registry[architecture_name]
        model = model_class(common_params, mlp_params, architecture_config)
        
        # Ensure float64 precision
        model = model.double()
        for param in model.parameters():
            if param.dtype != torch.float64:
                param.data = param.data.double()
        
            
        # Log model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Created {architecture_name} model:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Input dimension: {common_params['input_dim']}")
        print(f"  Task: {common_params['task']}")
        print(f"  Precision: float64")
        
        return model
    
    def _build_common_parameters(self):
        """Build common parameters used by all architectures"""
        input_dim = DatasetInspector.get_input_dimensions(
            self.config.get('processed_data_path')
        )
        
        return {
            'input_dim': input_dim,
            'activation': self.config['common_parameters']['activation_function'],
            'task': self.config['common_parameters']['task'],
            'volume_scale_factor': self.config['common_parameters']['volume_scale_factor'],
            'dropout_rate': self.config['common_parameters'].get('dropout_rate', 0.0)
        }
    
    def _extract_mlp_parameters(self, architecture_config):
        """Extract MLP-related parameters from architecture config"""
        return {
            'shared_layers': architecture_config.get('shared_layers', []),
            'classification_head': architecture_config.get('classification_head', []),
            'regression_head': architecture_config.get('regression_head', [])
        }
    
    def list_available_architectures(self):
        """Get list of available architectures"""
        return list(self.architecture_registry.keys())
    
    def get_architecture_info(self, architecture_name):
        """Get information about a specific architecture"""
        if architecture_name not in self.architecture_registry:
            return None
        
        model_class = self.architecture_registry[architecture_name]
        return {
            'name': architecture_name,
            'class': model_class.__name__,
            'docstring': model_class.__doc__
        }