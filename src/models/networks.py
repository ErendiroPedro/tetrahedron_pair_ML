import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
logger = logging.getLogger(__name__)

# ============================================================================
# ACTIVATION REGISTRY
# ============================================================================

class ActivationRegistry:
    """
    Centralized registry for activation functions.
    Supports both built-in and custom activations.
    """
    
    _registry: Dict[str, Callable[[], nn.Module]] = {
        'relu': lambda: nn.ReLU(),
        'tanh': lambda: nn.Tanh(),
        'leakyrelu': lambda: nn.LeakyReLU(),
        'elu': lambda: nn.ELU(),
        'gelu': lambda: nn.GELU(),
        'swish': lambda: nn.SiLU(),
        'mish': lambda: nn.Mish()
    }
    
    @classmethod
    def register(cls, name: str, factory: Callable[[], nn.Module]):
        """Register a custom activation function"""
        cls._registry[name.lower()] = factory
    
    @classmethod
    def get(cls, name: str) -> nn.Module:
        """Get activation instance by name"""
        factory = cls._registry.get(name.lower())
        if not factory:
            available = ', '.join(cls._registry.keys())
            raise ValueError(f"Unknown activation '{name}'. Available: {available}")
        return factory()

# ============================================================================
# AGGREGATION UTILITIES
# ============================================================================

def aggregate_features(features: torch.Tensor, method: str = 'max') -> torch.Tensor:
    """
    Aggregate features along dimension 1.
    
    Args:
        features: [batch_size, num_items, feature_dim]
        method: 'max', 'mean', 'sum', 'min', or 'hadamard_prod'
    
    Returns:
        Aggregated tensor [batch_size, feature_dim]
    """
    aggregators = {
        'max': lambda x: torch.max(x, dim=1)[0],
        'mean': lambda x: torch.mean(x, dim=1),
        'sum': lambda x: torch.sum(x, dim=1),
        'min': lambda x: torch.min(x, dim=1)[0],
        'hadamard_prod': lambda x: torch.prod(F.relu(x), dim=1)
    }
    
    if method not in aggregators:
        available = ', '.join(aggregators.keys())
        raise ValueError(f"Unknown aggregation '{method}'. Available: {available}")
    
    return aggregators[method](features)

# ============================================================================
# NETWORK BUILDING COMPONENTS
# ============================================================================

class NetworkBuilder:
    """Factory for constructing common network components"""
    
    @staticmethod
    def build_mlp(
        dims: List[int],
        activation: nn.Module,
        dropout: float = 0.0,
        final_activation: bool = False
    ) -> nn.Sequential:
        """
        Build a simple MLP.
        
        Args:
            dims: [input_dim, hidden1, ..., output_dim]
            activation: Activation function instance
            dropout: Dropout rate (applied to hidden layers only)
            final_activation: Whether to apply activation after final layer
        """
        if len(dims) < 2:
            raise ValueError("Need at least input and output dimensions")
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            is_final = (i == len(dims) - 2)
            if not is_final or final_activation:
                layers.append(activation)
            if dropout > 0.0 and not is_final:
                layers.append(nn.Dropout(dropout))
          
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def build_task_head(
        input_dim: int,
        hidden_dims: List[int],
        activation: nn.Module
    ) -> nn.Sequential:
        """
        Build a task head (always outputs single value).
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions (output dimension added automatically)
            activation: Activation function
        """
        if not hidden_dims:
            return nn.Sequential(nn.Linear(input_dim, 1))
        
        # Add output dimension if not present
        if hidden_dims[-1] != 1:
            hidden_dims = hidden_dims + [1]
        
        dims = [input_dim] + hidden_dims
        return NetworkBuilder.build_mlp(dims, activation, dropout=0.0)

# ============================================================================
# BASE ARCHITECTURE
# ============================================================================

class BaseNet(nn.Module, ABC):
    """
    Base class for all neural network architectures.
    
    Subclasses must implement:
        - _build_feature_extractor(): Build feature extraction layers
        - _get_feature_dim(): Return feature extractor output dimension
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Dictionary containing model and architecture parameters.
        """
        super().__init__()
        self.config = config
        self.task = config['task']
        self.volume_scale_factor = config['volume_scale_factor']
        self.input_dim = config['input_dim']
        self.activation = ActivationRegistry.get(config['activation'])
        self.dropout = config.get('dropout_rate', 0.0)
        # Build feature extractor (architecture-specific)
        self.feature_extractor = self._build_feature_extractor(config)
        feature_dim = self._get_feature_dim(config)
        # Build task-specific heads
        self.classification_head = NetworkBuilder.build_task_head(
            feature_dim,
            config.get('classification_head', []),
            self.activation
        )
        self.regression_head = NetworkBuilder.build_task_head(
            feature_dim,
            config.get('regression_head', []),
            self.activation
        )
        self.to(torch.float64)
        self._verify_dtype()
    
    def _verify_dtype(self):
        """Verify all parameters are float64"""
        for name, param in self.named_parameters():
            if param.dtype != torch.float64:
                logger.warning(f"Parameter {name} is {param.dtype}, converting to float64")
                param.data = param.data.double()
    
    @abstractmethod
    def _build_feature_extractor(self, config: Dict) -> nn.Module:
        """
        Build the feature extraction network.
        Args:
            config: Full configuration dictionary
        Returns:
            Feature extraction module (can be nn.Sequential, nn.ModuleDict, etc.)
        """
        pass
    
    @abstractmethod
    def _get_feature_dim(self, config: Dict) -> int:
        """
        Get the output dimension of the feature extractor.
        
        Args:
            config: Full configuration dictionary
        
        Returns:
            Feature dimension
        """
        pass
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input.
        Default implementation assumes feature_extractor is callable.
        Override if you need custom logic.
        """
        return self.feature_extractor(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - handles task routing"""
        features = self.extract_features(x)
        
        if self.task == 'IntersectionStatus':
            return self.classification_head(features)
        
        elif self.task == 'IntersectionVolume':
            volume = self.regression_head(features)
            return F.relu(volume)  # Ensure positive
        
        elif self.task == 'IntersectionStatus_IntersectionVolume':
            has_intersection = self.classification_head(features)
            volume = F.relu(self.regression_head(features))
            return torch.cat([has_intersection, volume], dim=1)
        
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Generate predictions with post-processing"""
        self.eval()
        with torch.no_grad():
            output = self(x)
            
            if self.task == 'IntersectionStatus':
                return (output > 0).int().squeeze()
            
            elif self.task == 'IntersectionVolume':
                return output.squeeze() / self.volume_scale_factor
            
            elif self.task == 'IntersectionStatus_IntersectionVolume':
                has_intersection = (output[:, 0] > 0).int()
                volume = output[:, 1] / self.volume_scale_factor
                return torch.stack([has_intersection.double(), volume], dim=1)

# ============================================================================
# CONCRETE ARCHITECTURES
# ============================================================================

class SimpleMLP(BaseNet):
    """Simple feedforward baseline"""
    
    def _build_feature_extractor(self, config: Dict) -> nn.Module:
        shared_layers = config.get('shared_layers', [])
        if not shared_layers:
            return nn.Identity()
        dims = [config['input_dim']] + shared_layers
        return NetworkBuilder.build_mlp(dims, self.activation, self.dropout)
    
    def _get_feature_dim(self, config: Dict) -> int:
        shared_layers = config.get('shared_layers', [])
        return shared_layers[-1] if shared_layers else config['input_dim']


class TetrahedronPairNet(BaseNet):
    """
    Hierarchical architecture for tetrahedron pair processing.
    Implements permutation invariance via DeepSets-style aggregation.
    """
    
    def _build_feature_extractor(self, config: Dict) -> nn.Module:
        # Extract architecture-specific params
        vertex_layers = config['per_vertex_layers']
        tet_layers = config['per_tetrahedron_layers']
        pair_layers = config['per_two_tetrahedra_layers']

        # Store aggregation methods
        self.vertex_agg = config['vertices_aggregation_function']
        self.tet_agg = config['tetrahedra_aggregation_function']

        # Store dimensions
        self.vertex_dim = vertex_layers[-1]
        self.tet_dim = tet_layers[-1]
        self.pair_dim = pair_layers[-1]

        # Build vertex processors (8 = 4 vertices × 2 tetrahedra)
        self.vertex_net = NetworkBuilder.build_mlp([3] + vertex_layers, nn.ReLU(), self.dropout)

        self.vertex_shortcuts = nn.ModuleList([
            nn.Linear(3, self.vertex_dim) for _ in range(8)
        ])

        # Build tetrahedron processors
        self.tet_net = NetworkBuilder.build_mlp([self.vertex_dim] + tet_layers, self.activation, self.dropout)
        self.tet_shortcut_1 = nn.Linear(12, self.tet_dim)
        self.tet_shortcut_2 = nn.Linear(12, self.tet_dim)
        # Build pair processor
        pair_net = NetworkBuilder.build_mlp([self.tet_dim] + pair_layers, self.activation, self.dropout)
        global_shortcut = nn.Linear(config['input_dim'], self.pair_dim)
        # Return as ModuleDict for clarity
        return nn.ModuleDict({
            'pair_net': pair_net,
            'global_shortcut': global_shortcut
        })
    
    def _get_feature_dim(self, config: Dict) -> int:
        pair_layers = config.get('per_two_tetrahedra_layers', [12, 12])
        return pair_layers[-1]
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Hierarchical feature extraction with residual connections"""
        if x.size(1) not in [12, 24]:
            raise ValueError(f"Input must be 12 or 24 dims, got {x.size(1)}")
        
        # Process first tetrahedron
        tet1_features = self._process_tetrahedron(x[:, :12], tet_idx=0)
        
        if x.size(1) == 12:
            # Single tetrahedron
            combined = self.feature_extractor['pair_net'](tet1_features)
        else:
            # Two tetrahedra - aggregate then process
            tet2_features = self._process_tetrahedron(x[:, 12:24], tet_idx=1)
            stacked = torch.stack([tet1_features, tet2_features], dim=1)
            aggregated = aggregate_features(stacked, self.tet_agg)
            combined = self.feature_extractor['pair_net'](aggregated)
        
        # Global residual connection
        return combined + self.feature_extractor['global_shortcut'](x)
    
    def _process_tetrahedron(self, coords: torch.Tensor, tet_idx: int) -> torch.Tensor:
        """
        Process a single tetrahedron.
        
        Args:
            coords: [batch, 12] - flattened (x,y,z) for 4 vertices
            tet_idx: 0 or 1 (which tetrahedron)
        """
        batch_size = coords.size(0)
        vertices = coords.view(batch_size, 4, 3)
        
        # Process each vertex with residual
        vertex_features = []
        offset = tet_idx * 4
        for i in range(4):
            v = vertices[:, i, :]
            processed = self.vertex_net(v)
            shortcut = self.vertex_shortcuts[offset + i](v)
            vertex_features.append(processed + shortcut)
        
        # Aggregate vertices (permutation invariance)
        stacked = torch.stack(vertex_features, dim=1)
        aggregated = aggregate_features(stacked, self.vertex_agg)
        
        if tet_idx == 0:
            return self.tet_net(aggregated) + self.tet_shortcut_1(coords)
        else:
            return self.tet_net(aggregated) + self.tet_shortcut_2(coords)


# ============================================================================
# ARCHITECTURE REGISTRY
# ============================================================================

class ArchitectureRegistry:
    """
    Central registry for architecture classes.
    Supports registration via decorator for easy extensibility.
    """
    
    _registry: Dict[str, Type[BaseNet]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a new architecture.
        
        Usage:
            @ArchitectureRegistry.register('my_model')
            class MyModel(BaseNet):
                ...
        """
        def decorator(model_class: Type[BaseNet]):
            cls._registry[name.lower()] = model_class
            logger.debug(f"Registered architecture: {name}")
            return model_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseNet]:
        """Get architecture class by name"""
        model_class = cls._registry.get(name.lower())
        if not model_class:
            available = ', '.join(cls._registry.keys())
            raise ValueError(f"Unknown architecture '{name}'. Available: {available}")
        return model_class
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered architectures"""
        return list(cls._registry.keys())


# Register built-in architectures
ArchitectureRegistry.register('mlp')(SimpleMLP)
ArchitectureRegistry.register('tetrahedronpairnet')(TetrahedronPairNet)

# ============================================================================
# EXAMPLE: Adding a custom architecture
# ============================================================================

# Researchers can add new models without modifying core code:
#
# @ArchitectureRegistry.register('graph_net')
# class GraphNet(BaseNet):
#     def _build_feature_extractor(self, config):
#         # Custom implementation
#         pass
#     
#     def _get_feature_dim(self, config):
#         return config.get('graph_output_dim', 64)
