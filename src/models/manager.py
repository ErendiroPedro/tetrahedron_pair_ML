import logging
import os
from typing import Dict, List

import pandas as pd

from src.models.networks import ArchitectureRegistry, BaseNet
from src.utils.file_management import processed_split_path

logger = logging.getLogger(__name__)

def infer_input_dim(data_path: str) -> int:
    """Infer input dimension from training data"""
    train_file = processed_split_path(data_path, "train")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found: {train_file}")
    
    try:
        # Read just header
        df = pd.read_csv(train_file, nrows=0)
        # Exclude last 2 columns (targets)
        feature_cols = df.columns[:-2]
        input_dim = len(feature_cols)
        logger.info(f"Detected {input_dim} input features from training data")
        return input_dim
    except Exception as e:
        raise ValueError(f"Error inspecting dataset: {e}")

class CArchitectureManager:
    """
    High-level manager for model creation within ML pipelines.
    Provides a stable interface for model instantiation and metadata access.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the architecture manager.
        
        Args:
            config: Configuration dictionary with structure:
                {
                    'architecture': {
                        'use_model': 'mlp',
                        'mlp': { ... },
                        'tetrahedronpairnet': { ... }
                    },
                    'common_parameters': {
                        'task': 'IntersectionStatus_IntersectionVolume',
                        'activation_function': 'relu',
                        'volume_scale_factor': 1000.0,
                        'dropout_rate': 0.1
                    },
                    'processed_data_path': '/path/to/data'
                }
        """
        self.config = config
        self._model_config = None  # Cached model config
        self._input_dim = None     # Cached input dimension
    
    def get_model(self) -> BaseNet:
        """
        Create and return a new model instance.
        
        Returns:
            Fresh model instance (randomly initialized)
        """
        model_config = self._get_model_config()
        arch_name = self.config['architecture']['use_model'].lower()
        
        # Create model
        model_class = ArchitectureRegistry.get(arch_name)
        model = model_class(model_config)
        
        # Log creation
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Created {arch_name}: {total_params:,} params "
            f"({trainable_params:,} trainable), "
            f"input_dim={model_config['input_dim']}, "
            f"task={model_config['task']}"
        )
        
        return model
    
    def _get_model_config(self) -> Dict:
        """
        Build the unified model configuration.
        Caches the result to avoid repeated file I/O.
        """
        if self._model_config is not None:
            return self._model_config
        
        # Extract architecture config
        arch_config = self.config['architecture']
        arch_name = arch_config['use_model'].lower()
        arch_specific_config = arch_config.get(arch_name, {})
        
        # Build unified config
        common_params = self.config['common_parameters']
        self._model_config = {
            'input_dim': self.get_input_dim(),
            'task': common_params['task'],
            'activation': common_params['activation_function'],
            'volume_scale_factor': common_params['volume_scale_factor'],
            'dropout_rate': common_params.get('dropout_rate', 0.0),
            **arch_specific_config  # Merge architecture-specific params
        }
        
        return self._model_config
    
    def get_input_dim(self) -> int:
        """Get the input dimension (cached after first call)"""
        if self._input_dim is None:
            self._input_dim = infer_input_dim(self.config['processed_data_path'])
        return self._input_dim
    
    def get_architecture_name(self) -> str:
        """Get the name of the configured architecture"""
        return self.config['architecture']['use_model'].lower()
    
    def get_task(self) -> str:
        """Get the task type"""
        return self.config['common_parameters']['task']
    
    def list_available_architectures(self) -> List[str]:
        """List all available architectures"""
        return ArchitectureRegistry.list_available()
    
    def get_model_info(self) -> Dict:
        """
        Get metadata about the configured model.
        Useful for logging and validation without instantiating the model.
        """
        model_config = self._get_model_config()
        arch_name = self.get_architecture_name()
        
        return {
            'architecture': arch_name,
            'task': model_config['task'],
            'input_dim': model_config['input_dim'],
            'activation': model_config['activation'],
            'volume_scale_factor': model_config['volume_scale_factor'],
            'dropout_rate': model_config['dropout_rate']
        }


# ============================================================================
# STANDALONE FACTORY (for direct use)
# ============================================================================

def create_model(config: Dict) -> BaseNet:
    """
    Standalone factory function for creating models.
    For pipeline use, prefer ArchitectureManager.
    
    Args:
        config: Configuration dictionary (same structure as ArchitectureManager)
    
    Returns:
        Instantiated model (float64 precision)
    """
    manager = CArchitectureManager(config)
    return manager.get_model()
