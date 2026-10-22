import datetime
import hashlib
import json
import os
import platform
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from src.artifacts.handlers import (
    DictHandler,
    MetricsHandler,
    ModelHandler,
    ReportHandler,
    VisualizationHandler,
)
from src.artifacts.types import (
    ArtifactHandler,
    ArtifactMetadata,
    ArtifactType,
    ExperimentContext,
    StorageLevel,
)


class SystemContextCapture:
    """Capture system and experiment context automatically"""
    
    @staticmethod
    def capture_system_info() -> Dict[str, Any]:
        """Capture comprehensive system information"""
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'memory_gb': SystemContextCapture._get_memory_info(),
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        
        return info
    
    @staticmethod
    def _get_memory_info() -> Optional[float]:
        """Get system memory in GB"""
        try:
            import psutil
            return round(psutil.virtual_memory().total / (1024**3), 2)
        except ImportError:
            return None
    
    @staticmethod
    def capture_git_info() -> Optional[str]:
        """Capture git commit hash"""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    @staticmethod
    def compute_config_hash(config: Dict[str, Any]) -> str:
        """Compute hash of configuration"""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

# ============================================================================
# 4. MAIN ARTIFACTS MANAGER
# ============================================================================

class CArtifactsManager:
    """
    Enhanced artifacts manager implementing hierarchical organization,
    smart categorization, and extensible architecture.
    """
    
    def __init__(self, config: Dict[str, Any], storage_level: StorageLevel = StorageLevel.STANDARD):
        self.config = config
        self.storage_level = storage_level
        
        # Initialize context
        self.context = self._create_experiment_context()
        
        # Create experiment directory
        self.artifacts_path = self._create_experiment_folder()
        
        # Initialize handlers
        self.handlers = self._initialize_handlers()
        
        # Save initial context
        self._save_experiment_context()
    
    def _create_experiment_context(self) -> ExperimentContext:
        """Create comprehensive experiment context"""
        timestamp = datetime.datetime.now()
        
        return ExperimentContext(
            experiment_id=timestamp.strftime("%m%d_%H%M%S"),
            timestamp=timestamp,
            git_commit=SystemContextCapture.capture_git_info(),
            system_info=SystemContextCapture.capture_system_info(),
            data_fingerprint=None,  # Could be computed from dataset
            config_hash=SystemContextCapture.compute_config_hash(self.config)
        )
    
    def _create_experiment_folder(self) -> Path:
        """Create hierarchical experiment folder structure"""
        # Extract config details for naming
        model_arch = self.config.get('model_config', {}).get('architecture', {}).get('use_model', 'unknown')
        task = self.config.get('model_config', {}).get('common_parameters', {}).get('task', 'unknown')
        train_samples = self.config.get('processor_config', {}).get('num_train_samples', 0)
        val_samples = self.config.get('processor_config', {}).get('num_val_samples', 0)
        epochs = self.config.get('trainer_config', {}).get('epochs', 0)
        
        # Volume range for specialized experiments
        volume_config = self.config.get('processor_config', {}).get('volume_binning', {})
        volume_range = volume_config.get('volume_range', [])
        vol_str = f"vol{'-'.join(map(str, volume_range))}" if volume_range else "no-vol-filter"
        
        # Create descriptive folder name
        folder_name = (
            f"id{self.context.experiment_id}_{model_arch}_{task}_"
            f"{train_samples//1000}k-train_{val_samples//1000}k-val_"
            f"{vol_str}_e{epochs}"
        )
        
        # Create full path
        base_path = Path(self.config.get('artifacts_config', {}).get('save_artifacts_to', './artifacts'))
        full_path = base_path / folder_name
        
        # Create directory structure
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            "metadata", "configuration", "model_artifacts", 
            "training_artifacts", "evaluation_artifacts", 
            "visualization_artifacts", "logs", "derivatives"
        ]
        
        for subdir in subdirs:
            (full_path / subdir).mkdir(exist_ok=True)
        
        return full_path
    
    def _initialize_handlers(self) -> List[ArtifactHandler]:
        """Initialize artifact handlers"""
        return [
            ModelHandler(self.config),
            VisualizationHandler(),
            MetricsHandler(),
            ReportHandler(),
            DictHandler()
        ]
    
    def _save_experiment_context(self):
        """Save experiment context and configuration"""
        # Save context metadata
        context_path = self.artifacts_path / "metadata" / "experiment_context.json"
        with open(context_path, 'w') as f:
            json.dump(asdict(self.context), f, indent=2, default=str)
        
        # Save configuration
        config_path = self.artifacts_path / "configuration" / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Save system snapshot
        system_path = self.artifacts_path / "metadata" / "system_info.json"
        with open(system_path, 'w') as f:
            json.dump(self.context.system_info, f, indent=2)
    
    def save_artifacts(self, *args, **kwargs):
        """
        Enhanced artifact saving with smart categorization and multiple formats.
        
        Args:
            *args: Positional artifacts
            **kwargs: Named artifacts for better organization
        """
        print(f"\n=== Saving Artifacts to {self.artifacts_path.name} ===")
        
        saved_artifacts = []
        
        # Handle named artifacts first (better organization)
        for name, artifact in kwargs.items():
            try:
                metadata = self._create_artifact_metadata(name, artifact)
                result = self._save_single_artifact(artifact, metadata)
                if result:
                    saved_artifacts.append({
                        'name': name,
                        'metadata': metadata,
                        'results': result
                    })
            except Exception as e:
                print(f"Error saving named artifact '{name}': {e}")
                traceback.print_exc()
        
        # Handle positional artifacts
        for index, artifact in enumerate(args):
            try:
                name = f"artifact_{index}"
                metadata = self._create_artifact_metadata(name, artifact)
                result = self._save_single_artifact(artifact, metadata)
                if result:
                    saved_artifacts.append({
                        'name': name,
                        'metadata': metadata,
                        'results': result
                    })
            except Exception as e:
                print(f"Error saving positional artifact {index}: {e}")
                traceback.print_exc()
        
        # Save artifacts manifest
        self._save_artifacts_manifest(saved_artifacts)
        
        print(f"=== Saved {len(saved_artifacts)} artifacts successfully ===")
        print(f"Experiment folder: {self.artifacts_path}")
        
        return saved_artifacts
    
    def _create_artifact_metadata(self, name: str, artifact: Any) -> ArtifactMetadata:
        """Create metadata for an artifact"""
        # Determine artifact type
        handler = self._find_handler(artifact)
        artifact_type = handler.get_artifact_type() if handler else ArtifactType.DATA
        
        # Compute size and checksum
        size_bytes = self._estimate_size(artifact)
        checksum = self._compute_checksum(artifact)
        
        # Determine format
        format_name = self._determine_format(artifact)
        
        # Create tags based on content
        tags = self._generate_tags(name, artifact, artifact_type)
        
        return ArtifactMetadata(
            name=name,
            type=artifact_type,
            format=format_name,
            size_bytes=size_bytes,
            checksum=checksum,
            created_at=datetime.datetime.now(),
            tags=tags,
            description=self._generate_description(name, artifact_type)
        )
    
    def _find_handler(self, artifact: Any) -> Optional[ArtifactHandler]:
        """Find appropriate handler for artifact"""
        for handler in self.handlers:
            if handler.can_handle(artifact):
                return handler
        return None
    
    def _save_single_artifact(self, artifact: Any, metadata: ArtifactMetadata) -> Optional[Dict[str, Any]]:
        """Save a single artifact using appropriate handler"""
        handler = self._find_handler(artifact)
        
        if handler:
            print(f"Saving {metadata.type.value}: {metadata.name}")
            result = handler.save(artifact, self.artifacts_path, metadata)
            
            # Save metadata
            metadata_path = self.artifacts_path / "metadata" / f"{metadata.name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
            
            return result
        else:
            print(f"Warning: No handler found for artifact '{metadata.name}' of type {type(artifact)}")
            return None
    
    def _estimate_size(self, artifact: Any) -> int:
        """Estimate artifact size in bytes"""
        try:
            if isinstance(artifact, torch.nn.Module):
                return sum(p.numel() * p.element_size() for p in artifact.parameters())
            elif isinstance(artifact, str):
                return len(artifact.encode('utf-8'))
            elif isinstance(artifact, dict):
                return len(json.dumps(artifact, default=str).encode('utf-8'))
            else:
                return 0
        except Exception:
            return 0
    
    def _compute_checksum(self, artifact: Any) -> str:
        """Compute checksum for artifact"""
        try:
            if isinstance(artifact, torch.nn.Module):
                # Use model state dict for checksum
                state_dict_str = str(sorted(artifact.state_dict().items()))
                return hashlib.md5(state_dict_str.encode()).hexdigest()[:8]
            elif isinstance(artifact, str):
                return hashlib.md5(artifact.encode()).hexdigest()[:8]
            elif isinstance(artifact, dict):
                dict_str = json.dumps(artifact, sort_keys=True, default=str)
                return hashlib.md5(dict_str.encode()).hexdigest()[:8]
            else:
                return hashlib.md5(str(artifact).encode()).hexdigest()[:8]
        except Exception:
            return "unknown"
    
    def _determine_format(self, artifact: Any) -> str:
        """Determine artifact format"""
        if isinstance(artifact, torch.nn.Module):
            return "pytorch_module"
        elif isinstance(artifact, str):
            if len(artifact) > 1000 and any(artifact.startswith(prefix) for prefix in ['data:image/', 'iVBORw0KGgo', '/9j/', 'R0lGOD']):
                return "base64_image"
            else:
                return "text"
        elif isinstance(artifact, dict):
            if "image" in artifact:
                return "image_data"
            else:
                return "json"
        else:
            return "unknown"
    
    def _generate_tags(self, name: str, artifact: Any, artifact_type: ArtifactType) -> List[str]:
        """Generate tags based on content analysis"""
        tags = [artifact_type.value]
        
        name_lower = name.lower()
        
        # Content-specific tags
        if 'loss' in name_lower:
            tags.extend(['loss', 'training'])
        if 'best' in name_lower or 'optimal' in name_lower:
            tags.append('best_model')
        if 'final' in name_lower:
            tags.append('final')
        if 'evaluation' in name_lower or 'eval' in name_lower:
            tags.append('evaluation')
        if 'distribution' in name_lower:
            tags.append('distribution')
        
        # Storage level tag
        tags.append(self.storage_level.value)
        
        return tags
    
    def _generate_description(self, name: str, artifact_type: ArtifactType) -> str:
        """Generate description based on name and type"""
        descriptions = {
            ArtifactType.MODEL: f"Trained model artifact: {name}",
            ArtifactType.METRICS: f"Training metrics and statistics: {name}",
            ArtifactType.VISUALIZATION: f"Visualization plot: {name}",
            ArtifactType.REPORT: f"Text report: {name}",
            ArtifactType.CONFIG: f"Configuration file: {name}",
            ArtifactType.LOG: f"Log file: {name}",
            ArtifactType.DATA: f"Data artifact: {name}"
        }
        
        return descriptions.get(artifact_type, f"Artifact: {name}")
    
    def _save_artifacts_manifest(self, saved_artifacts: List[Dict[str, Any]]):
        """Save manifest of all saved artifacts"""
        manifest = {
            'experiment_id': self.context.experiment_id,
            'created_at': datetime.datetime.now().isoformat(),
            'storage_level': self.storage_level.value,
            'total_artifacts': len(saved_artifacts),
            'artifacts': saved_artifacts
        }
        
        manifest_path = self.artifacts_path / "metadata" / "artifacts_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of current experiment"""
        return {
            'experiment_id': self.context.experiment_id,
            'path': str(self.artifacts_path),
            'timestamp': self.context.timestamp.isoformat(),
            'config_hash': self.context.config_hash,
            'git_commit': self.context.git_commit,
            'system_info': self.context.system_info
        }
    
    # Add this simple method to CArtifactsManager class:

    def run_data_analysis(self, notebook_path: str = None) -> Dict[str, Any]:
        """
        Execute data analysis notebook and save HTML report only
        
        Args:
            notebook_path: Path to notebook file. If None, uses default data_analysis.ipynb
        """
        if notebook_path is None:
            notebook_path = "notebooks/data_analysis.ipynb"
        
        # Convert to absolute path if needed
        if not os.path.isabs(notebook_path):
            home_path = self.config.get('home', '')
            if home_path:
                notebook_path = os.path.join(home_path, notebook_path)
            else:
                notebook_path = os.path.abspath(notebook_path)
        
        print(f"\n=== Running Data Analysis ===")
        print(f"Notebook: {notebook_path}")
        
        if not os.path.exists(notebook_path):
            error_msg = f"Notebook not found: {notebook_path}"
            print(f"❌ {error_msg}")
            return {'error': error_msg}
        
        try:
            # Generate HTML report directly
            results = self.save_artifacts(data_analysis_notebook=notebook_path)
            
            if results.get('data_analysis_notebook', {}).get('status') == 'success':
                html_path = results['data_analysis_notebook']['html_report']
                print(f"✅ Data analysis complete! Report: {html_path}")
                return {'html_report': html_path, 'status': 'success'}
            else:
                error = results.get('data_analysis_notebook', {}).get('error', 'Unknown error')
                print(f"❌ Data analysis failed: {error}")
                return {'error': error}
                
        except Exception as e:
            error_msg = f"Failed to run data analysis: {e}"
            print(f"❌ {error_msg}")
            return {'error': error_msg}
