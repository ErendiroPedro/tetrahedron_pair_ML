import os
import base64
import torch
import shutil
import hashlib
import platform
import subprocess
from io import BytesIO
from PIL import Image
import datetime
import json
import yaml
import traceback 
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum

# ============================================================================
# 1. CORE DATA STRUCTURES
# ============================================================================

class ArtifactType(Enum):
    MODEL = "model"
    METRICS = "metrics"
    VISUALIZATION = "visualization"
    REPORT = "report"
    CONFIG = "config"
    LOG = "log"
    DATA = "data"

class StorageLevel(Enum):
    MINIMAL = "minimal"      # Essential files only
    STANDARD = "standard"    # Balanced save
    COMPLETE = "complete"    # Everything including debug info

@dataclass
class ArtifactMetadata:
    name: str
    type: ArtifactType
    format: str
    size_bytes: int
    checksum: str
    created_at: datetime.datetime
    tags: List[str]
    description: Optional[str] = None
    relationships: Optional[Dict[str, str]] = None

@dataclass
class ExperimentContext:
    experiment_id: str
    timestamp: datetime.datetime
    git_commit: Optional[str]
    system_info: Dict[str, Any]
    data_fingerprint: Optional[str]
    config_hash: str

# ============================================================================
# 2. ARTIFACT HANDLER SYSTEM
# ============================================================================

class ArtifactHandler(ABC):
    """Base class for handling specific artifact types"""
    
    @abstractmethod
    def can_handle(self, artifact: Any) -> bool:
        pass
    
    @abstractmethod
    def save(self, artifact: Any, path: Path, metadata: ArtifactMetadata) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_artifact_type(self) -> ArtifactType:
        pass

class ModelHandler(ArtifactHandler):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def can_handle(self, artifact: Any) -> bool:
        return isinstance(artifact, torch.nn.Module)
    
    def get_artifact_type(self) -> ArtifactType:
        return ArtifactType.MODEL
    
    def save(self, artifact: torch.nn.Module, path: Path, metadata: ArtifactMetadata) -> Dict[str, Any]:
        """Save model in multiple formats with optimization"""
        model_dir = path / "model_artifacts"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        try:
            # 1. State Dict (always save)
            state_dict_path = model_dir / "model_state_dict.pth"
            torch.save({
                'model_state_dict': artifact.state_dict(),
                'model_class': artifact.__class__.__name__,
                'architecture': self._extract_architecture_info(artifact)
            }, state_dict_path)
            results['state_dict'] = str(state_dict_path)
            
            # 2. Weights Export (for C++ deployment)
            weights_path = model_dir / "model_weights.json"
            weights = self._extract_weights(artifact)
            with open(weights_path, 'w') as f:
                json.dump(weights, f, indent=2)
            results['weights'] = str(weights_path)
            
            # 3. Optimized Model (TorchScript + torch.compile)
            optimized_path = self._save_optimized_model(artifact, model_dir)
            if optimized_path:
                results['optimized'] = str(optimized_path)
            
            # 4. Model Summary
            summary_path = model_dir / "model_summary.txt"
            self._save_model_summary(artifact, summary_path)
            results['summary'] = str(summary_path)
            
        except Exception as e:
            print(f"Error saving model: {e}")
            traceback.print_exc()
        
        return results
    
    def _extract_architecture_info(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Extract model architecture information"""
        info = {
            'class_name': model.__class__.__name__,
            'input_dim': getattr(model, 'input_dim', None),
            'task': getattr(model, 'task', None),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # Extract custom attributes
        for attr in ['common_params', 'mlp_params', 'volume_scale_factor']:
            if hasattr(model, attr):
                info[attr] = str(getattr(model, attr))
        
        return info
    
    def _extract_weights(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Extract model weights in structured format"""
        weights = {}
        
        # Model-specific weight extraction
        if 'TetrahedronPairNet' in str(model.__class__):
            weights = self._extract_tpn_weights(model)
        elif 'MLP' in str(model.__class__):
            weights = self._extract_mlp_weights(model)
        else:
            weights = self._extract_generic_weights(model)
        
        return weights
    
    def _extract_tpn_weights(self, model) -> Dict[str, Any]:
        """Extract TetrahedronPairNet specific weights"""
        weights = {}
        
        # Vertex processors
        for i in range(8):
            if hasattr(model, 'vertex_mlps') and len(model.vertex_mlps) > i:
                weights[f'vertex_mlp_{i}'] = self._extract_sequential_weights(model.vertex_mlps[i])
            
            if hasattr(model, 'vertex_residual_layers') and len(model.vertex_residual_layers) > i:
                weights[f'vertex_residual_{i}'] = self._extract_linear_weights(model.vertex_residual_layers[i])
        
        # Post-pooling processors
        for component in ['process_tet_post_pooling_1', 'process_tet_post_pooling_2', 
                         'process_combined_embd', 'global_residual', 'shared_layers',
                         'classifier_head', 'regressor_head']:
            if hasattr(model, component):
                attr = getattr(model, component)
                if hasattr(attr, 'weight'):
                    weights[component] = self._extract_linear_weights(attr)
                else:
                    weights[component] = self._extract_sequential_weights(attr)
        
        return weights
    
    def _extract_mlp_weights(self, model) -> Dict[str, Any]:
        """Extract MLP specific weights"""
        weights = {}
        for component in ['shared_layers', 'classifier_head', 'regressor_head']:
            if hasattr(model, component):
                weights[component] = self._extract_sequential_weights(getattr(model, component))
        return weights
    
    def _extract_generic_weights(self, model) -> Dict[str, Any]:
        """Generic weight extraction"""
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = param.detach().cpu().numpy().tolist()
        return weights
    
    def _extract_sequential_weights(self, module) -> List[Dict[str, List]]:
        """Extract weights from Sequential module"""
        layer_weights = []
        for name, layer in module.named_children():
            if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
                layer_weights.append(self._extract_linear_weights(layer))
        return layer_weights
    
    def _extract_linear_weights(self, module) -> Dict[str, List]:
        """Extract weights from Linear layer"""
        weight_data = module.weight.detach().cpu().numpy().tolist()
        bias_data = module.bias.detach().cpu().numpy().tolist() if module.bias is not None else []
        return {'weight': weight_data, 'bias': bias_data}
    
    def _save_optimized_model(self, model: torch.nn.Module, model_dir: Path) -> Optional[str]:
        """Save optimized model using torch.compile and TorchScript"""
        try:
            model.eval()
            
            # Get optimization settings
            optimization_config = self.config.get('evaluator_config', {})
            compile_mode = optimization_config.get('optimization_mode', 'default')
            batch_size = optimization_config.get('batch_size', 32)
            
            input_dim = getattr(model, 'input_dim', 24)
            device = next(model.parameters()).device
            sample_input = torch.randn(batch_size, input_dim, device=device, dtype=torch.float64)
            
            # Compile model
            compiled_model = torch.compile(model, mode=compile_mode, fullgraph=True, dynamic=False)
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = compiled_model(sample_input)
            
            # Create deployment wrapper
            class DeploymentWrapper(torch.nn.Module):
                def __init__(self, compiled_model):
                    super().__init__()
                    self.compiled_model = compiled_model
                
                def forward(self, x):
                    return self.compiled_model(x)
            
            wrapper = DeploymentWrapper(compiled_model)
            wrapper.eval()
            
            # Trace and save
            traced_model = torch.jit.trace(wrapper, sample_input, strict=False)
            optimized_path = model_dir / "model_optimized.pt"
            traced_model.save(optimized_path)
            
            # Also save as main model.pt
            main_model_path = model_dir / "model.pt"
            traced_model.save(main_model_path)
            
            return str(optimized_path)
            
        except Exception as e:
            print(f"Model optimization failed: {e}")
            # Fallback to regular TorchScript
            try:
                input_dim = getattr(model, 'input_dim', 24)
                device = next(model.parameters()).device
                example_input = torch.randn(1, input_dim, device=device, dtype=torch.float64)
                
                traced_model = torch.jit.trace(model, example_input, strict=False)
                fallback_path = model_dir / "model.pt"
                traced_model.save(fallback_path)
                return str(fallback_path)
            except Exception as fallback_e:
                print(f"Fallback save also failed: {fallback_e}")
                return None
    
    def _save_model_summary(self, model: torch.nn.Module, summary_path: Path):
        """Save human-readable model summary"""
        with open(summary_path, 'w') as f:
            f.write("MODEL SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Architecture: {model.__class__.__name__}\n")
            f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
            f.write(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
            
            if hasattr(model, 'input_dim'):
                f.write(f"Input Dimension: {model.input_dim}\n")
            if hasattr(model, 'task'):
                f.write(f"Task: {model.task}\n")
            
            f.write("\nModel Structure:\n")
            f.write(str(model))

class VisualizationHandler(ArtifactHandler):
    def can_handle(self, artifact: Any) -> bool:
        return (isinstance(artifact, str) and self._is_base64_image(artifact)) or \
               (isinstance(artifact, dict) and "image" in artifact)
    
    def get_artifact_type(self) -> ArtifactType:
        return ArtifactType.VISUALIZATION
    
    def save(self, artifact: Any, path: Path, metadata: ArtifactMetadata) -> Dict[str, Any]:
        """Save visualization artifacts"""
        viz_dir = path / "visualization_artifacts"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        try:
            if isinstance(artifact, dict) and "image" in artifact:
                base64_data = artifact["image"]
                viz_type = artifact.get("type", "plot")
            else:
                base64_data = artifact
                viz_type = "plot"
            
            # Detect plot type from metadata or content
            plot_type = self._detect_plot_type(metadata.name, viz_type)
            
            # Save in multiple formats
            formats = ['png', 'svg'] if plot_type == "loss_curve" else ['png']
            
            for fmt in formats:
                file_path = viz_dir / f"{metadata.name}.{fmt}"
                success = self._save_base64_image(base64_data, file_path, fmt)
                if success:
                    results[fmt] = str(file_path)
            
            # Save plot data if available
            if isinstance(artifact, dict) and "data" in artifact:
                data_path = viz_dir / f"{metadata.name}_data.json"
                with open(data_path, 'w') as f:
                    json.dump(artifact["data"], f, indent=2)
                results['data'] = str(data_path)
            
        except Exception as e:
            print(f"Error saving visualization: {e}")
            traceback.print_exc()
        
        return results
    
    def _is_base64_image(self, base64_str: str) -> bool:
        """Enhanced base64 image detection"""
        if not isinstance(base64_str, str) or len(base64_str) < 100:
            return False
        
        try:
            # Handle data URL format
            if base64_str.startswith('data:image/'):
                base64_data = base64_str.split(',')[1] if ',' in base64_str else base64_str
            else:
                base64_data = base64_str
            
            # Validate base64
            if not all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in base64_data):
                return False
            
            # Decode and verify
            decoded_data = base64.b64decode(base64_data, validate=True)
            if len(decoded_data) < 50:
                return False
            
            # Verify it's an image
            image_buffer = BytesIO(decoded_data)
            with Image.open(image_buffer) as img:
                img.verify()
            
            return True
            
        except Exception:
            return False
    
    def _detect_plot_type(self, name: str, viz_type: str) -> str:
        """Detect the type of plot from name and metadata"""
        name_lower = name.lower()
        
        if 'loss' in name_lower or 'curve' in name_lower:
            return "loss_curve"
        elif 'distribution' in name_lower or 'histogram' in name_lower:
            return "distribution"
        elif 'confusion' in name_lower or 'matrix' in name_lower:
            return "confusion_matrix"
        else:
            return viz_type or "generic"

    def _save_base64_image(self, base64_str: str, file_path: Path, format: str = 'png') -> bool:
        """Save base64 image with format support"""
        try:
            # Handle data URL format
            if base64_str.startswith('data:image/'):
                base64_data = base64_str.split(',')[1]
            else:
                base64_data = base64_str
            
            # Decode image
            decoded_image = base64.b64decode(base64_data)
            image_buffer = BytesIO(decoded_image)
            
            # Skip SVG format - PIL doesn't support SVG output
            if format.lower() == 'svg':
                print(f"   Skipping SVG format (not supported by PIL)")
                return False
            
            # Open and save
            with Image.open(image_buffer) as image:
                # Convert mode if necessary
                if format.lower() in ('jpg', 'jpeg') and image.mode in ('RGBA', 'LA', 'P'):
                    image = image.convert('RGB')
                
                # Save with appropriate format
                save_format = 'JPEG' if format.lower() in ('jpg', 'jpeg') else format.upper()
                quality = 95 if save_format == 'JPEG' else None
                
                image.save(file_path, format=save_format, quality=quality)
                return True
                
        except Exception as e:
            print(f"Error saving image to {file_path}: {e}")
            return False

class MetricsHandler(ArtifactHandler):
    def can_handle(self, artifact: Any) -> bool:
        return isinstance(artifact, dict) and any(
            key in artifact for key in ['loss', 'accuracy', 'metrics', 'epoch', 'training_history']
        )
    
    def get_artifact_type(self) -> ArtifactType:
        return ArtifactType.METRICS
    
    def save(self, artifact: Dict[str, Any], path: Path, metadata: ArtifactMetadata) -> Dict[str, Any]:
        """Save metrics and training statistics"""
        metrics_dir = path / "training_artifacts"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        try:
            # Save complete metrics as JSON
            metrics_path = metrics_dir / "training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(artifact, f, indent=2, default=str)
            results['metrics'] = str(metrics_path)
            
            # Create human-readable summary
            summary_path = metrics_dir / "training_summary.txt"
            self._create_training_summary(artifact, summary_path)
            results['summary'] = str(summary_path)
            
            # Extract and save training curves data
            if 'training_history' in artifact:
                curves_path = metrics_dir / "training_curves.csv"
                self._save_training_curves_csv(artifact['training_history'], curves_path)
                results['curves'] = str(curves_path)
            
        except Exception as e:
            print(f"Error saving metrics: {e}")
            traceback.print_exc()
        
        return results
    
    def _create_training_summary(self, metrics: Dict[str, Any], summary_path: Path):
        """Create human-readable training summary"""
        with open(summary_path, 'w') as f:
            f.write("TRAINING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Final metrics
            if 'final_metrics' in metrics:
                f.write("Final Metrics:\n")
                for key, value in metrics['final_metrics'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Training progress
            if 'training_history' in metrics:
                history = metrics['training_history']
                if history:
                    f.write("Training Progress:\n")
                    f.write(f"  Total Epochs: {len(history)}\n")
                    if 'train_loss' in history[-1]:
                        f.write(f"  Final Train Loss: {history[-1]['train_loss']:.6f}\n")
                    if 'val_loss' in history[-1]:
                        f.write(f"  Final Val Loss: {history[-1]['val_loss']:.6f}\n")
                    
                    # Find best epoch
                    if 'val_loss' in history[0]:
                        best_epoch = min(history, key=lambda x: x.get('val_loss', float('inf')))
                        f.write(f"  Best Epoch: {best_epoch.get('epoch', 'Unknown')}\n")
                        f.write(f"  Best Val Loss: {best_epoch.get('val_loss', 'Unknown'):.6f}\n")
                f.write("\n")
            
            # Performance metrics
            if 'performance' in metrics:
                perf = metrics['performance']
                f.write("Performance Metrics:\n")
                for key, value in perf.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
    
    def _save_training_curves_csv(self, history: List[Dict], curves_path: Path):
        """Save training curves as CSV for easy plotting"""
        if not history:
            return
        
        import csv
        
        # Get all possible fields
        all_fields = set()
        for epoch_data in history:
            all_fields.update(epoch_data.keys())
        
        with open(curves_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted(all_fields))
            writer.writeheader()
            for epoch_data in history:
                writer.writerow(epoch_data)

class ReportHandler(ArtifactHandler):
    def can_handle(self, artifact: Any) -> bool:
        return isinstance(artifact, str) and not self._is_base64_image(artifact)
    
    def get_artifact_type(self) -> ArtifactType:
        return ArtifactType.REPORT
    
    def save(self, artifact: str, path: Path, metadata: ArtifactMetadata) -> Dict[str, Any]:
        """Save text reports"""
        reports_dir = path / "evaluation_artifacts"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file type based on content
        file_extension = self._determine_file_type(artifact, metadata.name)
        file_path = reports_dir / f"{metadata.name}.{file_extension}"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(artifact)
        
        return {'report': str(file_path)}
    
    def _is_base64_image(self, text: str) -> bool:
        """Quick check if text is base64 image"""
        return len(text) > 1000 and (
            text.startswith('data:image/') or
            text.startswith('iVBORw0KGgoAAAANSUhEUgAA') or
            text.startswith('/9j/') or
            text.startswith('R0lGOD')
        )
    
    def _determine_file_type(self, content: str, name: str) -> str:
        """Determine appropriate file extension"""
        if 'json' in name.lower() or content.strip().startswith('{'):
            return 'json'
        elif 'yaml' in name.lower() or 'yml' in name.lower():
            return 'yaml'
        elif 'csv' in name.lower() or ',' in content[:100]:
            return 'csv'
        else:
            return 'txt'

class DictHandler(ArtifactHandler):
    """Handler for dictionary artifacts (evaluation reports, summaries, etc.)"""
    
    def can_handle(self, artifact: Any) -> bool:
        return isinstance(artifact, dict)
    
    def get_artifact_type(self) -> ArtifactType:
        return ArtifactType.REPORT
    
    def save(self, artifact: dict, base_path: Path, metadata: ArtifactMetadata) -> Dict[str, Any]:
        try:
            # Determine appropriate subdirectory
            if 'evaluation' in metadata.name.lower():
                subdir = "evaluation_artifacts"
            elif 'pipeline' in metadata.name.lower() or 'summary' in metadata.name.lower():
                subdir = "metadata"
            elif 'training' in metadata.name.lower():
                subdir = "training_artifacts"
            else:
                subdir = "derivatives"
            
            file_path = base_path / subdir / f"{metadata.name}.json"
            
            # Clean artifact for JSON serialization
            cleaned_artifact = self._clean_for_json(artifact)
            
            with open(file_path, 'w') as f:
                json.dump(cleaned_artifact, f, indent=2, default=str)
            
            return {
                'file_path': str(file_path),
                'format': 'json',
                'status': 'success',
                'size_mb': file_path.stat().st_size / (1024 * 1024)
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _clean_for_json(self, obj):
        """Convert complex types to JSON-serializable formats"""
        import numpy as np
        import torch
        
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif obj is None or isinstance(obj, (str, int, float)):
            return obj
        elif hasattr(obj, '__dict__'):
            return self._clean_for_json(obj.__dict__)
        else:
            return str(obj)

# In CArtifactsManager.py, replace the NotebookHandler with this simpler version:

class NotebookHandler(ArtifactHandler):
    """Lightweight handler for executing notebooks and saving HTML reports only"""
    
    def can_handle(self, artifact: Any) -> bool:
        return isinstance(artifact, str) and artifact.endswith('.ipynb')
    
    def get_artifact_type(self) -> ArtifactType:
        return ArtifactType.REPORT
    
    def save(self, notebook_path: str, path: Path, metadata: ArtifactMetadata) -> Dict[str, Any]:
        """Execute notebook and save only HTML report"""
        try:
            html_path = self._execute_and_convert_to_html(notebook_path, path, metadata.name)
            if html_path:
                return {'html_report': str(html_path), 'status': 'success'}
            else:
                return {'status': 'failed', 'error': 'HTML conversion failed'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _execute_and_convert_to_html(self, notebook_path: str, output_dir: Path, name: str) -> Optional[str]:
        """Execute notebook and directly convert to HTML in one step"""
        import subprocess
        
        try:
            # Create simple output filename
            html_path = output_dir / f"data_analysis_report.html"
            
            # Execute notebook and convert to HTML in one command
            cmd = [
                "jupyter", "nbconvert",
                "--to", "html",
                "--execute",
                "--output", str(html_path),
                str(notebook_path)
            ]
            
            print(f"Executing notebook and generating HTML report...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"✅ HTML report saved to: {html_path}")
                return str(html_path)
            else:
                print(f"❌ Failed to generate HTML: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            return None
# ============================================================================
# 3. CONTEXT CAPTURE SYSTEM
# ============================================================================

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