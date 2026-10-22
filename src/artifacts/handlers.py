import base64
import json
import subprocess
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from src.artifacts.types import ArtifactHandler, ArtifactMetadata, ArtifactType


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
