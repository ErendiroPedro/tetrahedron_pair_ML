import os
import base64
import torch
from io import BytesIO
from PIL import Image
import datetime
import json
import yaml
import traceback 
from typing import Dict, Any, List

class CArtifactsManager:
    def __init__(self, config):
        self.config = config
        self.artifacts_path = self._create_experiment_folder()

    def save_artifacts(self, *args):
        """
        Save various types of artifacts including PyTorch models, images, and text files.

        Args:
            *args: Unnamed artifacts to save.
                - torch.nn.Module: Saved with weights and architecture.
                - str: Saved as a text file or decoded image if Base64-encoded.
                - dict with "image" and "type" keys: Encoded in Base64 and saved as a text file.
        """
        for index, artifact in enumerate(args):
            if isinstance(artifact, torch.nn.Module):
                self._save_model_state_dict(artifact)
                self._save_optimized_model(artifact)  # Changed from _save_model_with_weights
            elif isinstance(artifact, str) and self._is_base64_image(artifact):
                self._save_base64_image(artifact, f"loss_report")
            elif isinstance(artifact, dict):
                self._save_evaluation_report(artifact)
            else:
                raise ValueError(f"Unsupported artifact type for argument at position {index}")

    def _create_experiment_folder(self):
        """
        Generate a descriptive folder name based on the configuration and save the config file in the folder.

        Returns:
            str: Path to the created folder.
        """
        # Generate a timestamp in reverse chronological order for sorting
        id = datetime.datetime.now().strftime("%m%d%H%M%S")
        
        # Extract relevant configuration details
        model_architecture = self.config['model_config']['architecture']['use_model']
        task = self.config['model_config']['common_parameters']['task']
        num_train_samples = self.config['processor_config']['num_train_samples']
        num_val_samples = self.config['processor_config']['num_val_samples']
        volume_range = "-".join(map(str, self.config['processor_config']['volume_range']))
        epochs = self.config['trainer_config']['epochs']
        optimizer = self.config['trainer_config']['optimizer']
        learning_rate = self.config['trainer_config']['learning_rate']

        # Add distribution details
        distributions = self.config['processor_config']['intersection_distributions']
        dist_str = "_".join(f"{key}-{value}" for key, value in distributions.items() if value > 0)

        # Construct the folder name
        folder_name = (
            f"id{id}_{model_architecture}_{task}_"
            f"{num_train_samples}-train_{num_val_samples}-val_"
            f"vol-{volume_range}_epochs-{epochs}_{optimizer}-{learning_rate}_"
            f"distributions-{dist_str}"
        )
        
        # Create the folder
        artifacts_path = self.config['artifacts_config']['save_artifacts_to']
        full_folder_path = os.path.join(artifacts_path, folder_name)
        os.makedirs(full_folder_path, exist_ok=True)

        # Save config
        config_file_path = os.path.join(full_folder_path, "config.yaml")
        with open(config_file_path, 'w') as config_file:
            yaml.dump(self.config, config_file, default_flow_style=False)

        return full_folder_path
    
    def _save_optimized_model(self, model):
        """
        Save a PyTorch model using torch.compile optimization and export for deployment.
        
        Args:
            model (torch.nn.Module): PyTorch model to save.
        """
        # File paths
        model_path_original = os.path.join(self.artifacts_path, "model_original.pth")
        model_path_optimized = os.path.join(self.artifacts_path, "model_optimized.pt")
        final_model_path = os.path.join(self.artifacts_path, "model.pt")
        weights_path = os.path.join(self.artifacts_path, "model_weights.json")
        config_path = os.path.join(self.artifacts_path, "model_config.json")
        
        try:
            # Ensure model is in evaluation mode
            model.eval()
            
            # Export model configuration and weights first
            print("Exporting model weights and configuration...")
            self._export_model_weights(model, weights_path, config_path)
            
            # Save original model state dict as backup
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'config': self._extract_model_config(model)
            }, model_path_original)
            print(f"Original model state saved to {model_path_original}")
            
            # --- Optimize with torch.compile ---
            print("Optimizing model with torch.compile...")
            
            # Get optimization settings from config
            optimization_config = self.config.get('evaluator_config', {})
            compile_mode = optimization_config.get('optimization_mode', 'default')
            batch_size = optimization_config.get('batch_size', 32)
            
            # Infer input dimension
            input_dim = model.input_dim if hasattr(model, 'input_dim') else None
            if input_dim is None:
                raise ValueError("Model does not have an input_dim attribute. Please provide the input dimension.")
            
            # Create sample input for optimization
            device = next(model.parameters()).device
            sample_input = torch.randn(batch_size, input_dim, device=device, dtype=torch.float32)
            
            # Compile the model
            print(f"Compiling model with mode='{compile_mode}', batch_size={batch_size}")
            compiled_model = torch.compile(
                model,
                mode=compile_mode,        # "default", "reduce-overhead", "max-autotune"
                fullgraph=True,          # Compile entire graph for better optimization
                dynamic=False            # Fixed shapes for better optimization
            )
            
            # Warm up the compiled model
            print("Warming up compiled model (this may take a moment)...")
            with torch.no_grad():
                warmup_runs = 5
                for i in range(warmup_runs):
                    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                    
                    if start_time:
                        start_time.record()
                    
                    _ = compiled_model(sample_input)
                    
                    if start_time:
                        end_time.record()
                        torch.cuda.synchronize()
                        elapsed_time = start_time.elapsed_time(end_time)
                        print(f"  Warmup run {i+1}/{warmup_runs}: {elapsed_time:.2f}ms")
                    else:
                        print(f"  Warmup run {i+1}/{warmup_runs} completed")
            
            print("Model compilation and warmup completed!")
            
            # --- Export optimized model for deployment ---
            print("Exporting optimized model for C++ deployment...")
            
            # Create a deployment wrapper that includes the compiled model
            deployment_model = self._create_deployment_wrapper(compiled_model, sample_input)
            
            # Trace the warmed-up compiled model for deployment
            traced_optimized = torch.jit.trace(deployment_model, sample_input, strict=False)
            traced_optimized.save(model_path_optimized)
            print(f"Optimized model exported to {model_path_optimized}")
            
            # Create the final model.pt (use optimized version)
            traced_optimized.save(final_model_path)
            print(f"Final optimized model saved as {final_model_path}")
            
            # Performance verification
            self._verify_model_performance(model, compiled_model, sample_input)
            
            return final_model_path
            
        except Exception as e:
            print(f"\nError during model optimization: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            
            # Fallback: save original model without optimization
            print("\nFalling back to saving original model without optimization...")
            return self._save_fallback_model(model, final_model_path)

    def _create_deployment_wrapper(self, compiled_model, sample_input):
        """
        Create a deployment wrapper around the compiled model.
        This ensures the optimizations are preserved during tracing.
        """
        class DeploymentWrapper(torch.nn.Module):
            def __init__(self, compiled_model):
                super().__init__()
                self.compiled_model = compiled_model
            
            def forward(self, x):
                return self.compiled_model(x)
        
        wrapper = DeploymentWrapper(compiled_model)
        wrapper.eval()
        
        # Ensure the wrapper works with the sample input
        with torch.no_grad():
            _ = wrapper(sample_input)
        
        return wrapper

    def _verify_model_performance(self, original_model, compiled_model, sample_input):
        """
        Verify that the compiled model produces the same results and measure performance.
        """
        print("Verifying model performance...")
        
        with torch.no_grad():
            # Check numerical equivalence
            original_output = original_model(sample_input)
            compiled_output = compiled_model(sample_input)
            
            # Check if outputs are close
            max_diff = torch.max(torch.abs(original_output - compiled_output))
            print(f"Maximum difference between original and compiled outputs: {max_diff:.2e}")
            
            if max_diff < 1e-5:
                print("✓ Outputs are numerically equivalent")
            else:
                print("⚠ Warning: Outputs differ more than expected")
            
            # Performance benchmark
            num_runs = 50
            
            # Benchmark original
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
                for _ in range(num_runs):
                    _ = original_model(sample_input)
                end_time.record()
                torch.cuda.synchronize()
                original_time = start_time.elapsed_time(end_time) / num_runs
            else:
                import time
                start = time.time()
                for _ in range(num_runs):
                    _ = original_model(sample_input)
                original_time = (time.time() - start) * 1000 / num_runs
            
            # Benchmark compiled
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if start_time:
                start_time.record()
                for _ in range(num_runs):
                    _ = compiled_model(sample_input)
                end_time.record()
                torch.cuda.synchronize()
                compiled_time = start_time.elapsed_time(end_time) / num_runs
            else:
                start = time.time()
                for _ in range(num_runs):
                    _ = compiled_model(sample_input)
                compiled_time = (time.time() - start) * 1000 / num_runs
            
            speedup = original_time / compiled_time
            print(f"Performance comparison (average over {num_runs} runs):")
            print(f"  Original model: {original_time:.2f}ms per batch")
            print(f"  Compiled model: {compiled_time:.2f}ms per batch")
            print(f"  Speedup: {speedup:.2f}x")

    def _save_fallback_model(self, model, final_model_path):
        """
        Fallback method to save model using TorchScript if torch.compile fails.
        """
        try:
            print("Attempting TorchScript tracing as fallback...")
            input_dim = model.input_dim if hasattr(model, 'input_dim') else 24
            device = next(model.parameters()).device
            example_input = torch.randn(1, input_dim, device=device)
            
            traced_model = torch.jit.trace(model, example_input, strict=False)
            traced_model.save(final_model_path)
            print(f"Fallback: Model saved using TorchScript tracing to {final_model_path}")
            return final_model_path
            
        except Exception as fallback_e:
            print(f"Fallback also failed: {fallback_e}")
            # Last resort: save state dict
            torch.save(model.state_dict(), final_model_path.replace('.pt', '_state_dict.pth'))
            print(f"Last resort: Saved model state dict to {final_model_path.replace('.pt', '_state_dict.pth')}")
            return None

    def _extract_model_config(self, model):
        """Extract model configuration for saving."""
        config = {}
        
        if hasattr(model, 'common_params'):
            config['common_params'] = {
                'input_dim': getattr(model.common_params, 'input_dim', None),
                'task': getattr(model.common_params, 'task', None),
                'activation': getattr(model.common_params, 'activation', None),
                'dropout_rate': getattr(model.common_params, 'dropout_rate', None),
                'volume_scale_factor': getattr(model.common_params, 'volume_scale_factor', None)
            }
        
        if hasattr(model, 'mlp_params'):
            config['mlp_params'] = {
                'shared_layers': getattr(model.mlp_params, 'shared_layers', None),
                'classification_head': getattr(model.mlp_params, 'classification_head', None),
                'regression_head': getattr(model.mlp_params, 'regression_head', None)
            }
        
        return config

    def _save_model_state_dict(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, epoch: int = None):
        """
        Save model state_dict along with optional optimizer state and epoch.
        """
        checkpoint = {
            'model_state_dict': model.state_dict()
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch

        path = os.path.join(self.artifacts_path, 'model_state_dict.pth')
        torch.save(checkpoint, path)
        print(f"Model state_dict saved to {path}")


    def _save_model_with_weights(self, model):
        """
        Save a PyTorch model using TorchScript and export weights in multiple formats.
        
        Args:
            model (torch.nn.Module): PyTorch model to save.
        """
        model_path_trace = os.path.join(self.artifacts_path, "model_traced.pt")
        model_path_script = os.path.join(self.artifacts_path, "model_scripted.pt")
        final_model_path = os.path.join(self.artifacts_path, "model.pt")
        weights_path = os.path.join(self.artifacts_path, "model_weights.json")
        config_path = os.path.join(self.artifacts_path, "model_config.json")
        
        try:
            # Ensure model is in evaluation mode
            model.eval()
            
            # Export model configuration and weights first
            print("Exporting model weights and configuration...")
            self._export_model_weights(model, weights_path, config_path)
            
            # --- Try Tracing First ---
            print("Attempting to save model using torch.jit.trace...")
            
            # Infer input dimension
            input_dim = model.input_dim if hasattr(model, 'input_dim') else None
            if input_dim is None:
                raise ValueError("Model does not have an input_dim attribute. Please provide the input dimension.")
            
            # Create dummy input
            device = next(model.parameters()).device
            example_input = torch.randn(1, input_dim, device=device)
            
            # Trace
            traced_model = torch.jit.trace(model, example_input, strict=False)
            
            # Save traced model
            traced_model.save(model_path_trace)
            print(f"Model successfully traced and saved to {model_path_trace}")
            
            # Create the generic 'model.pt' link/copy
            torch.jit.load(model_path_trace).save(final_model_path)
            print(f"Saved final model as {final_model_path} (using traced version)")
            
            return # Success
            
        except Exception as trace_e:
            print(f"\nError saving model using torch.jit.trace: {trace_e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            print("\nAttempting to save using torch.jit.script as fallback...")
            
            # --- Fallback to Scripting ---
            try:
                model.eval()
                script_model = torch.jit.script(model)
                script_model.save(model_path_script)
                print(f"Model successfully scripted and saved to {model_path_script}")
                
                # Create the generic 'model.pt' link/copy
                torch.jit.load(model_path_script).save(final_model_path)
                print(f"Saved final model as {final_model_path} (using scripted version)")
                
                return # Success
                
            except Exception as script_e:
                print(f"\nError saving model using torch.jit.script as well: {script_e}")
                print(f"Traceback:\n{traceback.format_exc()}")
                print("\nFailed to save model using both trace and script methods.")

    def _export_model_weights(self, model, weights_path: str, config_path: str):
        """
        Export model weights and configuration to JSON files for C++ loading.
        
        Args:
            model: PyTorch model
            weights_path: Path to save weights JSON
            config_path: Path to save configuration JSON
        """
        weights_dict = {}
        config_dict = {}
        
        # Extract model configuration
        if hasattr(model, 'common_params'):
            config_dict['common_params'] = {
                'input_dim': model.common_params.input_dim,
                'task': model.common_params.task,
                'activation': model.common_params.activation,
                'dropout_rate': model.common_params.dropout_rate,
                'volume_scale_factor': model.common_params.volume_scale_factor
            }
        
        if hasattr(model, 'mlp_params'):
            config_dict['mlp_params'] = {
                'shared_layers': model.mlp_params.shared_layers,
                'classification_head': model.mlp_params.classification_head,
                'regression_head': model.mlp_params.regression_head
            }
        
        if hasattr(model, 'tpn_params'):
            config_dict['tpn_params'] = {
                'per_vertex_layers': model.tpn_params.per_vertex_layers,
                'per_tetrahedron_layers': model.tpn_params.per_tetrahedron_layers,
                'per_two_tetrahedra_layers': model.tpn_params.per_two_tetrahedra_layers,
                'vertices_aggregation_function': model.tpn_params.vertices_aggregation_function,
                'tetrahedra_aggregation_function': model.tpn_params.tetrahedra_aggregation_function
            }
        
        # Extract weights based on model type
        if hasattr(model, '__class__') and 'TetrahedronPairNet' in str(model.__class__):
            weights_dict = self._extract_tpn_weights(model)
        elif hasattr(model, '__class__') and 'MLP' in str(model.__class__):
            weights_dict = self._extract_mlp_weights(model)
        else:
            # Generic weight extraction
            weights_dict = self._extract_generic_weights(model)
        
        # Save to JSON files
        with open(weights_path, 'w') as f:
            json.dump(weights_dict, f, indent=2)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Weights exported to {weights_path}")
        print(f"Configuration exported to {config_path}")

    def _extract_tpn_weights(self, model) -> Dict[str, Any]:
        """Extract weights from TetrahedronPairNet model."""
        weights = {}
        
        # Vertex MLPs (8 vertex processors)
        for i in range(8):
            if hasattr(model, f'vertex_mlps') and len(model.vertex_mlps) > i:
                vertex_mlp = model.vertex_mlps[i]
                weights[f'vertex_mlp_{i}'] = self._extract_sequential_weights(vertex_mlp)
            
            if hasattr(model, f'vertex_residual_layers') and len(model.vertex_residual_layers) > i:
                res_layer = model.vertex_residual_layers[i]
                weights[f'vertex_residual_{i}'] = self._extract_linear_weights(res_layer)
        
        # Post-pooling processors
        if hasattr(model, 'process_tet_post_pooling_1'):
            weights['process_tet_post_pooling_1'] = self._extract_sequential_weights(model.process_tet_post_pooling_1)
        
        if hasattr(model, 'process_tet_post_pooling_2'):
            weights['process_tet_post_pooling_2'] = self._extract_sequential_weights(model.process_tet_post_pooling_2)
        
        if hasattr(model, 'process_combined_embd'):
            weights['process_combined_embd'] = self._extract_sequential_weights(model.process_combined_embd)
        
        # Global residual
        if hasattr(model, 'global_residual'):
            weights['global_residual'] = self._extract_linear_weights(model.global_residual)
        
        # Shared layers
        if hasattr(model, 'shared_layers'):
            weights['shared_layers'] = self._extract_sequential_weights(model.shared_layers)
        
        # Task heads
        if hasattr(model, 'classifier_head'):
            weights['classifier_head'] = self._extract_sequential_weights(model.classifier_head)
        
        if hasattr(model, 'regressor_head'):
            weights['regressor_head'] = self._extract_sequential_weights(model.regressor_head)
        
        return weights

    def _extract_mlp_weights(self, model) -> Dict[str, Any]:
        """Extract weights from MLP model."""
        weights = {}
        
        if hasattr(model, 'shared_layers'):
            weights['shared_layers'] = self._extract_sequential_weights(model.shared_layers)
        
        if hasattr(model, 'classifier_head'):
            weights['classifier_head'] = self._extract_sequential_weights(model.classifier_head)
        
        if hasattr(model, 'regressor_head'):
            weights['regressor_head'] = self._extract_sequential_weights(model.regressor_head)
        
        return weights

    def _extract_sequential_weights(self, sequential_module) -> List[Dict[str, List]]:
        """Extract weights from a Sequential-like module."""
        layer_weights = []
        
        for name, module in sequential_module.named_children():
            if hasattr(module, 'weight') and hasattr(module, 'bias'):
                layer_weights.append(self._extract_linear_weights(module))
        
        return layer_weights

    def _extract_linear_weights(self, linear_module) -> Dict[str, List]:
        """Extract weights and biases from a Linear module."""
        weight_data = linear_module.weight.detach().cpu().numpy().tolist()
        bias_data = linear_module.bias.detach().cpu().numpy().tolist() if linear_module.bias is not None else []
        
        return {
            'weight': weight_data,
            'bias': bias_data
        }

    def _extract_generic_weights(self, model) -> Dict[str, Any]:
        """Generic weight extraction for any PyTorch model."""
        weights = {}
        
        for name, param in model.named_parameters():
            weights[name] = param.detach().cpu().numpy().tolist()
        
        return weights

    def _save_evaluation_report(self, report):
        """
        Save the evaluation report as a JSON file.

        Args:
            report (dict): The evaluation report dictionary to save.
        """
        filepath = os.path.join(self.artifacts_path, f"evaluation_report.json")
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

    def _is_base64_image(self, base64_str):
        """
        Check if a string is a Base64-encoded image.

        Args:
            base64_str (str): The string to check.
        
        Returns:
            bool: True if the string is a Base64-encoded image, otherwise False.
        """
        try:
            base64.b64decode(base64_str)  # Check if it decodes correctly
            return True
        except Exception:
            return False

    def _save_base64_image(self, base64_str, filename):
        """
        Decode a Base64-encoded image and save it as a file.

        Args:
            base64_str (str): The Base64 string of the image.
            filename (str): Name of the file to save the image as.
        """
        decoded_image = base64.b64decode(base64_str)
        image_buffer = BytesIO(decoded_image)
        image = Image.open(image_buffer)
        filepath = os.path.join(self.artifacts_path, f"{filename}.png")
        image.save(filepath, format="PNG")