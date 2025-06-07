import yaml
import os
import torch
from src.CDataProcessor import CDataProcessor
from src.CArchitectureManager import CArchitectureManager
from src.CModelTrainer import CModelTrainer
from src.CArtifactsManager import CArtifactsManager
from src.evaluator.CEvaluator import CEvaluator

class CPipelineOrchestrator:
    def __init__(self, config_file="config/config.yaml"):
        self.config = self._load_config(config_file)
        self.artifacts_manager = CArtifactsManager(self.config) 
        self.architecture_manager = CArchitectureManager(self.config["model_config"])
        self.data_processor = CDataProcessor(self.config["processor_config"])
        self.model_trainer = CModelTrainer(self.config["trainer_config"])
        self.evaluator = CEvaluator(self.config["evaluator_config"])

    def run(self):
        """Execute the complete pipeline."""
        state = {}

        if self.config['processor_config'].get('skip_processing', True):
            print("-- Skipped Processing Data --")
        else:
            self._process_data_step(state)
            print("---- Data Processing Complete ----")

        if self.config['model_config'].get('skip_building', True):
            print("-- Skipped Model Building --")            
        else:
            self._build_model_step(state)
            print("-- Model Building Complete ---")


        if not self.config['model_config'].get('skip_training', False):
            self._train_model_step(state)
            print("---- Model Training Complete ----")

        elif self.config['model_config'].get('model_path'):
            self._load_model_step(state)
            print("---- Model Loaded Successfully ----")
        else:
            raise ValueError("No model path specified and training is skipped. Cannot proceed without a model.")


        if not self.config['evaluator_config'].get('skip_evaluation', False):
            self._evaluate_model_step(state)
        else:
            print("-- Skipped Evaluation --")
    
    def _load_config(self, config_file):

        with open(config_file, "r") as file:
            print("- Configuration Loaded -")
            config = yaml.safe_load(file)

        home = config.get('home')
        if not home:
            raise ValueError("The 'home' path is missing from the configuration.")

        def join_with_home(path):
            # If path is already absolute, leave it unchanged.
            if os.path.isabs(path):
                return path
            return os.path.join(home, path)

        # Update artifacts_config path.
        artifacts = config.get('artifacts_config', {})
        if 'save_artifacts_to' in artifacts:
            artifacts['save_artifacts_to'] = join_with_home(artifacts['save_artifacts_to'])
        
        # Update dataset paths under processor_config.
        processor = config.get('processor_config', {})
        dataset_paths = processor.get('dataset_paths', {})
        for key, value in dataset_paths.items():
            dataset_paths[key] = join_with_home(value)
        
        # Update trainer_config paths for fine-tuning
        trainer = config.get('trainer_config', {})
        if 'fine_tune_model_path' in trainer:
            trainer['fine_tune_model_path'] = join_with_home(trainer['fine_tune_model_path'])
        if 'fine_tune_data_path' in trainer:
            trainer['fine_tune_data_path'] = join_with_home(trainer['fine_tune_data_path'])
            
        # Update evaluator_config path.
        evaluator = config.get('evaluator_config', {})
        if 'test_data_path' in evaluator:
            evaluator['test_data_path'] = join_with_home(evaluator['test_data_path'])

        if 'model_path' in evaluator:
            evaluator['model_path'] = join_with_home(evaluator['model_path'])

        return config

    def _process_data_step(self, state):
        if self.config['processor_config'].get("skip_processing", True):
            print("-- Skipped Processing Data --")
            return
        self.data_processor.process()

    def _build_model_step(self, state):
        if self.config['model_config'].get("skip_building", True):
            print("-- Skipped building architecture --")
            return
        model_architecture = self.architecture_manager.get_model()
        state["model"] = model_architecture

    def _load_model_step(self, state):
        """Load a model from the specified path"""
        model_path = self.config['evaluator_config'].get("model_path")
        if model_path:
            try:
                # Convert relative path to absolute path
                if not os.path.isabs(model_path):
                    home_path = self.config.get('home', '')
                    full_model_path = os.path.join(home_path, model_path.lstrip('/'))
                else:
                    full_model_path = model_path
                
                print(f"-- Loading model from {full_model_path} --")
                
                # Load the model
                import torch
                class TorchScriptWrapper:
                    def __init__(self, model):
                        self.model = model
                        self.task = model.task
                        self.volume_scale_factor = model.volume_scale_factor

                    
                    def predict(self, x):
                        """Common prediction logic for all networks"""
                        self.model.eval()
                        with torch.no_grad():

                            processed_embeddings = self.model.forward(x) 

                            if self.task == 'IntersectionStatus':
                                return (processed_embeddings > 0.5).int().squeeze() # Using binary cross-entropy with logits, sgimoid is applied in the loss function
                            elif self.task == 'IntersectionVolume':
                                return processed_embeddings.squeeze() / self.volume_scale_factor
                            elif self.task == 'IntersectionStatus_IntersectionVolume':
                                cls_prediction = (processed_embeddings[:, 0] > 0.5).int().squeeze() # Using binary cross-entropy with logits, sgimoid is applied in the loss function
                                reg_prediction = processed_embeddings[:, 1].squeeze() / self.volume_scale_factor
                                return torch.stack([cls_prediction, reg_prediction], dim=1)         
                            raise ValueError(f"Unknown task: {self.task}")
            
                loaded_model = torch.jit.load(full_model_path)
                loaded_model_with_predict = TorchScriptWrapper(loaded_model)

                return loaded_model_with_predict
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"No model path specified in {model_path}.")

    def _train_model_step(self, state):
        """Train or load a model based on configuration"""
        
        # Check if fine-tuning is enabled
        if self.config['trainer_config'].get('fine_tune_on', False):
            self._fine_tune_model_step(state)
            return
        
        # Regular training flow
        model = state.get("model", None)
        
        if not model or (self.config['trainer_config']['skip_training'] and self.config['evaluator_config']['model_path']):
            # If a model path is specified, load the model
            loaded_model = self._load_model_step(state)
            state["model"] = loaded_model
            print(f"---- Will train model with {self.config['evaluator_config']['model_path']} ----")

        assert model, "Model must be provided for training."

        # Train the model
        trained_model, training_report = self.model_trainer.train_and_validate(model)
        if trained_model is None:
            raise ValueError("Training failed. No trained model was returned.")
        if training_report is None:
            raise ValueError("Training report is missing. Training might have failed.")
        
        state["trained_model"] = trained_model
        state["training_report"] = training_report

        # Save artifacts
        self.artifacts_manager.save_artifacts(trained_model, training_report)
        print("---- Training Complete ----")

    def _fine_tune_model_step(self, state):
        """Fine-tune a pre-trained model with new data"""
        
        # Get fine-tuning configuration
        trainer_config = self.config['trainer_config']
        fine_tune_model_path = trainer_config.get('fine_tune_model_path')
        fine_tune_data_path = trainer_config.get('fine_tune_data_path')
        
        if not fine_tune_model_path:
            raise ValueError("fine_tune_model_path is required when fine_tune_on is True")
        if not fine_tune_data_path:
            raise ValueError("fine_tune_data_path is required when fine_tune_on is True")
        
        print(f"---- Starting Fine-tuning ----")
        print(f"Model path: {fine_tune_model_path}")
        print(f"Fine-tune data: {fine_tune_data_path}")
        
        # Load the pre-trained model
        loaded_model = self._load_model_for_fine_tuning(fine_tune_model_path)
        
        # Prepare fine-tuning data
        fine_tune_data = self._prepare_fine_tune_data(fine_tune_data_path)
        
        # Fine-tune the model
        fine_tuned_model, training_report = self.model_trainer.fine_tune(
            loaded_model, fine_tune_data
        )
        
        if fine_tuned_model is None:
            raise ValueError("Fine-tuning failed. No model was returned.")
        
        state["trained_model"] = fine_tuned_model
        state["training_report"] = training_report
        
        # Save artifacts
        self.artifacts_manager.save_artifacts(fine_tuned_model, training_report)
        print("---- Fine-tuning Complete ----")

    def _load_model_for_fine_tuning(self, model_path):
        """Load a model specifically for fine-tuning (keeping it trainable)"""
        try:
            # Convert relative path to absolute path
            if not os.path.isabs(model_path):
                home_path = self.config.get('home', '')
                full_model_path = os.path.join(home_path, model_path.lstrip('/'))
            else:
                full_model_path = model_path
            
            print(f"-- Loading model for fine-tuning from {full_model_path} --")
            
            import torch
            
            # Determine device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            
            # Try to load as TorchScript first
            try:
                loaded_model = torch.jit.load(full_model_path, map_location=device)
                print("Loaded TorchScript model successfully")
            except Exception as e:
                print(f"Failed to load as TorchScript: {e}")
                # If TorchScript fails, try loading state dict
                checkpoint = torch.load(full_model_path, map_location=device)
                raise ValueError("State dict loading not implemented. Please use TorchScript models for fine-tuning.")
            
            # Convert TorchScript back to regular PyTorch module for fine-tuning
            fine_tune_model = self._convert_torchscript_to_pytorch(loaded_model)
            
            # Move model to correct device
            fine_tune_model = fine_tune_model.to(device)
            
            # Put model in training mode for fine-tuning
            fine_tune_model.train()
            
            # Enable gradients for all parameters
            for param in fine_tune_model.parameters():
                param.requires_grad = True
                
            return fine_tune_model
            
        except Exception as e:
            raise ValueError(f"Error loading model for fine-tuning: {e}")

    def _convert_torchscript_to_pytorch(self, torchscript_model):
        """
        Convert a TorchScript model back to a regular PyTorch model for fine-tuning.
        This is a workaround since TorchScript models are not easily fine-tunable.
        """
        # Determine device
        device = next(torchscript_model.parameters()).device
        print(f"TorchScript model is on device: {device}")
        
        try:
            # Try to extract model metadata from TorchScript model
            task = getattr(torchscript_model, 'task', None)
            input_dim = getattr(torchscript_model, 'input_dim', None)
            volume_scale_factor = getattr(torchscript_model, 'volume_scale_factor', None)
            
            print(f"Extracted metadata - Task: {task}, Input dim: {input_dim}, Volume scale: {volume_scale_factor}")
            
        except Exception as e:
            print(f"Could not extract metadata from TorchScript model: {e}")
            task = None
            input_dim = None
            volume_scale_factor = None
        
        # If metadata extraction fails, use defaults or infer from config
        if task is None:
            task = self.config['model_config']['common_parameters'].get('task', 'IntersectionStatus_IntersectionVolume')
            print(f"Using default task from config: {task}")
        
        if input_dim is None:
            input_dim = self.config['model_config']['common_parameters'].get('input_dim', 24)
            print(f"Using default input_dim from config: {input_dim}")
        
        if volume_scale_factor is None:
            volume_scale_factor = self.config['model_config']['common_parameters'].get('volume_scale_factor', 1.0)
            print(f"Using default volume_scale_factor from config: {volume_scale_factor}")
        
        # Update config to match the loaded model
        self.config['model_config']['common_parameters']['task'] = task
        self.config['model_config']['common_parameters']['input_dim'] = input_dim
        self.config['model_config']['common_parameters']['volume_scale_factor'] = volume_scale_factor
        
        # Create new model with same architecture
        new_model = self.architecture_manager.get_model()
        
        # Move new model to same device as TorchScript model
        new_model = new_model.to(device)
        
        # Copy weights from TorchScript model to new model
        try:
            # Get state dict from TorchScript model
            torchscript_state_dict = torchscript_model.state_dict()
            
            # Get state dict from new model
            new_model_state_dict = new_model.state_dict()
            
            print("Attempting to map parameters...")
            matched_params = 0
            total_params = len(new_model_state_dict)
            
            # Create a more flexible parameter mapping
            for new_key in new_model_state_dict.keys():
                param_found = False
                
                # Try exact match first
                if new_key in torchscript_state_dict:
                    if new_model_state_dict[new_key].shape == torchscript_state_dict[new_key].shape:
                        # Ensure tensor is on the correct device
                        new_model_state_dict[new_key] = torchscript_state_dict[new_key].to(device)
                        matched_params += 1
                        param_found = True
                
                # If exact match fails, try fuzzy matching
                if not param_found:
                    for ts_key in torchscript_state_dict.keys():
                        # Remove common TorchScript prefixes/suffixes
                        clean_ts_key = ts_key.replace('original_name.', '').replace('_c.', '.')
                        
                        if (new_key in clean_ts_key or clean_ts_key.endswith(new_key) or 
                            new_key.split('.')[-1] == clean_ts_key.split('.')[-1]):
                            
                            if new_model_state_dict[new_key].shape == torchscript_state_dict[ts_key].shape:
                                # Ensure tensor is on the correct device
                                new_model_state_dict[new_key] = torchscript_state_dict[ts_key].to(device)
                                matched_params += 1
                                param_found = True
                                break
                
                if not param_found:
                    print(f"Warning: Could not find matching parameter for {new_key}")
            
            print(f"Successfully mapped {matched_params}/{total_params} parameters")
            
            # Load the mapped state dict
            new_model.load_state_dict(new_model_state_dict, strict=False)
            print("Successfully copied weights from TorchScript to PyTorch model")
            
        except Exception as e:
            print(f"Warning: Could not copy weights: {e}")
            print("Fine-tuning will start from randomly initialized weights")
        
        return new_model

    def _prepare_fine_tune_data(self, fine_tune_data_path):
        """Prepare the fine-tuning dataset"""
        try:
            # Convert relative path to absolute path
            if not os.path.isabs(fine_tune_data_path):
                home_path = self.config.get('home', '')
                full_data_path = os.path.join(home_path, fine_tune_data_path.lstrip('/'))
            else:
                full_data_path = fine_tune_data_path
            
            print(f"-- Loading fine-tune data from {full_data_path} --")
            
            # Use the data processor to load and process the fine-tune data
            # You might need to modify CDataProcessor to handle single file loading
            import pandas as pd
            fine_tune_df = pd.read_csv(full_data_path)
            
            # Apply same transformations as training data if needed
            # This might require exposing some methods from CDataProcessor
            
            return fine_tune_df
            
        except Exception as e:
            raise ValueError(f"Error loading fine-tune data: {e}")

    def _evaluate_model_step(self, state):
        """Evaluate the model based on the configuration"""
        trained_model = state.get("trained_model")
        if trained_model is None:
            raise ValueError("Trained model is required for evaluation but was not provided.")
        evaluation_report = self.evaluator.evaluate(trained_model)
        self.artifacts_manager.save_artifacts(evaluation_report)
        state["evaluation_report"] = evaluation_report