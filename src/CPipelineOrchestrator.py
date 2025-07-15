import yaml
import os
import torch
from src.CDataProcessor import CDataProcessor
from src.CArchitectureManager import CArchitectureManager
from src.CModelTrainer import CModelTrainer
from src.CArtifactsManager import CArtifactsManager, StorageLevel
from src.evaluator.CEvaluator import CEvaluator
import traceback

class CPipelineOrchestrator:
    def __init__(self, config_file="config/config.yaml"):
        self.config = self._load_config(config_file)
        
        # Initialize artifacts manager with storage level from config
        storage_level = self.config.get('artifacts_config', {}).get('storage_level', 'standard')
        storage_level_enum = getattr(StorageLevel, storage_level.upper(), StorageLevel.STANDARD)
        self.artifacts_manager = CArtifactsManager(self.config, storage_level_enum)
        
        self.architecture_manager = CArchitectureManager(self.config["model_config"])
        self.data_processor = CDataProcessor(self.config["processor_config"])
        self.model_trainer = CModelTrainer(self.config["trainer_config"])
        self.evaluator = CEvaluator(self.config["evaluator_config"], self.config["processor_config"])


    def run(self):
        """Execute the complete pipeline."""
        state = {}
        
        try:
            # Log experiment start
            print(f"\n=== Starting Pipeline Experiment ===")
            experiment_summary = self.artifacts_manager.get_experiment_summary()
            print(f"Experiment ID: {experiment_summary['experiment_id']}")
            print(f"Artifacts Path: {experiment_summary['path']}")

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

            # Check if we should skip training and load a model instead
            skip_training = self.config['model_config'].get('skip_training', False)
            has_evaluator_model_path = self.config['evaluator_config'].get('model_path') is not None
            has_model_config_path = self.config['model_config'].get('model_path') is not None

            if skip_training and (has_evaluator_model_path or has_model_config_path):
                # Load model for evaluation
                self._load_model_step(state)
                state["trained_model"] = state["model"]  # Set as trained_model for evaluation
                print("---- Model Loaded Successfully ----")
            elif not skip_training:
                # Train the model
                self._train_model_step(state)
                print("---- Model Training Complete ----")
            else:
                raise ValueError("No model path specified and training is skipped. Cannot proceed without a model.")

            if not self.config['evaluator_config'].get('skip_evaluation', False):
                self._evaluate_model_step(state)
            else:
                print("-- Skipped Evaluation --")
                
            # Save final pipeline summary
            self._save_pipeline_summary(state)
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            # Save error information
            error_info = {
                'error': str(e),
                'step': getattr(self, '_current_step', 'unknown'),
                'traceback': traceback.format_exc()
            }
            self.artifacts_manager.save_artifacts(error_report=error_info)
            raise

    def _load_config(self, config_file):
        with open(config_file, "r") as file:
            print("- Configuration Loaded -")
            config = yaml.safe_load(file)

        home = config.get('home')
        if not home:
            raise ValueError("The 'home' path is missing from the configuration.")

        def join_with_home(path):
            if os.path.isabs(path):
                return path
            return os.path.join(home, path)

        # Update all relevant paths
        artifacts = config.get('artifacts_config', {})
        if 'save_artifacts_to' in artifacts:
            artifacts['save_artifacts_to'] = join_with_home(artifacts['save_artifacts_to'])
        
        processor = config.get('processor_config', {})
        dataset_paths = processor.get('dataset_paths', {})
        for key, value in dataset_paths.items():
            dataset_paths[key] = join_with_home(value)
        
        trainer = config.get('trainer_config', {})
        if 'fine_tune_model_path' in trainer:
            trainer['fine_tune_model_path'] = join_with_home(trainer['fine_tune_model_path'])
        if 'fine_tune_data_path' in trainer:
            trainer['fine_tune_data_path'] = join_with_home(trainer['fine_tune_data_path'])
            
        evaluator = config.get('evaluator_config', {})
        if 'test_data_path' in evaluator:
            evaluator['test_data_path'] = join_with_home(evaluator['test_data_path'])
        if 'model_path' in evaluator:
            evaluator['model_path'] = join_with_home(evaluator['model_path'])

        return config

    def _process_data_step(self, state):
        self._current_step = 'data_processing'
        if self.config['processor_config'].get("skip_processing", True):
            print("-- Skipped Processing Data --")
            return
        
        processing_results = self.data_processor.process()
        state["processing_results"] = processing_results
        
        # Save data processing artifacts if any
        if processing_results:
            self.artifacts_manager.save_artifacts(
                data_processing_results=processing_results
            )

    def _build_model_step(self, state):
        self._current_step = 'model_building'
        if self.config['model_config'].get("skip_building", True):
            print("-- Skipped building architecture --")
            return
        
        model_architecture = self.architecture_manager.get_model()
        state["model"] = model_architecture
        
        # Save model architecture info
        architecture_info = {
            'model_class': model_architecture.__class__.__name__,
            'parameter_count': sum(p.numel() for p in model_architecture.parameters()),
            'trainable_parameters': sum(p.numel() for p in model_architecture.parameters() if p.requires_grad),
            'input_dim': getattr(model_architecture, 'input_dim', None),
            'task': getattr(model_architecture, 'task', None)
        }
        
        self.artifacts_manager.save_artifacts(
            model_architecture_info=architecture_info
        )

    def _load_model_step(self, state):
        self._current_step = 'model_loading'
        model_path = self.config['evaluator_config'].get("model_path")
        if model_path:
            try:
                if not os.path.isabs(model_path):
                    home_path = self.config.get('home', '')
                    full_model_path = os.path.join(home_path, model_path.lstrip('/'))
                else:
                    full_model_path = model_path
                
                print(f"-- Loading model from {full_model_path} --")
                
                import torch
                
                class TorchScriptWrapper(torch.nn.Module):
                    def __init__(self, model, config):
                        super(TorchScriptWrapper, self).__init__()
                        self.model = model
                        
                        # Try to get attributes from the model, fall back to config
                        try:
                            self.task = getattr(model, 'task', None)
                        except:
                            self.task = None
                        
                        try:
                            self.volume_scale_factor = getattr(model, 'volume_scale_factor', None)
                        except:
                            self.volume_scale_factor = None
                        
                        # Use config defaults if attributes not found
                        if self.task is None:
                            self.task = config.get('evaluator_config', {}).get('task', 'IntersectionStatus_IntersectionVolume')
                            print(f"Using task from config: {self.task}")
                        
                        if self.volume_scale_factor is None:
                            self.volume_scale_factor = config.get('evaluator_config', {}).get('volume_scale_factor', 1.0)
                            print(f"Using volume_scale_factor from config: {self.volume_scale_factor}")
                    
                    def predict(self, x):
                        self.model.eval()
                        with torch.no_grad():
                            processed_embeddings = self.model.forward(x) 

                            if self.task == 'IntersectionStatus':
                                return (processed_embeddings > 0.5).int().squeeze()
                            elif self.task == 'IntersectionVolume':
                                return processed_embeddings.squeeze() / self.volume_scale_factor
                            elif self.task == 'IntersectionStatus_IntersectionVolume':
                                cls_prediction = (processed_embeddings[:, 0] > 0.5).int().squeeze()
                                reg_prediction = processed_embeddings[:, 1].squeeze() / self.volume_scale_factor
                                return torch.stack([cls_prediction, reg_prediction], dim=1)         
                            raise ValueError(f"Unknown task: {self.task}")
                    
                    def forward(self, x):
                        """Standard forward method for torch.nn.Module"""
                        return self.model(x)
                    
                    def __call__(self, x):
                        """Allow the wrapper to be called directly like a model"""
                        return self.forward(x)
                    
                    def eval(self):
                        """Forward eval call to underlying model"""
                        self.model.eval()
                        return super().eval()
                    
                    def to(self, device):
                        """Forward device movement to underlying model"""
                        self.model = self.model.to(device)
                        return super().to(device)
                    
                    def float(self):
                        """Forward float conversion to underlying model"""
                        self.model = self.model.float()
                        return super().float()
                    
                    def parameters(self):
                        """Forward parameters call to underlying model"""
                        return self.model.parameters()
            
                loaded_model = torch.jit.load(full_model_path)
                loaded_model_with_predict = TorchScriptWrapper(loaded_model, self.config)
                
                state["model"] = loaded_model_with_predict
                
                # Save model loading info
                model_info = {
                    'loaded_from': full_model_path,
                    'task': loaded_model_with_predict.task,
                    'volume_scale_factor': loaded_model_with_predict.volume_scale_factor
                }
                
                self.artifacts_manager.save_artifacts(
                    loaded_model_info=model_info
                )
                
                return loaded_model_with_predict
                
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        else:
            print(f"No model path specified in {model_path}.")


    def _train_model_step(self, state):
        """Train or load a model based on configuration"""
        self._current_step = 'model_training'
        
        # Check if fine-tuning is enabled
        if self.config['trainer_config'].get('fine_tune_on', False):
            self._fine_tune_model_step(state)
            return
        
        # Check if we should skip training and just load a model
        if self.config['trainer_config'].get('skip_training', False):
            if self.config['evaluator_config'].get('model_path'):
                loaded_model = self._load_model_step(state)
                state["trained_model"] = loaded_model  # Set as trained_model for evaluation
                print(f"---- Model loaded for evaluation from {self.config['evaluator_config']['model_path']} ----")
                return  # Skip the training part entirely
            else:
                raise ValueError("Training is skipped but no model path provided in evaluator_config.")
        
        # Regular training flow
        model = state.get("model", None)
        
        if not model:
            raise ValueError("Model must be provided for training.")

        # Train the model - handle variable return values
        training_results = self.model_trainer.train_and_validate(model)
        
        # Flexible unpacking to handle different return formats
        if isinstance(training_results, tuple):
            if len(training_results) == 3:
                trained_model, training_report, training_metrics = training_results
            elif len(training_results) == 4:
                trained_model, training_report, training_metrics, loss_plot = training_results
                state["loss_plot"] = loss_plot
            elif len(training_results) == 2:
                trained_model, training_report = training_results
                training_metrics = None
            else:
                raise ValueError(f"Unexpected number of return values from train_and_validate: {len(training_results)}")
        else:
            # Single return value
            trained_model = training_results
            training_report = None
            training_metrics = None
        
        # Validate critical components
        if trained_model is None:
            raise ValueError("Training failed. No trained model was returned.")
        
        # Store in state
        state["trained_model"] = trained_model
        state["training_report"] = training_report
        state["training_metrics"] = training_metrics

        # Save artifacts using named parameters for better organization
        artifacts_to_save = {
            'model': trained_model
        }
        
        if training_report is not None:
            artifacts_to_save['training_report'] = training_report
            
        if training_metrics is not None:
            artifacts_to_save['training_metrics'] = training_metrics
            
        if 'loss_plot' in state:
            artifacts_to_save['loss_plot'] = state['loss_plot']

        self.artifacts_manager.save_artifacts(**artifacts_to_save)
        print("---- Training Artifacts Saved ----")

    def _fine_tune_model_step(self, state):
        """Fine-tune a pre-trained model with new data"""
        self._current_step = 'fine_tuning'
        
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
        
        # Fine-tune the model - handle variable return values
        fine_tune_results = self.model_trainer.fine_tune(loaded_model, fine_tune_data)
        
        # Flexible unpacking for fine-tuning results
        if isinstance(fine_tune_results, tuple):
            if len(fine_tune_results) == 3:
                fine_tuned_model, training_report, training_metrics = fine_tune_results
            elif len(fine_tune_results) == 4:
                fine_tuned_model, training_report, training_metrics, loss_plot = fine_tune_results
                state["loss_plot"] = loss_plot
            elif len(fine_tune_results) == 2:
                fine_tuned_model, training_report = fine_tune_results
                training_metrics = None
            else:
                raise ValueError(f"Unexpected number of return values from fine_tune: {len(fine_tune_results)}")
        else:
            fine_tuned_model = fine_tune_results
            training_report = None
            training_metrics = None
        
        if fine_tuned_model is None:
            raise ValueError("Fine-tuning failed. No model was returned.")
        
        # Store in state
        state["trained_model"] = fine_tuned_model
        state["training_report"] = training_report
        state["training_metrics"] = training_metrics
        
        # Save artifacts
        artifacts_to_save = {
            'fine_tuned_model': fine_tuned_model,
            'fine_tune_base_model_path': fine_tune_model_path,
            'fine_tune_data_path': fine_tune_data_path
        }
        
        if training_report is not None:
            artifacts_to_save['fine_tune_training_report'] = training_report
            
        if training_metrics is not None:
            artifacts_to_save['fine_tune_training_metrics'] = training_metrics
            
        if 'loss_plot' in state:
            artifacts_to_save['fine_tune_loss_plot'] = state['loss_plot']

        self.artifacts_manager.save_artifacts(**artifacts_to_save)
        print("---- Fine-tuning Complete ----")

    def _load_model_for_fine_tuning(self, model_path):
        """Load a model specifically for fine-tuning (keeping it trainable)"""
        try:
            if not os.path.isabs(model_path):
                home_path = self.config.get('home', '')
                full_model_path = os.path.join(home_path, model_path.lstrip('/'))
            else:
                full_model_path = model_path
            
            print(f"-- Loading model for fine-tuning from {full_model_path} --")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            
            try:
                loaded_model = torch.jit.load(full_model_path, map_location=device)
                print("Loaded TorchScript model successfully")
            except Exception as e:
                print(f"Failed to load as TorchScript: {e}")
                checkpoint = torch.load(full_model_path, map_location=device)
                raise ValueError("State dict loading not implemented. Please use TorchScript models for fine-tuning.")
            
            fine_tune_model = self._convert_torchscript_to_pytorch(loaded_model)
            fine_tune_model = fine_tune_model.to(device)
            fine_tune_model.train()
            
            for param in fine_tune_model.parameters():
                param.requires_grad = True
                
            return fine_tune_model
            
        except Exception as e:
            raise ValueError(f"Error loading model for fine-tuning: {e}")

    def _convert_torchscript_to_pytorch(self, torchscript_model):
        """Convert a TorchScript model back to a regular PyTorch model for fine-tuning."""
        device = next(torchscript_model.parameters()).device
        print(f"TorchScript model is on device: {device}")
        
        try:
            task = getattr(torchscript_model, 'task', None)
            input_dim = getattr(torchscript_model, 'input_dim', None)
            volume_scale_factor = getattr(torchscript_model, 'volume_scale_factor', None)
            
            print(f"Extracted metadata - Task: {task}, Input dim: {input_dim}, Volume scale: {volume_scale_factor}")
            
        except Exception as e:
            print(f"Could not extract metadata from TorchScript model: {e}")
            task = None
            input_dim = None
            volume_scale_factor = None
        
        if task is None:
            task = self.config['model_config']['common_parameters'].get('task', 'IntersectionStatus_IntersectionVolume')
            print(f"Using default task from config: {task}")
        
        if input_dim is None:
            input_dim = self.config['model_config']['common_parameters'].get('input_dim', 24)
            print(f"Using default input_dim from config: {input_dim}")
        
        if volume_scale_factor is None:
            volume_scale_factor = self.config['model_config']['common_parameters'].get('volume_scale_factor', 1.0)
            print(f"Using default volume_scale_factor from config: {volume_scale_factor}")
        
        self.config['model_config']['common_parameters']['task'] = task
        self.config['model_config']['common_parameters']['input_dim'] = input_dim
        self.config['model_config']['common_parameters']['volume_scale_factor'] = volume_scale_factor
        
        new_model = self.architecture_manager.get_model()
        new_model = new_model.to(device)
        
        try:
            torchscript_state_dict = torchscript_model.state_dict()
            new_model_state_dict = new_model.state_dict()
            
            print("Attempting to map parameters...")
            matched_params = 0
            total_params = len(new_model_state_dict)
            
            for new_key in new_model_state_dict.keys():
                param_found = False
                
                if new_key in torchscript_state_dict:
                    if new_model_state_dict[new_key].shape == torchscript_state_dict[new_key].shape:
                        new_model_state_dict[new_key] = torchscript_state_dict[new_key].to(device)
                        matched_params += 1
                        param_found = True
                
                if not param_found:
                    for ts_key in torchscript_state_dict.keys():
                        clean_ts_key = ts_key.replace('original_name.', '').replace('_c.', '.')
                        
                        if (new_key in clean_ts_key or clean_ts_key.endswith(new_key) or 
                            new_key.split('.')[-1] == clean_ts_key.split('.')[-1]):
                            
                            if new_model_state_dict[new_key].shape == torchscript_state_dict[ts_key].shape:
                                new_model_state_dict[new_key] = torchscript_state_dict[ts_key].to(device)
                                matched_params += 1
                                param_found = True
                                break
                
                if not param_found:
                    print(f"Warning: Could not find matching parameter for {new_key}")
            
            print(f"Successfully mapped {matched_params}/{total_params} parameters")
            new_model.load_state_dict(new_model_state_dict, strict=False)
            print("Successfully copied weights from TorchScript to PyTorch model")
            
        except Exception as e:
            print(f"Warning: Could not copy weights: {e}")
            print("Fine-tuning will start from randomly initialized weights")
        
        return new_model

    def _prepare_fine_tune_data(self, fine_tune_data_path):
        """Prepare the fine-tuning dataset"""
        try:
            if not os.path.isabs(fine_tune_data_path):
                home_path = self.config.get('home', '')
                full_data_path = os.path.join(home_path, fine_tune_data_path.lstrip('/'))
            else:
                full_data_path = fine_tune_data_path
            
            print(f"-- Loading fine-tune data from {full_data_path} --")
            
            import pandas as pd
            fine_tune_df = pd.read_csv(full_data_path)
            
            return fine_tune_df
            
        except Exception as e:
            raise ValueError(f"Error loading fine-tune data: {e}")

    def _evaluate_model_step(self, state):
        """Evaluate the model based on the configuration"""
        self._current_step = 'evaluation'
        
        trained_model = state.get("trained_model")
        if trained_model is None:
            raise ValueError("Trained model is required for evaluation but was not provided.")
        
        training_metrics = state.get("training_metrics")
        
        try:
            evaluation_report = self.evaluator.evaluate_model(trained_model, training_metrics)
            state["evaluation_report"] = evaluation_report
            
            # Save evaluation artifacts
            self.artifacts_manager.save_artifacts(
                evaluation_report=evaluation_report
            )
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            # Save evaluation error
            evaluation_error = {
                'error': str(e),
                'model_available': trained_model is not None,
                'training_metrics_available': training_metrics is not None
            }
            self.artifacts_manager.save_artifacts(
                evaluation_error=evaluation_error
            )
            raise

    def _run_data_analysis_step(self, state):
        """Run data analysis notebook and save HTML report only"""
        self._current_step = 'data_analysis'
        
        try:
            notebook_path = self.config.get('artifacts_config', {}).get(
                'data_analysis_notebook', 
                'notebooks/data_analysis.ipynb'
            )
            
            print(f"---- Generating Data Analysis Report ----")
            analysis_results = self.artifacts_manager.run_data_analysis(notebook_path)
            
            if 'error' in analysis_results:
                print(f"Data analysis failed: {analysis_results['error']}")
                state['data_analysis_error'] = analysis_results['error']
            else:
                print(f"---- Data Analysis Report Generated ----")
                state['data_analysis_results'] = analysis_results
                
        except Exception as e:
            error_msg = f"Error in data analysis step: {e}"
            print(error_msg)
            state['data_analysis_error'] = error_msg

    def _save_pipeline_summary(self, state):
        """Save a comprehensive pipeline summary"""
        summary = {
            'pipeline_completed': True,
            'experiment_id': self.artifacts_manager.context.experiment_id,
            'timestamp': self.artifacts_manager.context.timestamp.isoformat(),
            'config_hash': self.artifacts_manager.context.config_hash,
            'git_commit': self.artifacts_manager.context.git_commit,
            'system_info': self.artifacts_manager.context.system_info,
            'steps_completed': {
                'data_processing': 'processing_results' in state,
                'model_building': 'model' in state,
                'training': 'trained_model' in state,
                'evaluation': 'evaluation_report' in state
            },
            'artifacts_saved': True
        }
        
        # Add final metrics if available
        if 'training_metrics' in state and state['training_metrics']:
            summary['final_training_metrics'] = state['training_metrics']
            
        if 'evaluation_report' in state and state['evaluation_report']:
            summary['evaluation_summary'] = state['evaluation_report']
        
        self.artifacts_manager.save_artifacts(
            pipeline_summary=summary
        )
        
        print(f"\n=== Pipeline Complete ===")
        print(f"Experiment ID: {summary['experiment_id']}")
        print(f"All artifacts saved to: {self.artifacts_manager.artifacts_path}")