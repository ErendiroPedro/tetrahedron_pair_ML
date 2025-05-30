import yaml
import os
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

        # Train the model
        if not self.config['model_config'].get('skip_training', False):
            self._train_model_step(state)
            print("---- Model Training Complete ----")
        # Load the model if specified
        elif self.config['model_config'].get('model_path'):
            self._load_model_step(state)
            print("---- Model Loaded Successfully ----")
            # Continue with evaluation if loading was successful
        else:
            print("-- Skipped Model Training --")


        # Evaluate the model
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

                            if self.task == 'binary_classification':
                                return (processed_embeddings > 0.5).int().squeeze() # Using binary cross-entropy with logits, sgimoid is applied in the loss function
                            elif self.task == 'regression':
                                return processed_embeddings.squeeze() / self.volume_scale_factor
                            elif self.task == 'classification_and_regression':
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

        model = state.get("model", None)

        if model is None and self.config['trainer_config']['skip_training'] and self.config['evaluator_config']['model_path']:
                # If a model path is specified, load the model
                loaded_model = self._load_model_step(state)
                state["model"] = loaded_model
                print(f"---- Will train model with {self.config['evaluator_config']['model_path']} ----")

        else:
            # If no model is provided, build a new one
            self._build_model_step(state)
            model = state.get("model", None)
            print("---- Model Built Successfully ----")
            

        assert model, "Model must be provided for training."

        # Train the model
        trained_model, training_report = self.model_trainer.train_and_validate(state["model"])
        if trained_model is None:
            raise ValueError("Training failed. No trained model was returned.")
        if training_report is None:
            raise ValueError("Training report is missing. Training might have failed.")
        
        state["trained_model"] = trained_model
        state["training_report"] = training_report

        # Save artifacts
        self.artifacts_manager.save_artifacts(trained_model, training_report)

        print("---- Training Complete ----")

    def _evaluate_model_step(self, state):
        """Evaluate the model based on the configuration"""
        trained_model = state.get("trained_model")
        if trained_model is None:
            raise ValueError("Trained model is required for evaluation but was not provided.")
        evaluation_report = self.evaluator.evaluate(trained_model)
        self.artifacts_manager.save_artifacts(evaluation_report)
        state["evaluation_report"] = evaluation_report