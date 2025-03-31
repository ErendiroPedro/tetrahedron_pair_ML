import yaml
import os
from src.CDataProcessor import CDataProcessor
from src.CModelBuilder import CModelBuilder
from src.CModelTrainer import CModelTrainer
from src.CArtifactsManager import CArtifactsManager
from src.evaluator.CEvaluator import CEvaluator

class CPipelineOrchestrator:
    def __init__(self, config_file="config/config.yaml"):
        self.config = self._load_config(config_file)
        self.artifacts_manager = CArtifactsManager(self.config)
        self.data_processor = CDataProcessor(self.config["processor_config"])
        self.model_builder = CModelBuilder(self.config["model_config"])
        self.model_trainer = CModelTrainer(self.config["trainer_config"])
        self.evaluator = CEvaluator(self.config["evaluator_config"])

    def run(self):
        print("- Running Pipeline -")
        state = {}
        self._process_data_step(state)
        self._build_model_step(state)
        self._train_model_step(state)
        self._evaluate_model_step(state)
        print("- Pipeline Done -")
        return state
    
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
        model_architecture = self.model_builder.build()
        state["model_architecture"] = model_architecture

    def _train_model_step(self, state):
        if self.config['trainer_config'].get("skip_training", True):
            print("-- Skipped training model --")
            return
        model_architecture = state.get("model_architecture")
        if model_architecture is None:
            raise ValueError("Model architecture is required for training but was not provided.")
        trained_model, training_report = self.model_trainer.train_and_validate(model_architecture)
        self.artifacts_manager.save_artifacts(trained_model, training_report)
        state["trained_model"] = trained_model
        state["training_report"] = training_report

    def _evaluate_model_step(self, state):
        if self.config['evaluator_config'].get("skip_evaluation", True):
            print("-- Skipped Evaluation --")
            return
        trained_model = state.get("trained_model")
        if trained_model is None:
            raise ValueError("Trained model is required for evaluation but was not provided.")
        evaluation_report = self.evaluator.evaluate(trained_model)
        self.artifacts_manager.save_artifacts(evaluation_report)
        state["evaluation_report"] = evaluation_report