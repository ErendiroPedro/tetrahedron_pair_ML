import yaml
from src.DataProcessor import DataProcessor
from src.ModelBuilder import ModelBuilder
from src.ModelTrainer import ModelTrainer
from src.ArtifactsManager import ArtifactsManager
from src.evaluator.Evaluator import Evaluator

class PipelineOrchestrator:
    def __init__(self, config_file="config.yaml"):
        self.config = self._load_config(config_file)
        self.artifacts_manager = ArtifactsManager(self.config)
        self.data_processor = DataProcessor(self.config["processor_config"])
        self.model_builder = ModelBuilder(self.config["model_config"])
        self.model_trainer = ModelTrainer(self.config["trainer_config"])
        self.evaluator = Evaluator(self.config["evaluator_config"])

    def _load_config(self, config_file):
        with open(config_file, "r") as file:
            print("- Configuration Loaded -")
            return yaml.safe_load(file)
    
    def run(self):
        print("- Running Pipeline -")
        state = {}
        self._process_data_step(state)
        self._build_model_step(state)
        self._train_model_step(state)
        self._evaluate_model_step(state)
        print("- Pipeline Done -")

    def _process_data_step(self, state):
        if self.config.get("skip_processing", False):
            print("-- Skipped Processing Data --")
            return
        self.data_processor.process()

    def _build_model_step(self, state):
        if self.config.get("skip_building", False):
            print("-- Skipped building architecture --")
            return
        model_architecture = self.model_builder.build()
        state["model_architecture"] = model_architecture

    def _train_model_step(self, state):
        if self.config.get("skip_training", False):
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
        if self.config.get("skip_evaluation", False):
            print("-- Skipped Evaluation --")
            return
        trained_model = state.get("trained_model")
        if trained_model is None:
            raise ValueError("Trained model is required for evaluation but was not provided.")
        evaluation_report = self.evaluator.evaluate(trained_model)
        self.artifacts_manager.save_artifacts(evaluation_report)
        state["evaluation_report"] = evaluation_report

PipelineOrchestrator().run()