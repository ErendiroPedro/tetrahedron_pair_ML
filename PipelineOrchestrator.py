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
        if self.config["skip_processing"]:
            print("-- Skipped Processing Data --")
        else:
            self.data_processor.process()
        
        if self.config["skip_building"]:
            print("-- Skipped building architecture --")
        else:
            model_architecture = self.model_builder.build()
        
        if self.config["skip_training"]:
            print("-- Skipped training model --")
        else:
            trained_model, training_report = self.model_trainer.train_and_validate(model_architecture)   
            self.artifacts_manager.save_artifacts(trained_model, training_report)
        if self.config.get('skip_evaluation', False):
            print("-- Skipped Evaluation --")
        else:
            evaluation_report = self.evaluator.evaluate(trained_model) 
            self.artifacts_manager.save_artifacts(evaluation_report
                                                  ) 
        print("- Pipeline Done -")

PipelineOrchestrator().run()