import traceback

import yaml

from src.artifacts import Artifacts
from src.data_processing import DataProcessor
from src.evaluation import evaluate
from src.models import LoadedModel, build_model, load_exported_model
from src.pipeline.config_validation import enabled_steps, validate_config
from src.training import train


class PipelineOrchestrator:
    """Runs data processing, model building, training (or loading) and evaluation for one config file."""

    def __init__(self, config_file="config/config.yaml"):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)  # paths in it are relative to the working directory
        validate_config(self.config)
        self.steps = enabled_steps(self.config)

        self.artifacts = Artifacts(self.config)
        self.experiment_id = self.artifacts.experiment_id
        self.state = {}
        self._current_step = "init"

    def run(self):
        print(f"\n=== Starting Experiment {self.experiment_id} ===")
        try:
            if self.steps["data_processing"]:
                self._run_step("data_processing", self._process_data_step)
            if self.steps["model_building"]:
                self._run_step("model_building", self._build_model_step)
            self._run_step("model_acquisition", self._acquire_model_step)
            if self.steps["evaluation"]:
                self._run_step("evaluation", self._evaluate_model_step)
            print(f"\n=== Experiment {self.experiment_id} Complete ===\nArtifacts: {self.artifacts.path}")
        except Exception as error:
            self._record_failure(error)
            raise

    def _run_step(self, name, step):
        self._current_step = name
        step()
        print(f"---- {name.replace('_', ' ').title()} Complete ----")

    def _process_data_step(self):
        DataProcessor(self.config["processor_config"]).process()

    def _build_model_step(self):
        self.state["model"] = build_model(self.config["model_config"])

    def _acquire_model_step(self):
        """Trains the model built in the previous step, or loads one when training is skipped."""
        if self.steps["training"]:
            results = train(self.state["model"], self.config["trainer_config"])
            self.state.update(results)
            self.artifacts.save_model(results["trained_model"])
            self.artifacts.save_plot(results["loss_plot"])
        else:
            evaluator_config = self.config["evaluator_config"]
            path = evaluator_config.get("model_path") or self.config["model_config"].get("model_path")
            print(f"-- Loading model from {path} --")
            self.state["trained_model"] = LoadedModel(
                load_exported_model(path), evaluator_config["task"], evaluator_config["volume_scale_factor"])

    def _evaluate_model_step(self):
        model = self.state.get("trained_model") or self.state.get("model")
        report = evaluate(model, self.config["evaluator_config"], self.config["processor_config"],
                          self.state.get("training_metrics"))
        self.artifacts.save_json("evaluation_report", report)

    def _record_failure(self, error):
        print(f"Experiment {self.experiment_id} failed at [{self._current_step}]: {error}")
        self.artifacts.save_json("error", {
            "error": str(error),
            "step": self._current_step,
            "traceback": traceback.format_exc(),
        })
