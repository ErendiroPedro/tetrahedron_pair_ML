import torch
import traceback
import yaml
from src.artifacts import Artifacts
from src.data_processing import CDataProcessor
from src.evaluator import evaluate
from src.models import TorchScriptModel, build_model
from src.training import train

class CPipelineOrchestrator:
    def __init__(self, config_file="config/config.yaml"):
        # 1. Load Config (paths in it are relative to the working directory)
        with open(config_file) as f:
            self.config = yaml.safe_load(f)

        # 2. Initialize Components
        self.artifacts = Artifacts(self.config)
        self.data_processor = CDataProcessor(self.config["processor_config"])

        self.state = {}
        self.id = self.artifacts.id
        self._current_step = "init"

    def run(self):
        print(f"\n=== Starting Experiment {self.id} ===")
        try:
            # Step 1: Processing
            if not self.config['processor_config'].get('skip_processing', True):
                self._run_step('processing', self._process_data_step)
            
            # Step 2: Architecture
            if not self.config['model_config'].get('skip_building', True):
                self._run_step('building', self._build_model_step)

            # Step 3: Training / Loading Logic
            self._run_step('model_acquisition', self._acquire_model_step)

            # Step 4: Evaluation
            if not self.config['evaluator_config'].get('skip_evaluation', False):
                self._run_step('evaluation', self._evaluate_model_step)

            print(f"\n=== Experiment {self.id} Complete ===\nArtifacts: {self.artifacts.path}")

        except Exception as e:
            self._handle_failure(e)

    def _run_step(self, name, func):
        self._current_step = name
        func()
        print(f"---- {name.replace('_', ' ').title()} Complete ----")

    def _process_data_step(self):
        self.data_processor.process()

    def _build_model_step(self):
        self.state["model"] = build_model(self.config["model_config"])

    def _acquire_model_step(self):
        """Either load an existing model (skip_training) or train the one built in the previous step."""
        if self.config['trainer_config'].get('skip_training', False):
            eval_cfg = self.config['evaluator_config']
            path = eval_cfg.get('model_path') or self.config['model_config'].get('model_path')
            if not path:
                raise ValueError("skip_training=True but no model_path provided.")
            print(f"-- Loading model from {path} --")
            self.state["trained_model"] = TorchScriptModel(
                torch.jit.load(path), eval_cfg['task'], eval_cfg['volume_scale_factor'])
        else:
            if "model" not in self.state:
                raise ValueError("Model architecture not built. Check 'skip_building' config.")
            results = train(self.state["model"], self.config['trainer_config'])
            self.state.update(results)
            self.artifacts.save_model(results["trained_model"])
            if results["loss_plot"]:
                self.artifacts.save_plot(results["loss_plot"])

    def _evaluate_model_step(self):
        model = self.state.get("trained_model") or self.state.get("model")
        report = evaluate(model, self.config["evaluator_config"], self.config["processor_config"], self.state.get("training_metrics"))
        self.artifacts.save_json("evaluation_report", report)

    def _handle_failure(self, e):
        print(f"Experiment {self.id} failed at [{self._current_step}]: {e}")
        self.artifacts.save_json("error", {'error': str(e), 'step': self._current_step, 'traceback': traceback.format_exc()})
        raise e
