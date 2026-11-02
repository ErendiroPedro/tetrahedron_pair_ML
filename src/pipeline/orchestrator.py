import torch
import traceback
from src.artifacts import CArtifactsManager
from src.data_processing import CDataProcessor
from src.evaluator.CEvaluator import CEvaluator
from src.models import TorchScriptModel, build_model
from src.training import train
from src.utils.file_management import load_config

class CPipelineOrchestrator:
    def __init__(self, config_file="config/config.yaml"):
        # 1. Load and Clean Config
        self.config = load_config(config_file)

        # 2. Initialize Components
        self.artifacts_manager = CArtifactsManager(self.config)
        self.data_processor = CDataProcessor(self.config["processor_config"])
        self.evaluator = CEvaluator(self.config["evaluator_config"], self.config["processor_config"])
        
        self.state = {}
        self.id = self.artifacts_manager.context.experiment_id
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

            print(f"\n=== Experiment {self.id} Complete ===\nArtifacts: {self.artifacts_manager.artifacts_path}")

        except Exception as e:
            self._handle_failure(e)

    def _run_step(self, name, func):
        self._current_step = name
        func()
        print(f"---- {name.replace('_', ' ').title()} Complete ----")

    def _process_data_step(self):
        results = self.data_processor.process()
        self.state["processing_results"] = results
        self.artifacts_manager.save_artifacts(data_processing_results=results)

    def _build_model_step(self):
        model = build_model(self.config["model_config"])
        self.state["model"] = model
        meta = {
            'model_class': model.__class__.__name__,
            'total_params': sum(p.numel() for p in model.parameters())
        }
        self.artifacts_manager.save_artifacts(model_architecture_info=meta)

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
            self.artifacts_manager.save_artifacts(loaded_model_info={'path': path})
        else:
            if "model" not in self.state:
                raise ValueError("Model architecture not built. Check 'skip_building' config.")
            results = train(self.state["model"], self.config['trainer_config'])
            self.state.update(results)
            self.artifacts_manager.save_artifacts(**{k: v for k, v in results.items() if v is not None})

    def _evaluate_model_step(self):
        model = self.state.get("trained_model") or self.state.get("model")
        report = self.evaluator.evaluate(model, self.state.get("training_metrics"))
        self.artifacts_manager.save_artifacts(evaluation_report=report)

    def _handle_failure(self, e):
        print(f"Experiment {self.id} failed at [{self._current_step}]: {e}")
        error_info = {
            'error': str(e),
            'step': self._current_step,
            'traceback': traceback.format_exc()
        }
        self.artifacts_manager.save_artifacts(error_report=error_info)
        raise e