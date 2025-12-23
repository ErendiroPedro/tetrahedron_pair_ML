import yaml
import os
import torch
import traceback
import pandas as pd
from typing import Dict
from src.CDataProcessor import CDataProcessor
from src.CArchitectureManager import CArchitectureManager, ModelUtility
from src.CModelTrainer import CModelTrainer
from src.CArtifactsManager import CArtifactsManager
from src.evaluator.CEvaluator import CEvaluator

# --- HELPER UTILITIES ---

class ConfigResolver:
    """Handles recursive path resolution based on a 'home' directory."""
    @staticmethod
    def resolve(config: Dict) -> Dict:
        home = config.get('home')
        if not home:
            raise ValueError("The 'home' path is missing from the configuration.")

        def _walk(node):
            if isinstance(node, dict):
                return {k: _walk(v) for k, v in node.items()}
            if isinstance(node, str) and (node.startswith('/') or './' in node or any(ext in node for ext in ['.csv', '.pt', '.pth', '.ipynb'])):
                return node if os.path.isabs(node) else os.path.join(home, node.lstrip('/'))
            return node
        
        return _walk(config)

class CPipelineOrchestrator:
    def __init__(self, config_file="config/config.yaml"):
        # 1. Load and Clean Config
        with open(config_file, "r") as f:
            raw_config = yaml.safe_load(f)
        self.config = ConfigResolver.resolve(raw_config)
        
        # 2. Initialize Components
        self.artifacts_manager = CArtifactsManager(self.config)
        self.architecture_manager = CArchitectureManager(self.config["model_config"])
        self.data_processor = CDataProcessor(self.config["processor_config"])
        self.model_trainer = CModelTrainer(self.config["trainer_config"])
        self.evaluator = CEvaluator(self.config["evaluator_config"], self.config["processor_config"])
        
        self.state = {}
        self._current_step = "init"

    def run(self):
        """Orchestrates the pipeline using a clean step-based approach."""
        print(f"\n=== Starting Pipeline Experiment ===")
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

            print(f"\n=== Pipeline Complete ===\nArtifacts: {self.artifacts_manager.artifacts_path}")

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
        model = self.architecture_manager.get_model()
        self.state["model"] = model
        meta = {
            'model_class': model.__class__.__name__,
            'total_params': sum(p.numel() for p in model.parameters())
        }
        self.artifacts_manager.save_artifacts(model_architecture_info=meta)

    def _acquire_model_step(self):
        """Determines whether to train, fine-tune, or load a model."""
        trainer_cfg = self.config['trainer_config']
        eval_cfg = self.config['evaluator_config']

        # Case A: Fine-tuning
        if trainer_cfg.get('fine_tune_on', False):
            self._fine_tune_flow()
        
        # Case B: Skip Training (Load existing)
        elif trainer_cfg.get('skip_training', False):
            path = eval_cfg.get('model_path') or self.config['model_config'].get('model_path')
            if not path:
                raise ValueError("skip_training=True but no model_path provided.")
            self._load_model_for_inference(path)
        
        # Case C: Standard Training
        else:
            if "model" not in self.state:
                raise ValueError("Model architecture not built. Check 'skip_building' config.")
            results = self.model_trainer.train_and_validate(self.state["model"])
            self._unpack_trainer_results(results)

    def _load_model_for_inference(self, path):
        print(f"-- Loading model from {path} --")
        raw_model = torch.jit.load(path)
        wrapped = ModelUtility.TorchScriptWrapper(raw_model, self.config)
        self.state["trained_model"] = wrapped
        self.artifacts_manager.save_artifacts(loaded_model_info={'path': path})

    def _fine_tune_flow(self):
        t_cfg = self.config['trainer_config']
        print(f"-- Fine-tuning from {t_cfg.get('fine_tune_model_path')} --")
        
        # 1. Load weights into new architecture
        ts_model = torch.jit.load(t_cfg['fine_tune_model_path'])
        base_model = self.architecture_manager.get_model()
        fine_tune_model = ModelUtility.map_weights(ts_model, base_model)
        
        # 2. Load data and train
        data = pd.read_csv(t_cfg['fine_tune_data_path'])
        results = self.model_trainer.fine_tune(fine_tune_model, data)
        self._unpack_trainer_results(results, is_finetune=True)

    def _unpack_trainer_results(self, results, is_finetune=False):
        """Normalizes the messy return values from trainer/fine-tuner."""
        keys = ["trained_model", "training_report", "training_metrics", "loss_plot"]
        if isinstance(results, tuple):
            res_dict = {keys[i]: results[i] for i in range(len(results))}
        else:
            res_dict = {"trained_model": results}
        
        self.state.update(res_dict)
        prefix = "fine_tune_" if is_finetune else ""
        labeled_artifacts = {f"{prefix}{k}": v for k, v in res_dict.items() if v is not None}
        self.artifacts_manager.save_artifacts(**labeled_artifacts)

    def _evaluate_model_step(self):
        model = self.state.get("trained_model") or self.state.get("model")
        report = self.evaluator.evaluate(model, self.state.get("training_metrics"))
        self.artifacts_manager.save_artifacts(evaluation_report=report)

    def _handle_failure(self, e):
        print(f"Pipeline failed at [{self._current_step}]: {e}")
        error_info = {
            'error': str(e),
            'step': self._current_step,
            'traceback': traceback.format_exc()
        }
        self.artifacts_manager.save_artifacts(error_report=error_info)
        raise e