import base64
import datetime
import json
from pathlib import Path

import torch
import yaml

from src.models import export_model


def _experiment_label(config):
    volume_range = config["processor_config"].get("volume_binning", {}).get("volume_range")
    volume = f"vol{'-'.join(map(str, volume_range))}" if volume_range else "no-vol-filter"
    return (
        f"{config['model_config']['architecture']['use_model']}_{config['model_config']['common_parameters']['task']}_"
        f"{config['processor_config']['num_train_samples'] // 1000}k-train_"
        f"{config['processor_config']['num_val_samples'] // 1000}k-val_{volume}_e{config['trainer_config']['epochs']}"
    )


def _json_default(obj):
    return obj.tolist() if hasattr(obj, "tolist") else str(obj)  # numpy arrays/scalars, torch tensors


class Artifacts:
    """One folder per experiment: config.yaml, loss_plot.png, evaluation_report.json and model/."""

    def __init__(self, config):
        self.experiment_id = datetime.datetime.now().strftime("%m%d_%H%M%S")
        root = Path(config.get("artifacts_config", {}).get("save_artifacts_to", "artifacts"))
        self.path = root / f"id{self.experiment_id}_{_experiment_label(config)}"
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def save_json(self, name, data):
        with open(self.path / f"{name}.json", "w") as f:
            json.dump(data, f, indent=2, default=_json_default)

    def save_plot(self, base64_png):
        (self.path / "loss_plot.png").write_bytes(base64.b64decode(base64_png))

    def save_model(self, model):
        """The exported program (what skip_training reloads), plus the state dict and a readable summary."""
        folder = self.path / "model"
        folder.mkdir(exist_ok=True)
        torch.save(model.state_dict(), folder / "state_dict.pth")
        param_count = sum(p.numel() for p in model.parameters())
        (folder / "summary.txt").write_text(f"{type(model).__name__}: {param_count:,} parameters\n\n{model}\n")
        export_model(model, folder / "model.pt2")
