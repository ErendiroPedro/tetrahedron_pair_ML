import base64
import json

import numpy as np
import torch
import yaml

from src.artifacts import Artifacts
from src.models import build_model


def test_the_experiment_folder_is_named_after_the_run_and_keeps_the_config(config):
    artifacts = Artifacts(config)

    assert artifacts.path.name == (f"id{artifacts.experiment_id}_tetrahedronpairnet_IntersectionStatus_IntersectionVolume"
                                   "_0k-train_0k-val_vol0.0001-0.01_e2")
    assert yaml.safe_load((artifacts.path / "config.yaml").read_text()) == config


def test_save_json_writes_numpy_and_torch_values_as_plain_json(config):
    artifacts = Artifacts(config)
    artifacts.save_json("report", {"array": np.arange(3), "scalar": np.float32(1.5), "tensor": torch.tensor([1, 2]), "text": "x"})
    assert json.loads((artifacts.path / "report.json").read_text()) == {"array": [0, 1, 2], "scalar": 1.5, "tensor": [1, 2], "text": "x"}


def test_save_plot_decodes_the_base64_image(config):
    artifacts = Artifacts(config)
    png = b"\x89PNG\r\n\x1a\nnot really an image"
    artifacts.save_plot(base64.b64encode(png).decode())
    assert (artifacts.path / "loss_plot.png").read_bytes() == png


def test_save_model_writes_a_traced_model_a_loadable_state_dict_and_a_summary(config):
    torch.manual_seed(0)
    model = build_model(config["model_config"])
    artifacts = Artifacts(config)

    artifacts.save_model(model)

    folder = artifacts.path / "model"
    assert sorted(path.name for path in folder.iterdir()) == ["model.pt", "state_dict.pth", "summary.txt"]
    batch = torch.randn(11, 24)  # the trace was made with one row; any batch size must work
    torch.testing.assert_close(torch.jit.load(folder / "model.pt")(batch), model.eval()(batch))
    restored = build_model(config["model_config"])
    restored.load_state_dict(torch.load(folder / "state_dict.pth"))
    torch.testing.assert_close(restored.eval()(batch), model(batch))
    assert "TetrahedronPairNet" in (folder / "summary.txt").read_text()
