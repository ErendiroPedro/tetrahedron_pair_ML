import base64
import math

import pytest
import torch
import torch.nn.functional as F

from src.models import build_model
from src.training import train
from src.training import trainer as trainer_module
from src.training.losses import EPSILON, LOSS_BY_TASK, rmsle, status_and_volume_loss


def test_there_is_one_loss_per_task():
    assert set(LOSS_BY_TASK) == {"IntersectionStatus", "IntersectionVolume", "IntersectionStatus_IntersectionVolume"}


def test_rmsle_is_zero_for_equal_values_and_the_log_ratio_otherwise():
    values = torch.tensor([1.0, 2.0, 3.0])
    assert rmsle(values, values).item() == pytest.approx(math.sqrt(EPSILON))
    assert rmsle(torch.tensor([1.0]), torch.tensor([3.0])).item() == pytest.approx(math.log(2), rel=1e-6)


def test_status_and_volume_loss_weights_both_parts_equally():
    predictions = torch.tensor([[0.5, 1.0], [-0.3, 2.0]])
    targets = torch.tensor([[1.0, 1.5], [0.0, 2.5]])
    expected = 0.5 * rmsle(predictions[:, 1], targets[:, 1]) + 0.5 * F.binary_cross_entropy_with_logits(predictions[:, 0], targets[:, 0])
    assert status_and_volume_loss(predictions, targets).item() == pytest.approx(expected.item())


def test_train_returns_the_trained_model_a_finite_best_loss_and_a_png_plot(config):
    model = build_model(config["model_config"])
    initial = [p.detach().clone() for p in model.parameters()]

    results = train(model, config["trainer_config"])

    assert set(results) == {"trained_model", "training_metrics", "loss_plot"}
    assert math.isfinite(results["training_metrics"]["final_val_loss"])
    assert base64.b64decode(results["loss_plot"]).startswith(b"\x89PNG")
    assert any(not torch.equal(before, after) for before, after in zip(initial, results["trained_model"].parameters()))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_train_fails_loudly_when_the_loss_is_not_finite(config, monkeypatch, bad_value):
    task = config["trainer_config"]["task"]
    monkeypatch.setitem(trainer_module.LOSS_BY_TASK, task, lambda predictions, targets: predictions.sum() * bad_value)

    with pytest.raises(FloatingPointError, match="epoch 1"):
        train(build_model(config["model_config"]), config["trainer_config"])
