import copy

import pytest
import torch

from src.models import TorchScriptModel, build_model
from src.models.networks import to_predictions

TASKS = [("IntersectionStatus", 1), ("IntersectionVolume", 1), ("IntersectionStatus_IntersectionVolume", 2)]


def model_config_for(config, architecture="tetrahedronpairnet", task="IntersectionStatus_IntersectionVolume"):
    model_config = copy.deepcopy(config["model_config"])
    model_config["architecture"]["use_model"] = architecture
    model_config["common_parameters"]["task"] = task
    return model_config


@pytest.mark.parametrize("architecture", ["mlp", "tetrahedronpairnet"])
@pytest.mark.parametrize("task, output_columns", TASKS)
def test_forward_and_predict_shapes(config, architecture, task, output_columns):
    model = build_model(model_config_for(config, architecture, task))
    batch = torch.randn(7, 24)

    assert model.input_dim == 24, "input_dim is the processed CSV width minus the two labels"
    assert model(batch).shape == (7, output_columns)
    assert model.predict(batch).shape == ((7,) if output_columns == 1 else (7, 2))


def test_to_predictions_turns_logits_into_status_and_unscales_volume():
    logits_and_volumes = torch.tensor([[2.0, 3000.0], [-1.0, 500.0]])
    combined = to_predictions(logits_and_volumes, "IntersectionStatus_IntersectionVolume", 1000.0)
    assert combined.tolist() == [[1.0, 3.0], [0.0, 0.5]]
    assert to_predictions(logits_and_volumes[:, :1], "IntersectionStatus", 1000.0).tolist() == [1, 0]
    assert to_predictions(logits_and_volumes[:, 1:], "IntersectionVolume", 1000.0).tolist() == [3.0, 0.5]


def test_to_predictions_rejects_unknown_tasks():
    with pytest.raises(ValueError, match="Unknown task"):
        to_predictions(torch.zeros(1, 1), "Other", 1.0)


def test_the_same_seed_gives_the_same_initial_weights_and_another_seed_does_not(config):
    def initial_weights(seed):
        torch.manual_seed(seed)
        return list(build_model(model_config_for(config)).parameters())

    first, same_seed, other_seed = initial_weights(0), initial_weights(0), initial_weights(1)
    assert all(torch.equal(a, b) for a, b in zip(first, same_seed))
    assert not all(torch.equal(a, b) for a, b in zip(first, other_seed))


def test_tetrahedronpairnet_rejects_inputs_that_are_not_one_or_two_tetrahedra(config):
    model = build_model(model_config_for(config))
    with pytest.raises(ValueError, match="12 or 24"):
        model(torch.randn(3, 18))


def test_torchscript_wrapper_predicts_like_the_original_model(config):
    model = build_model(model_config_for(config)).eval()
    traced = torch.jit.trace(model, torch.randn(1, 24))
    wrapper = TorchScriptModel(traced, "IntersectionStatus_IntersectionVolume", 1000.0)
    batch = torch.randn(9, 24)  # a different batch size than the one traced
    torch.testing.assert_close(wrapper.predict(batch), model.predict(batch))
