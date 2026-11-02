import pytest
import torch

from src.pipeline.config_validation import enabled_steps, validate_config


def set_path(config, dotted_path, value):
    *parents, key = dotted_path.split(".")
    for parent in parents:
        config = config[parent]
    config[key] = value


def error_message(config):
    with pytest.raises(ValueError) as error:
        validate_config(config)
    return str(error.value)


def test_a_valid_config_passes(config):
    validate_config(config)


@pytest.mark.parametrize("path, value, expected", [
    ("trainer_config.epochs", 0, "trainer_config.epochs must be an integer >= 1"),
    ("trainer_config.learning_rate", -1, "learning_rate must be a number > 0"),
    ("trainer_config.task", "IntersectionVolume", "the task must be identical"),
    ("trainer_config.processed_data_path", "elsewhere", "processed data path must be identical"),
    ("model_config.architecture.use_model", "transformer", "use_model must be one of"),
    ("model_config.common_parameters.task", "Nope", "model_config.common_parameters.task must be one of"),
    ("model_config.common_parameters.activation_function", "sigmoid", "activation_function must be one of"),
    ("model_config.common_parameters.dropout_rate", 1.0, "dropout_rate must be a number in [0, 1)"),
    ("model_config.architecture.tetrahedronpairnet.per_vertex_layers", [], "per_vertex_layers must be a non-empty list"),
    ("model_config.architecture.tetrahedronpairnet.tetrahedra_aggregation_function", "median", "tetrahedra_aggregation_function must be one of"),
    ("processor_config.dataset_paths.raw_data", "/nonexistent", "raw_data must be an existing directory"),
    ("processor_config.intersection_distributions.no_intersection", 5, "must sum to 100"),
    ("processor_config.intersection_distributions.cube_intersection", 1, "unknown types"),
    ("processor_config.volume_binning.volume_range", [0, 1e-2], "volume_range must be [low, high] with 0 < low < high"),
    ("processor_config.transformations", ["rotate"], "transformations"),
    ("processor_config.volume_scale_factor", 10.0, "volume_scale_factor must be identical"),
    ("evaluator_config.volume_range", [0, 0.5], "volume_range must be [low, high] with 0 <= low < high <="),
    ("evaluator_config.evaluation_n_bins", 1, "evaluation_n_bins must be an integer >= 2"),
    ("evaluator_config.test_data_path", "/nonexistent", "test_data_path must be an existing directory"),
])
def test_an_invalid_value_is_reported_with_its_config_key(config, path, value, expected):
    set_path(config, path, value)
    assert expected in error_message(config)


def test_every_problem_is_reported_at_once(config):
    config["trainer_config"]["epochs"] = 0
    config["evaluator_config"]["evaluation_n_bins"] = 1
    config["model_config"]["common_parameters"]["activation_function"] = "sigmoid"
    message = error_message(config)
    assert all(part in message for part in ("trainer_config.epochs", "evaluation_n_bins", "activation_function"))


def test_missing_sections_are_reported(config):
    del config["trainer_config"]
    assert "must define the sections" in error_message(config)


def test_a_cuda_device_is_rejected_when_cuda_is_unavailable(config, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config["trainer_config"]["device"] = "cuda"
    assert "CUDA is not available" in error_message(config)


def test_training_requires_the_model_to_be_built(config):
    config["model_config"]["skip_building"] = True
    assert "needs the model to be built" in error_message(config)


def test_steps_that_will_not_run_are_not_validated(config):
    config["evaluator_config"]["skip_evaluation"] = True
    config["evaluator_config"]["test_data_path"] = "/nonexistent"
    config["processor_config"]["skip_processing"] = True  # the processed split already exists
    config["processor_config"]["dataset_paths"]["raw_data"] = "/nonexistent"
    validate_config(config)


def test_skipping_processing_needs_the_processed_files_to_exist(config, tmp_path):
    config["processor_config"]["skip_processing"] = True
    for path in ("processor_config.dataset_paths.processed_data", "model_config.processed_data_path", "trainer_config.processed_data_path"):
        set_path(config, path, str(tmp_path / "nothing_here"))
    assert "does not exist" in error_message(config)


def test_skipping_training_needs_an_existing_model_file(config, tmp_path):
    config["trainer_config"]["skip_training"] = True
    config["model_config"]["skip_building"] = True
    config["evaluator_config"]["model_path"] = str(tmp_path / "missing.pt2")
    assert "model_path must be an existing file" in error_message(config)

    (tmp_path / "model.pt2").write_bytes(b"")
    config["evaluator_config"]["model_path"] = str(tmp_path / "model.pt2")
    validate_config(config)


def test_enabled_steps_defaults_skip_processing_and_building_only():
    sections = {name: {} for name in ("processor_config", "model_config", "trainer_config", "evaluator_config")}
    assert enabled_steps(sections) == {"data_processing": False, "model_building": False, "training": True, "evaluation": True}
