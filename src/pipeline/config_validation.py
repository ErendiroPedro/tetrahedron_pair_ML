"""Checks the whole config before any step runs, so mistakes surface in seconds instead of after processing or training."""

import os

import torch

from src.data_processing.config import parse_transformations
from src.models.networks import ACTIVATIONS, AGGREGATIONS, ARCHITECTURES
from src.training.losses import LOSS_BY_TASK
from src.utils.dataset_files import INTERSECTION_TYPES, processed_split_path

SECTIONS = ("processor_config", "model_config", "trainer_config", "evaluator_config")
TASKS = tuple(LOSS_BY_TASK)
VOLUME_TASKS = ("IntersectionVolume", "IntersectionStatus_IntersectionVolume")
MAX_TETRAHEDRON_VOLUME = 1 / 6  # the largest tetrahedron fitting the unit cube; upper bound for the evaluation `volume_range`
TETRAHEDRONPAIRNET_LAYERS = ("per_vertex_layers", "per_tetrahedron_layers", "per_two_tetrahedra_layers")
TETRAHEDRONPAIRNET_AGGREGATIONS = ("vertices_aggregation_function", "tetrahedra_aggregation_function")


def enabled_steps(config):
    """Which pipeline steps run (the skip_* defaults live here only)."""
    return {
        "data_processing": not config["processor_config"].get("skip_processing", True),
        "model_building": not config["model_config"].get("skip_building", True),
        "training": not config["trainer_config"].get("skip_training", False),
        "evaluation": not config["evaluator_config"].get("skip_evaluation", False),
    }


def validate_config(config):
    """Raises one ValueError listing every problem found in the steps that are going to run."""
    if not isinstance(config, dict) or not all(isinstance(config.get(name), dict) for name in SECTIONS):
        raise ValueError(f"Invalid config: it must define the sections {', '.join(SECTIONS)}")

    steps = enabled_steps(config)
    errors = _check_run_identity(config) + _check_consistency(config, steps)
    if steps["data_processing"]:
        errors += _check_data_processing(config["processor_config"])
    if steps["model_building"]:
        errors += _check_model(config["model_config"], steps)
    if steps["training"]:
        errors += _check_training(config["trainer_config"], steps)
    else:
        errors += _check_loading_model(config)
    if steps["evaluation"]:
        errors += _check_evaluation(config["evaluator_config"])

    if errors:
        raise ValueError("Invalid config:\n" + "\n".join(f"  - {error}" for error in errors))


def _check_run_identity(config):
    """Keys that name the experiment folder, so they are needed whichever steps run."""
    errors = []
    for key in ("processor_config.num_train_samples", "processor_config.num_val_samples", "trainer_config.epochs"):
        if not _is_int(_get(config, key), minimum=1):
            errors.append(f"{key} must be an integer >= 1, got {_get(config, key)!r}")
    architecture = _get(config, "model_config.architecture.use_model")
    if str(architecture).lower() not in ARCHITECTURES:
        errors.append(f"model_config.architecture.use_model must be one of {sorted(ARCHITECTURES)}, got {architecture!r}")
    errors += _check_task("model_config.common_parameters.task", _get(config, "model_config.common_parameters.task"))
    return errors


def _check_consistency(config, steps):
    """Values repeated across sections must agree, otherwise a run silently trains or evaluates on the wrong thing."""
    errors = []
    tasks = {"model_config.common_parameters.task": _get(config, "model_config.common_parameters.task")}
    if steps["training"]:
        tasks["trainer_config.task"] = _get(config, "trainer_config.task")
    if steps["evaluation"]:
        tasks["evaluator_config.task"] = _get(config, "evaluator_config.task")
    if len(set(map(str, tasks.values()))) > 1:
        errors.append(f"the task must be identical in every section, got {tasks}")

    scale_keys = ("processor_config.volume_scale_factor", "model_config.common_parameters.volume_scale_factor",
                  "evaluator_config.volume_scale_factor")
    scales = {key: _get(config, key) for key in scale_keys if _get(config, key) is not None}
    if len(set(scales.values())) > 1:
        errors.append(f"volume_scale_factor must be identical in every section, got {scales}")

    paths = {}
    if steps["data_processing"]:
        paths["processor_config.dataset_paths.processed_data"] = _get(config, "processor_config.dataset_paths.processed_data")
    if steps["model_building"]:
        paths["model_config.processed_data_path"] = _get(config, "model_config.processed_data_path")
    if steps["training"]:
        paths["trainer_config.processed_data_path"] = _get(config, "trainer_config.processed_data_path", "data/processed")
    if len({os.path.normpath(str(path)) for path in paths.values()}) > 1:
        errors.append(f"the processed data path must be identical in every section, got {paths}")
    return errors


def _check_data_processing(processor):
    errors = []
    raw_data = _get(processor, "dataset_paths.raw_data")
    if not (isinstance(raw_data, str) and os.path.isdir(raw_data)):
        errors.append(f"processor_config.dataset_paths.raw_data must be an existing directory, got {raw_data!r}")
    if not isinstance(_get(processor, "dataset_paths.processed_data"), str):
        errors.append("processor_config.dataset_paths.processed_data must be a path")
    if not _is_number(processor.get("volume_scale_factor", 1.0), exclusive_minimum=0):
        errors.append("processor_config.volume_scale_factor must be a number > 0")

    distributions = processor.get("intersection_distributions") or {}
    if unknown := sorted(set(distributions) - set(INTERSECTION_TYPES)):
        errors.append(f"processor_config.intersection_distributions has unknown types {unknown}; known: {list(INTERSECTION_TYPES)}")
    elif distributions and not all(_is_number(share, minimum=0) for share in distributions.values()):
        errors.append("processor_config.intersection_distributions shares must be numbers >= 0")
    elif distributions and abs(sum(distributions.values()) - 100.0) > 0.001:
        errors.append(f"processor_config.intersection_distributions must sum to 100, got {sum(distributions.values())}")

    binning = processor.get("volume_binning") or {}
    volume_range = binning.get("volume_range")
    if volume_range is not None and not _is_range(volume_range, minimum_exclusive=0):
        errors.append(f"processor_config.volume_binning.volume_range must be [low, high] with 0 < low < high, got {volume_range!r}")
    if not _is_int(binning.get("n_bins", 100), minimum=1):
        errors.append("processor_config.volume_binning.n_bins must be an integer >= 1")
    try:
        parse_transformations(processor.get("transformations"))
    except ValueError as error:
        errors.append(f"processor_config.transformations: {error}")
    return errors


def _check_model(model, steps):
    errors = []
    architecture = str(_get(model, "architecture.use_model")).lower()
    settings = _get(model, f"architecture.{architecture}")
    if architecture in ARCHITECTURES and not isinstance(settings, dict):
        errors.append(f"model_config.architecture.{architecture} must be a section")
    elif isinstance(settings, dict):
        for key in ("shared_layers", "classification_head", "regression_head"):
            if key in settings and not _is_layer_list(settings[key]):
                errors.append(f"model_config.architecture.{architecture}.{key} must be a list of integers >= 1")
        if architecture == "tetrahedronpairnet":
            errors += [f"model_config.architecture.tetrahedronpairnet.{key} must be a non-empty list of integers >= 1"
                       for key in TETRAHEDRONPAIRNET_LAYERS if not (_is_layer_list(settings.get(key)) and settings[key])]
            errors += [f"model_config.architecture.tetrahedronpairnet.{key} must be one of {sorted(AGGREGATIONS)}"
                       for key in TETRAHEDRONPAIRNET_AGGREGATIONS if settings.get(key) not in AGGREGATIONS]

    common = model.get("common_parameters") or {}
    if str(common.get("activation_function")).lower() not in ACTIVATIONS:
        errors.append(f"model_config.common_parameters.activation_function must be one of {sorted(ACTIVATIONS)}")
    if not _is_number(common.get("dropout_rate", 0.0), minimum=0) or common.get("dropout_rate", 0.0) >= 1:
        errors.append("model_config.common_parameters.dropout_rate must be a number in [0, 1)")
    if not _is_number(common.get("volume_scale_factor"), exclusive_minimum=0):
        errors.append("model_config.common_parameters.volume_scale_factor must be a number > 0")
    errors += _check_processed_files("model_config.processed_data_path", model.get("processed_data_path"), ("train",), steps)
    return errors


def _check_training(trainer, steps):
    errors = []
    if not steps["model_building"]:
        errors.append("training needs the model to be built: set model_config.skip_building to false")
    errors += _check_task("trainer_config.task", trainer.get("task"))
    errors += _check_device("trainer_config.device", trainer.get("device"))
    for key, default in (("batch_size", 1024), ("early_stopping_patience", 10)):
        if not _is_int(trainer.get(key, default), minimum=1):
            errors.append(f"trainer_config.{key} must be an integer >= 1")
    if not _is_number(trainer.get("learning_rate", 1e-3), exclusive_minimum=0):
        errors.append("trainer_config.learning_rate must be a number > 0")
    errors += _check_processed_files("trainer_config.processed_data_path", trainer.get("processed_data_path", "data/processed"),
                                     ("train", "val"), steps)
    return errors


def _check_loading_model(config):
    """Without training, the model is loaded from a saved model file."""
    evaluator, model = config["evaluator_config"], config["model_config"]
    errors = []
    path = evaluator.get("model_path") or model.get("model_path")
    if not (isinstance(path, str) and os.path.isfile(path)):
        errors.append(f"trainer_config.skip_training is true, so the model is loaded and model_path must be an existing file, got {path!r}")
    for key in ("task", "volume_scale_factor"):
        if evaluator.get(key) is None:
            errors.append(f"evaluator_config.{key} is required to load a model")
    return errors


def _check_evaluation(evaluator):
    errors = _check_task("evaluator_config.task", evaluator.get("task"))
    errors += _check_device("evaluator_config.device", evaluator.get("device"))
    test_data = evaluator.get("test_data_path")
    if not (isinstance(test_data, str) and os.path.isdir(test_data)):
        errors.append(f"evaluator_config.test_data_path must be an existing directory, got {test_data!r}")
    if evaluator.get("task") in VOLUME_TASKS:
        volume_range = evaluator.get("volume_range", (0, 0.01))
        if not _is_range(volume_range, minimum=0, maximum=MAX_TETRAHEDRON_VOLUME):
            errors.append(f"evaluator_config.volume_range must be [low, high] with 0 <= low < high <= {MAX_TETRAHEDRON_VOLUME:.6f}, got {volume_range!r}")
        if not _is_int(evaluator.get("evaluation_n_bins", 5), minimum=2):
            errors.append("evaluator_config.evaluation_n_bins must be an integer >= 2 (it counts interval edges)")
    thresholds = evaluator.get("regression_consistency_thresholds", {})
    if not (isinstance(thresholds, dict) and all(_is_number(value, minimum=0) for value in thresholds.values())):
        errors.append("evaluator_config.regression_consistency_thresholds must map rtol/atol to numbers >= 0")
    return errors


def _check_processed_files(name, processed_data_path, splits, steps):
    """The processed CSVs must exist unless this run creates them first."""
    if steps["data_processing"]:
        return []
    if not isinstance(processed_data_path, str):
        return [f"{name} must be a path"]
    return [f"{name}: {processed_split_path(processed_data_path, split)} does not exist (enable processing or generate it first)"
            for split in splits if not os.path.isfile(processed_split_path(processed_data_path, split))]


def _check_task(name, task):
    return [] if task in TASKS else [f"{name} must be one of {list(TASKS)}, got {task!r}"]


def _check_device(name, device):
    try:
        parsed = torch.device(device)
    except (RuntimeError, TypeError):
        return [f"{name} is not a valid torch device: {device!r}"]
    if parsed.type == "cuda" and not torch.cuda.is_available():
        return [f"{name} is {device!r} but CUDA is not available"]
    return []


def _get(node, dotted_path, default=None):
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _is_int(value, minimum):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_number(value, minimum=None, exclusive_minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return (minimum is None or value >= minimum) and (exclusive_minimum is None or value > exclusive_minimum)


def _is_layer_list(value):
    return isinstance(value, list) and all(_is_int(width, minimum=1) for width in value)


def _is_range(value, minimum=None, minimum_exclusive=None, maximum=None):
    if not (isinstance(value, (list, tuple)) and len(value) == 2 and all(_is_number(v) for v in value)):
        return False
    low, high = value
    return (low < high
            and (minimum is None or low >= minimum)
            and (minimum_exclusive is None or low > minimum_exclusive)
            and (maximum is None or high <= maximum))
