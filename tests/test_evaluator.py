import copy
import json
import warnings

import numpy as np
import pytest
import torch

from src.evaluation import evaluate
from src.evaluation.evaluator import (
    _binned_agreement,
    _classification_performance,
    _permutation_consistency,
    _weighted_mean,
)
from src.models import build_model
from tests.conftest import TEST_ROWS_PER_DATASET


def test_binned_agreement_leaves_kappa_undefined_when_everything_is_in_one_bin():
    zeros = np.zeros(100, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the old 0/0 RuntimeWarning must not come back
        kappa, accuracy = _binned_agreement(zeros, zeros)
    assert kappa is None and accuracy == 1.0


def test_binned_agreement_is_perfect_for_identical_volumes_and_uses_log_bins_for_positive_ones():
    volumes = np.logspace(-4, -1, 200).astype(np.float32)
    kappa, accuracy = _binned_agreement(volumes, volumes)
    assert kappa == pytest.approx(1.0) and accuracy == 1.0


def test_weighted_mean_skips_undefined_values_and_is_none_when_nothing_is_left():
    assert _weighted_mean([(1.0, 1), (3.0, 3), (None, 100)]) == pytest.approx(2.5)
    assert _weighted_mean([(None, 1)]) is None
    assert _weighted_mean([]) is None


@pytest.mark.parametrize("true, predicted, matrix", [
    ([0, 0, 0, 0], [0, 0, 1, 0], [[3, 1], [0, 0]]),
    ([0, 0, 0, 0], [0, 0, 0, 0], [[4, 0], [0, 0]]),  # only one class occurs anywhere
    ([1, 1, 1, 1], [1, 1, 1, 1], [[0, 0], [0, 4]]),
])
def test_classification_performance_always_reports_a_two_by_two_confusion_matrix(true, predicted, matrix):
    performance = _classification_performance(np.array(true), np.array(predicted))
    assert performance["confusion_matrix"] == matrix and performance["n_samples"] == 4


def test_permutation_consistency_counts_unchanged_predictions():
    predictions = np.array([[1, 1.00], [0, 2.00], [1, 3.00]])
    perturbed = np.array([[1, 1.00], [1, 2.05], [1, 9.00]])
    consistency = _permutation_consistency(predictions, perturbed, "IntersectionStatus_IntersectionVolume", rtol=0.1, atol=0.0)
    assert consistency["classification_consistency_rate"] == pytest.approx(2 / 3)
    assert consistency["IntersectionVolume_consistency_rate"] == pytest.approx(2 / 3)
    assert consistency["mean_absolute_difference"] == pytest.approx((0 + 0.05 + 6) / 3)


def evaluate_with_task(config, task):
    model_config = copy.deepcopy(config["model_config"])
    model_config["common_parameters"]["task"] = task
    config["evaluator_config"]["task"] = task
    torch.manual_seed(0)
    return evaluate(build_model(model_config), config["evaluator_config"], config["processor_config"], {"final_val_loss": 0.5})


@pytest.mark.parametrize("task, dataset_count, has_status, has_volume", [
    ("IntersectionStatus_IntersectionVolume", 5, True, True),
    ("IntersectionStatus", 5, True, False),
    ("IntersectionVolume", 4, False, True),  # no_intersection has no volume to learn from, so it is skipped
])
def test_evaluate_reports_exactly_what_the_task_produces(config, task, dataset_count, has_status, has_volume):
    report = evaluate_with_task(config, task)

    json.dumps(report, allow_nan=False)  # strict JSON: plain Python numbers, no NaN or Infinity
    assert report["task_type"] == task and report["training_metrics"] == {"final_val_loss": 0.5}
    assert len(report["dataset_reports"]) == dataset_count
    for dataset_report in report["dataset_reports"].values():
        assert ("classification_performance" in dataset_report) == has_status
        assert ("intersection_volume_performance" in dataset_report) == has_volume
        assert {"point_wise_permutation_consistency", "tetrahedron_wise_permutation_consistency", "inference_speed"} <= set(dataset_report)
    assert ("overall_classification_metrics" in report) == has_status
    assert ("overall_regression_metrics" in report) == has_volume


def test_evaluate_computes_the_overall_metrics_from_the_per_dataset_reports(config):
    report = evaluate_with_task(config, "IntersectionStatus_IntersectionVolume")
    datasets = report["dataset_reports"]
    classification = report["overall_classification_metrics"]

    priority = [d for name, d in datasets.items() if name.startswith(("no_intersection", "polyhedron_intersection"))]
    secondary = [d for name, d in datasets.items() if not name.startswith(("no_intersection", "polyhedron_intersection"))]
    accuracy = lambda group: np.mean([d["classification_performance"]["accuracy"] for d in group])  # equal sizes
    assert classification["priority_group_accuracy"] == pytest.approx(accuracy(priority))
    assert classification["secondary_group_accuracy"] == pytest.approx(accuracy(secondary))
    assert classification["weighted_80_20_accuracy"] == pytest.approx(0.8 * accuracy(priority) + 0.2 * accuracy(secondary))
    assert classification["overall_accuracy"] == pytest.approx(accuracy(list(datasets.values())))

    maes = [d["intersection_volume_performance"]["overall_IntersectionVolume_metrics"]["mae"] for d in datasets.values()]
    assert report["overall_regression_metrics"]["overall_mae"] == pytest.approx(np.mean(maes))
    assert report["overall_regression_metrics"]["total_samples"] == 5 * TEST_ROWS_PER_DATASET


def test_evaluate_marks_undefined_metrics_as_null_instead_of_infinity(config):
    report = evaluate_with_task(config, "IntersectionStatus_IntersectionVolume")
    no_intersection = report["dataset_reports"]["no_intersection/test.csv"]
    metrics = no_intersection["intersection_volume_performance"]["overall_IntersectionVolume_metrics"]
    assert metrics["relative_mae"] is None, "all true volumes are 0, so a relative error does not exist"
    assert no_intersection["classification_performance"]["confusion_matrix"][1] == [0, 0], "the dataset has no positives"
