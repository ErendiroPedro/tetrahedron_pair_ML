import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             mean_absolute_error, r2_score, roc_auc_score)

from src.data_processing import transform_data
from src.data_processing.config import parse_transformations
from src.utils.dataset_files import discover_csv_files
from src.utils.geometry import permute_vertices_within_tetrahedra, swap_tetrahedra

STATUS_COLUMN, VOLUME_COLUMN = "HasIntersection", "IntersectionVolume"
PRIORITY_TYPES = {"polyhedron_intersection", "no_intersection"}
SPEED_WARMUP_RUNS, SPEED_RUNS = 10, 50


def evaluate(model, evaluator_config, processor_config, training_metrics=None):
    """Runs the model (in float32) over every test dataset and returns the full metrics report."""
    task = evaluator_config["task"]
    device = torch.device(evaluator_config["device"])
    thresholds = evaluator_config.get("regression_consistency_thresholds", {})
    rtol, atol = thresholds.get("rtol", 1e-1), thresholds.get("atol", 1e-4)
    model = model.float().to(device)

    print(f"-- Evaluating {task} on {device} (float32) --")
    transformations = parse_transformations(processor_config.get("transformations"))
    datasets = _load_datasets(evaluator_config, task, transformations, device)

    report = {
        "task_type": task,
        "model_info": {"model_class": type(model).__name__, "total_parameters": sum(p.numel() for p in model.parameters())},
        "dataset_reports": {},
    }
    if training_metrics:
        report["training_metrics"] = training_metrics

    all_true_status, all_predicted_status = [], []
    for dataset in datasets:
        print(f"  {dataset['name']}")
        features = dataset["features"]
        predictions = _predict(model, features)
        predicted_status, predicted_volume = _split_predictions(predictions, task)

        dataset_report = {}
        if predicted_status is not None:
            dataset_report["classification_performance"] = _classification_performance(dataset["true_status"], predicted_status)
            all_true_status.append(dataset["true_status"])
            all_predicted_status.append(predicted_status)
        if predicted_volume is not None:
            dataset_report["intersection_volume_performance"] = _intersection_volume_performance(
                dataset["true_volume"], predicted_volume, evaluator_config)
        for report_key, perturb in (("point_wise_permutation_consistency", permute_vertices_within_tetrahedra),
                                    ("tetrahedron_wise_permutation_consistency", swap_tetrahedra)):
            perturbed_predictions = _predict(model, perturb(features))
            dataset_report[report_key] = _permutation_consistency(predictions, perturbed_predictions, task, rtol, atol)
        dataset_report["inference_speed"] = _inference_speed(model, features, dataset["transform_seconds"], device)
        report["dataset_reports"][dataset["name"]] = dataset_report

    dataset_reports = report["dataset_reports"]
    if task != "IntersectionVolume":
        report["overall_classification_metrics"] = _overall_classification_metrics(
            datasets, dataset_reports, all_true_status, all_predicted_status)
    if task != "IntersectionStatus":
        report["overall_regression_metrics"] = _overall_regression_metrics(dataset_reports)
    report["inference_speed_summary"] = _inference_speed_summary(dataset_reports, device)

    _print_summary(report)
    return report


def _load_datasets(config, task, transformations, device):
    intersection_types = ["polyhedron_intersection", "point_intersection", "segment_intersection", "polygon_intersection"]
    if task != "IntersectionVolume":
        intersection_types.insert(0, "no_intersection")

    datasets = []
    for intersection_type in intersection_types:
        for path in discover_csv_files(os.path.join(config["test_data_path"], intersection_type)):
            frame = pd.read_csv(path)
            start = time.perf_counter()
            frame = transform_data(frame, transformations)
            transform_seconds = time.perf_counter() - start
            feature_values = frame.drop(columns=[STATUS_COLUMN, VOLUME_COLUMN]).values
            datasets.append({
                "name": f"{intersection_type}/{os.path.basename(path)}",
                "type": intersection_type,
                "features": torch.tensor(feature_values, dtype=torch.float32, device=device),
                "true_status": frame[STATUS_COLUMN].values,
                "true_volume": frame[VOLUME_COLUMN].values.astype(np.float32),
                "transform_seconds": transform_seconds,
            })
            print(f"Loaded {path}: {len(frame):,} samples, transformed in {transform_seconds:.2f}s")
    return datasets


def _predict(model, features):
    return model.predict(features).cpu().numpy()


def _split_predictions(predictions, task):
    """Predictions -> (status, volume); the entry a task doesn't produce is None."""
    if task == "IntersectionStatus":
        return predictions, None
    if task == "IntersectionVolume":
        return None, predictions
    return predictions[:, 0], predictions[:, 1]


def _weighted_mean(value_weight_pairs):
    """Mean of (value, weight) pairs, skipping undefined values; None if nothing is left."""
    defined_pairs = [(value, weight) for value, weight in value_weight_pairs if value is not None]
    total_weight = sum(weight for _, weight in defined_pairs)
    return sum(value * weight for value, weight in defined_pairs) / total_weight if total_weight else None


def _classification_performance(true_status, predicted_status):
    predicted_status = (predicted_status > 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(true_status, predicted_status)),
        "confusion_matrix": confusion_matrix(true_status, predicted_status, labels=[0, 1]).tolist(),
        "n_samples": len(true_status),
    }


def _intersection_volume_performance(true_volume, predicted_volume, config):
    volume_range = config.get("volume_range", (0, 0.01))
    mean_true_volume = np.mean(true_volume)
    kappa, bin_accuracy = _binned_agreement(true_volume, predicted_volume)
    interval_metrics = {}
    interval_edges = np.linspace(volume_range[0], volume_range[1], config.get("evaluation_n_bins", 5))
    for low, high in zip(interval_edges[:-1], interval_edges[1:]):
        in_interval = (true_volume >= low) & (true_volume < high)
        interval_samples = int(in_interval.sum())
        correct_bin_predictions = int((in_interval & (predicted_volume >= low) & (predicted_volume < high)).sum())
        interval_true, interval_predicted = true_volume[in_interval], predicted_volume[in_interval]
        interval_metrics[f"{low:.8f}-{high:.8f}"] = {
            "mae": float(mean_absolute_error(interval_true, interval_predicted)) if interval_samples else None,
            "mse": float(np.mean((interval_true - interval_predicted) ** 2)) if interval_samples else None,
            "samples": interval_samples,
            "correct_bin_predictions": correct_bin_predictions,
            "bin_accuracy": correct_bin_predictions / interval_samples if interval_samples else None,
        }

    absolute_error = np.abs(true_volume - predicted_volume)
    return {
        "overall_IntersectionVolume_metrics": {
            "mae": float(mean_absolute_error(true_volume, predicted_volume)),
            "r2_score": float(r2_score(true_volume, predicted_volume)),
            "relative_mae": float(np.mean(absolute_error) / mean_true_volume) if mean_true_volume != 0 else None,
            "smape": float(100 * np.mean(2 * absolute_error / (np.abs(true_volume) + np.abs(predicted_volume) + 1e-8))),
            "kappa": None if kappa is None else float(kappa),
            "samples": len(true_volume),
            "overall_bin_accuracy": float(bin_accuracy),
        },
        "interval_metrics": interval_metrics,
    }


def _binned_agreement(true_volume, predicted_volume, n_bins=10):
    """Cohen's kappa and accuracy after binning both volumes into `n_bins` (log-spaced if all values > 0, else linear)."""
    low = min(true_volume.min(), predicted_volume.min())
    high = max(true_volume.max(), predicted_volume.max())
    bin_edges = np.logspace(np.log10(low), np.log10(high), n_bins + 1) if low > 0 else np.linspace(low, high, n_bins + 1)
    true_bins = np.clip(np.digitize(true_volume, bin_edges) - 1, 0, n_bins - 1)
    predicted_bins = np.clip(np.digitize(predicted_volume, bin_edges) - 1, 0, n_bins - 1)
    both_in_one_bin = len(np.union1d(true_bins, predicted_bins)) == 1  # kappa is 0/0 there
    kappa = None if both_in_one_bin else cohen_kappa_score(true_bins, predicted_bins)
    return kappa, accuracy_score(true_bins, predicted_bins)


def _permutation_consistency(predictions, perturbed_predictions, task, rtol, atol):
    """How often predictions survive an input perturbation that should not change them."""
    status, volume = _split_predictions(predictions, task)
    perturbed_status, perturbed_volume = _split_predictions(perturbed_predictions, task)
    consistency = {}
    if status is not None:
        consistency["classification_consistency_rate"] = float(np.mean(status == perturbed_status))
    if volume is not None:
        consistent_volumes = np.isclose(volume, perturbed_volume, rtol=rtol, atol=atol)
        consistency["IntersectionVolume_consistency_rate"] = float(np.mean(consistent_volumes))
        consistency["mean_absolute_difference"] = float(np.mean(np.abs(volume - perturbed_volume)))
    return consistency


def _inference_speed(model, features, transform_seconds, device):
    """Time to transform the dataset plus the mean time of a full forward pass over it."""
    def timed_predict():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model.predict(features)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start

    for _ in range(SPEED_WARMUP_RUNS):
        model.predict(features)
    inference_seconds = float(np.mean([timed_predict() for _ in range(SPEED_RUNS)]))
    total_seconds = transform_seconds + inference_seconds
    return {
        "transformation_time_seconds": transform_seconds,
        "inference_time_seconds": inference_seconds,
        "total_time_seconds": total_seconds,
        "avg_time_per_sample_seconds": total_seconds / len(features),
        "samples_per_second": len(features) / total_seconds,
        "total_samples": len(features),
    }


def _overall_classification_metrics(datasets, dataset_reports, all_true_status, all_predicted_status):
    """Sample-weighted accuracies; the 80/20 and 20/80 blends weight no/polyhedron intersection (priority) against the rest."""
    accuracy_by_type = []
    for dataset in datasets:
        performance = dataset_reports[dataset["name"]]["classification_performance"]
        accuracy_by_type.append((dataset["type"], (performance["accuracy"], performance["n_samples"])))

    priority_accuracy = _weighted_mean([pair for kind, pair in accuracy_by_type if kind in PRIORITY_TYPES])
    secondary_accuracy = _weighted_mean([pair for kind, pair in accuracy_by_type if kind not in PRIORITY_TYPES])
    both_groups_present = priority_accuracy is not None and secondary_accuracy is not None
    true_status, predicted_status = np.concatenate(all_true_status), np.concatenate(all_predicted_status)
    return {
        "overall_accuracy": _weighted_mean([pair for _, pair in accuracy_by_type]),
        "priority_group_accuracy": priority_accuracy,
        "secondary_group_accuracy": secondary_accuracy,
        "weighted_80_20_accuracy": 0.8 * priority_accuracy + 0.2 * secondary_accuracy if both_groups_present else None,
        "weighted_20_80_accuracy": 0.2 * priority_accuracy + 0.8 * secondary_accuracy if both_groups_present else None,
        "overall_auc": float(roc_auc_score(true_status, predicted_status)) if len(np.unique(true_status)) > 1 else None,
    }


def _overall_regression_metrics(dataset_reports):
    dataset_metrics = [report["intersection_volume_performance"]["overall_IntersectionVolume_metrics"]
                       for report in dataset_reports.values()]

    def sample_weighted_mean(metric):
        return _weighted_mean([(metrics[metric], metrics["samples"]) for metrics in dataset_metrics])

    return {
        "overall_mae": sample_weighted_mean("mae"),
        "overall_r2_score": sample_weighted_mean("r2_score"),
        "overall_relative_mae": sample_weighted_mean("relative_mae"),
        "overall_smape": sample_weighted_mean("smape"),
        "overall_kappa": sample_weighted_mean("kappa"),
        "overall_bin_accuracy": sample_weighted_mean("overall_bin_accuracy"),
        "total_samples": sum(metrics["samples"] for metrics in dataset_metrics),
    }


def _inference_speed_summary(dataset_reports, device):
    speeds = [report["inference_speed"]["samples_per_second"] for report in dataset_reports.values()]
    mean_speed = float(np.mean(speeds))
    return {
        "device": str(device),
        "total_samples": sum(report["inference_speed"]["total_samples"] for report in dataset_reports.values()),
        "avg_throughput_samples_per_second": mean_speed,
        "avg_latency_per_sample_ms": 1000 / mean_speed,
    }


def _format_or_na(value, format_spec):
    return "n/a" if value is None else format(value, format_spec)


def _print_summary(report):
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY - {report['model_info']['model_class']} ({report['model_info']['total_parameters']:,} parameters)")
    print("=" * 80)
    for dataset_name, dataset_report in report["dataset_reports"].items():
        summary = []
        if "classification_performance" in dataset_report:
            summary.append(f"accuracy {dataset_report['classification_performance']['accuracy']:.4f}")
        if "intersection_volume_performance" in dataset_report:
            volume_metrics = dataset_report["intersection_volume_performance"]["overall_IntersectionVolume_metrics"]
            summary.append(f"MAE {volume_metrics['mae']:.3e}")
        for label, report_key in (("point-wise", "point_wise_permutation_consistency"),
                                  ("tetra-wise", "tetrahedron_wise_permutation_consistency")):
            rates = [rate for name, rate in dataset_report[report_key].items() if name.endswith("consistency_rate")]
            summary.append(f"{label} consistency " + "/".join(f"{rate:.3f}" for rate in rates))
        summary.append(f"{dataset_report['inference_speed']['samples_per_second']:,.0f} samples/s")
        print(f"{dataset_name}\n    " + ", ".join(summary))

    print("\n--- OVERALL ---")
    if "overall_classification_metrics" in report:
        classification = report["overall_classification_metrics"]
        print(f"Overall Accuracy: {_format_or_na(classification['overall_accuracy'], '.4f')}")
        print(f"Weighted 80/20 Accuracy: {_format_or_na(classification['weighted_80_20_accuracy'], '.4f')}")
        print(f"Weighted 20/80 Accuracy: {_format_or_na(classification['weighted_20_80_accuracy'], '.4f')}")
    if "overall_regression_metrics" in report:
        regression = report["overall_regression_metrics"]
        print(f"Overall MAE: {_format_or_na(regression['overall_mae'], '.6e')}")
        print(f"Overall R2: {_format_or_na(regression['overall_r2_score'], '.4f')}")
        print(f"Overall Relative MAE: {_format_or_na(regression['overall_relative_mae'], '.4f')}")
        print(f"Overall SMAPE: {_format_or_na(regression['overall_smape'], '.2f')}%")
        print(f"Overall Kappa: {_format_or_na(regression['overall_kappa'], '.4f')}")
        print(f"Overall Bin Accuracy: {_format_or_na(regression['overall_bin_accuracy'], '.4f')}")
    speed = report["inference_speed_summary"]
    print(f"Throughput: {speed['avg_throughput_samples_per_second']:,.0f} samples/s on {speed['device']} "
          f"({speed['avg_latency_per_sample_ms']:.6f} ms/sample)")
    print("=" * 80)
