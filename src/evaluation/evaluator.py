import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             mean_absolute_error, r2_score, roc_auc_score)

import src.utils.geometry as gu
from src.data_processing import transform_data
from src.data_processing.config import parse_transformations
from src.utils.file_management import discover_csv_files

STATUS_COLUMN, VOLUME_COLUMN = "HasIntersection", "IntersectionVolume"
PRIORITY_TYPES = {"polyhedron_intersection", "no_intersection"}
SPEED_WARMUP_RUNS, SPEED_RUNS = 10, 50
MAX_TETRAHEDRON_VOLUME = 1 / 6  # unit-cube bound used to validate `volume_range`


def evaluate(model, evaluator_config, processor_config, training_metrics=None):
    """Runs the model (in float32) over every test dataset and returns the full metrics report."""
    task = evaluator_config["task"]
    device = torch.device(evaluator_config["device"])
    thresholds = evaluator_config.get("regression_consistency_thresholds", {})
    rtol, atol = thresholds.get("rtol", 1e-1), thresholds.get("atol", 1e-4)
    model = model.float().to(device)

    print(f"-- Evaluating {task} on {device} (float32) --")
    datasets = _load_datasets(evaluator_config, task, parse_transformations(processor_config.get("transformations")), device)

    report = {
        "task_type": task,
        "model_info": {"model_class": type(model).__name__, "total_parameters": sum(p.numel() for p in model.parameters())},
        "dataset_reports": {},
    }
    if training_metrics:
        report["training_metrics"] = training_metrics

    status_true, status_pred_all = [], []
    for dataset in datasets:
        print(f"  {dataset['name']}")
        X = dataset["X"]
        pred = _predict(model, X)
        status_pred, volume_pred = _split(pred, task)

        result = {}
        if status_pred is not None:
            result["classification_performance"] = _classification(dataset["status"], status_pred)
            status_true.append(dataset["status"])
            status_pred_all.append(status_pred)
        if volume_pred is not None:
            result["intersection_volume_performance"] = _volume_performance(dataset["volume"], volume_pred, evaluator_config)
        for name, perturb in (("point_wise_permutation_consistency", gu.permute_points_within_tetrahedrons),
                              ("tetrahedron_wise_permutation_consistency", gu.swap_tetrahedrons)):
            result[name] = _consistency(pred, _predict(model, perturb(X)), task, rtol, atol)
        result["inference_speed"] = _inference_speed(model, X, dataset["transform_seconds"], device)
        report["dataset_reports"][dataset["name"]] = result

    reports = report["dataset_reports"]
    if task != "IntersectionVolume":
        report["overall_classification_metrics"] = _overall_classification(datasets, reports, status_true, status_pred_all)
    if task != "IntersectionStatus":
        report["overall_regression_metrics"] = _overall_regression(reports)
    report["inference_speed_summary"] = _inference_speed_summary(reports, device)

    _print_summary(report)
    return report


def _load_datasets(config, task, transformations, device):
    folders = ["polyhedron_intersection", "point_intersection", "segment_intersection", "polygon_intersection"]
    if task != "IntersectionVolume":
        folders.insert(0, "no_intersection")

    datasets = []
    for folder in folders:
        for path in discover_csv_files(os.path.join(config["test_data_path"], folder)):
            df = pd.read_csv(path)
            start = time.perf_counter()
            df = transform_data(df, transformations)
            transform_seconds = time.perf_counter() - start
            features = df.drop(columns=[STATUS_COLUMN, VOLUME_COLUMN]).values
            datasets.append({
                "name": f"{folder}/{os.path.basename(path)}",
                "type": folder,
                "X": torch.tensor(features, dtype=torch.float32, device=device),
                "status": df[STATUS_COLUMN].values,
                "volume": df[VOLUME_COLUMN].values.astype(np.float32),
                "transform_seconds": transform_seconds,
            })
            print(f"Loaded {path}: {len(df):,} samples, transformed in {transform_seconds:.2f}s")
    return datasets


def _predict(model, X):
    return model.predict(X).cpu().numpy()


def _split(pred, task):
    """Predictions -> (status, volume); the entry a task doesn't produce is None."""
    if task == "IntersectionStatus":
        return pred, None
    if task == "IntersectionVolume":
        return None, pred
    return pred[:, 0], pred[:, 1]


def _defined(value):
    return None if value is None or np.isnan(value) else float(value)


def _weighted_mean(pairs):
    """Mean of (value, weight) pairs, skipping undefined values; None if nothing is left."""
    pairs = [(v, w) for v, w in pairs if v is not None]
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else None


def _classification(true, pred):
    pred = (pred > 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "confusion_matrix": confusion_matrix(true, pred, labels=[0, 1]).tolist(),
        "n_samples": len(true),
    }


def _volume_performance(true, pred, config):
    volume_range = config.get("volume_range", (0, 0.01))
    if volume_range[0] < 0 or volume_range[1] > MAX_TETRAHEDRON_VOLUME:
        raise ValueError(f"volume_range must lie within [0, {MAX_TETRAHEDRON_VOLUME:.8f}], got {volume_range}")

    mean_true = np.mean(true)
    kappa, bin_accuracy = _binned_agreement(true, pred)
    intervals = {}
    edges = np.linspace(volume_range[0], volume_range[1], config.get("evaluation_n_bins", 5))
    for low, high in zip(edges[:-1], edges[1:]):
        in_interval = (true >= low) & (true < high)
        samples = int(in_interval.sum())
        correct = int((in_interval & (pred >= low) & (pred < high)).sum())
        intervals[f"{low:.8f}-{high:.8f}"] = {
            "mae": float(mean_absolute_error(true[in_interval], pred[in_interval])) if samples else None,
            "mse": float(np.mean((true[in_interval] - pred[in_interval]) ** 2)) if samples else None,
            "samples": samples,
            "correct_bin_predictions": correct,
            "bin_accuracy": correct / samples if samples else None,
        }

    return {
        "overall_IntersectionVolume_metrics": {
            "mae": float(mean_absolute_error(true, pred)),
            "r2_score": float(r2_score(true, pred)),
            "relative_mae": float(np.mean(np.abs(true - pred)) / mean_true) if mean_true != 0 else None,
            "smape": float(100 * np.mean(2 * np.abs(pred - true) / (np.abs(true) + np.abs(pred) + 1e-8))),
            "kappa": _defined(kappa),
            "samples": len(true),
            "overall_bin_accuracy": float(bin_accuracy),
        },
        "interval_metrics": intervals,
    }


def _binned_agreement(true, pred, n_bins=10):
    """Cohen's kappa and accuracy after binning both volumes into `n_bins` (log-spaced if all values > 0, else linear)."""
    low, high = min(true.min(), pred.min()), max(true.max(), pred.max())
    edges = np.logspace(np.log10(low), np.log10(high), n_bins + 1) if low > 0 else np.linspace(low, high, n_bins + 1)
    true_bins = np.clip(np.digitize(true, edges) - 1, 0, n_bins - 1)
    pred_bins = np.clip(np.digitize(pred, edges) - 1, 0, n_bins - 1)
    return cohen_kappa_score(true_bins, pred_bins), accuracy_score(true_bins, pred_bins)


def _consistency(pred, perturbed_pred, task, rtol, atol):
    """How often predictions survive an input perturbation that should not change them."""
    status, volume = _split(pred, task)
    perturbed_status, perturbed_volume = _split(perturbed_pred, task)
    result = {}
    if status is not None:
        result["classification_consistency_rate"] = float(np.mean(status == perturbed_status))
    if volume is not None:
        result["IntersectionVolume_consistency_rate"] = float(np.mean(np.isclose(volume, perturbed_volume, rtol=rtol, atol=atol)))
        result["mean_absolute_difference"] = float(np.mean(np.abs(volume - perturbed_volume)))
    return result


def _inference_speed(model, X, transform_seconds, device):
    """Time to transform the dataset plus the mean time of a full forward pass over it."""
    def timed_predict():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model.predict(X)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start

    for _ in range(SPEED_WARMUP_RUNS):
        model.predict(X)
    inference_seconds = float(np.mean([timed_predict() for _ in range(SPEED_RUNS)]))
    total_seconds = transform_seconds + inference_seconds
    return {
        "transformation_time_seconds": transform_seconds,
        "inference_time_seconds": inference_seconds,
        "total_time_seconds": total_seconds,
        "avg_time_per_sample_seconds": total_seconds / len(X),
        "samples_per_second": len(X) / total_seconds,
        "total_samples": len(X),
    }


def _overall_classification(datasets, reports, status_true, status_pred):
    """Sample-weighted accuracies; the 80/20 and 20/80 blends weight no/polyhedron intersection (priority) against the rest."""
    pairs = [(d["type"], (reports[d["name"]]["classification_performance"]["accuracy"],
                          reports[d["name"]]["classification_performance"]["n_samples"])) for d in datasets]
    priority = _weighted_mean([pair for t, pair in pairs if t in PRIORITY_TYPES])
    secondary = _weighted_mean([pair for t, pair in pairs if t not in PRIORITY_TYPES])
    true, pred = np.concatenate(status_true), np.concatenate(status_pred)
    both = priority is not None and secondary is not None
    return {
        "overall_accuracy": _weighted_mean([pair for _, pair in pairs]),
        "priority_group_accuracy": priority,
        "secondary_group_accuracy": secondary,
        "weighted_80_20_accuracy": 0.8 * priority + 0.2 * secondary if both else None,
        "weighted_20_80_accuracy": 0.2 * priority + 0.8 * secondary if both else None,
        "overall_auc": float(roc_auc_score(true, pred)) if len(np.unique(true)) > 1 else None,
    }


def _overall_regression(reports):
    metrics = [r["intersection_volume_performance"]["overall_IntersectionVolume_metrics"] for r in reports.values()]

    def mean(key):
        return _weighted_mean([(m[key], m["samples"]) for m in metrics])

    return {
        "overall_mae": mean("mae"),
        "overall_r2_score": mean("r2_score"),
        "overall_relative_mae": mean("relative_mae"),
        "overall_smape": mean("smape"),
        "overall_kappa": mean("kappa"),
        "overall_bin_accuracy": mean("overall_bin_accuracy"),
        "total_samples": sum(m["samples"] for m in metrics),
    }


def _inference_speed_summary(reports, device):
    speeds = [r["inference_speed"]["samples_per_second"] for r in reports.values()]
    mean_speed = float(np.mean(speeds))
    return {
        "device": str(device),
        "total_samples": sum(r["inference_speed"]["total_samples"] for r in reports.values()),
        "avg_throughput_samples_per_second": mean_speed,
        "avg_latency_per_sample_ms": 1000 / mean_speed,
    }


def _fmt(value, spec):
    return "n/a" if value is None else format(value, spec)


def _print_summary(report):
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY - {report['model_info']['model_class']} ({report['model_info']['total_parameters']:,} parameters)")
    print("=" * 80)
    for name, result in report["dataset_reports"].items():
        parts = []
        if "classification_performance" in result:
            parts.append(f"accuracy {result['classification_performance']['accuracy']:.4f}")
        if "intersection_volume_performance" in result:
            parts.append(f"MAE {result['intersection_volume_performance']['overall_IntersectionVolume_metrics']['mae']:.3e}")
        for label, key in (("point-wise", "point_wise_permutation_consistency"), ("tetra-wise", "tetrahedron_wise_permutation_consistency")):
            consistency = result[key]
            parts.append(f"{label} consistency " + "/".join(f"{v:.3f}" for k, v in consistency.items() if k.endswith("consistency_rate")))
        parts.append(f"{result['inference_speed']['samples_per_second']:,.0f} samples/s")
        print(f"{name}\n    " + ", ".join(parts))

    print("\n--- OVERALL ---")
    if "overall_classification_metrics" in report:
        cls = report["overall_classification_metrics"]
        print(f"Overall Accuracy: {_fmt(cls['overall_accuracy'], '.4f')}")
        print(f"Weighted 80/20 Accuracy: {_fmt(cls['weighted_80_20_accuracy'], '.4f')}")
        print(f"Weighted 20/80 Accuracy: {_fmt(cls['weighted_20_80_accuracy'], '.4f')}")
    if "overall_regression_metrics" in report:
        reg = report["overall_regression_metrics"]
        print(f"Overall MAE: {_fmt(reg['overall_mae'], '.6e')}")
        print(f"Overall R2: {_fmt(reg['overall_r2_score'], '.4f')}")
        print(f"Overall Relative MAE: {_fmt(reg['overall_relative_mae'], '.4f')}")
        print(f"Overall SMAPE: {_fmt(reg['overall_smape'], '.2f')}%")
        print(f"Overall Kappa: {_fmt(reg['overall_kappa'], '.4f')}")
        print(f"Overall Bin Accuracy: {_fmt(reg['overall_bin_accuracy'], '.4f')}")
    speed = report["inference_speed_summary"]
    print(f"Throughput: {speed['avg_throughput_samples_per_second']:,.0f} samples/s on {speed['device']} "
          f"({speed['avg_latency_per_sample_ms']:.6f} ms/sample)")
    print("=" * 80)
