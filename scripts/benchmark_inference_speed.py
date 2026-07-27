"""
TetrahedronPairNet Inference Benchmark
Measures preprocessing and inference performance on CPU and GPU.
"""
import torch
import os
import time
import numpy as np
import pandas as pd
import json
from src.CArchitectureManager import CArchitectureManager

# Configuration
MODEL_PATH = "models_to_eval/model.pt"
CONFIG_PATH = "config/config.yaml"
EVAL_DATA_DIR = "src/evaluator/test_data"

# Benchmark settings
BATCH_SIZE = 2048
REPEAT = 30
DTYPE = torch.float32



def apply_principal_axis_transformation_batch(data_tensor):
    """
    Vectorized principal axis transformation for batch processing.
    Centers on T1's centroid and rotates to align with principal axes.
    
    Args:
        data_tensor: torch.Tensor of shape (batch_size, 24)
        
    Returns:
        torch.Tensor of shape (batch_size, 24) with transformed data
    """
    batch_size = data_tensor.shape[0]
    
    # Reshape and extract tetrahedra
    tetrahedra = data_tensor.reshape(batch_size, 2, 4, 3)
    tetra1 = tetrahedra[:, 0]
    tetra2 = tetrahedra[:, 1]
    
    # Center on T1's centroid
    centroid = tetra1.mean(dim=1, keepdim=True)
    tetra1_centered = tetra1 - centroid
    tetra2_centered = tetra2 - centroid
    
    # Compute covariance and eigendecomposition
    cov = torch.bmm(tetra1_centered.transpose(1, 2), tetra1_centered)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # Select eigenvectors in descending order of eigenvalues
    sorted_idx = torch.argsort(eigenvalues, dim=1, descending=True)
    R = torch.gather(eigenvectors, 2, sorted_idx.unsqueeze(1).expand(-1, 3, -1))
    
    # Apply rotation: R.T @ X
    R_T = R.transpose(1, 2)
    tetra1_transformed = torch.bmm(R_T, tetra1_centered.transpose(1, 2)).transpose(1, 2)
    tetra2_transformed = torch.bmm(R_T, tetra2_centered.transpose(1, 2)).transpose(1, 2)
    
    # Combine and flatten
    result = torch.stack([tetra1_transformed, tetra2_transformed], dim=1)
    return result.reshape(batch_size, 24)

def apply_unitary_tetrahedron_transformation_batch(data_tensor):
    """
    Vectorized unitary tetrahedron transformation for batch processing.
    Transforms T2 coordinates using T1 as the basis.
    
    Args:
        data_tensor: torch.Tensor of shape (batch_size, 24)
        
    Returns:
        torch.Tensor of shape (batch_size, 12) with transformed T2 coordinates
    """
    batch_size = data_tensor.shape[0]
    device = data_tensor.device
    dtype = data_tensor.dtype
    
    # Reshape and extract tetrahedra
    tetrahedra = data_tensor.reshape(batch_size, 2, 4, 3)
    tetra1 = tetrahedra[:, 0]
    tetra2 = tetrahedra[:, 1]
    
    # Extract reference point and edge vectors from T1
    v0 = tetra1[:, 0, :]
    edges = tetra1[:, 1:, :] - v0.unsqueeze(1)
    
    # Compute inverse transformation, handling singular matrices
    try:
        inv_transform = torch.linalg.inv(edges)
    except RuntimeError:
        # Fallback: compute inverses individually
        inv_transform = torch.zeros_like(edges)
        for i in range(batch_size):
            try:
                inv_transform[i] = torch.linalg.inv(edges[i])
            except RuntimeError:
                inv_transform[i] = torch.eye(3, dtype=dtype, device=device)
    
    # Transform T2 coordinates
    tetra2_translated = tetra2 - v0.unsqueeze(1)
    tetra2_transformed = torch.bmm(tetra2_translated, inv_transform)
    
    return tetra2_transformed.reshape(batch_size, 12)


class ModelWrapper(torch.nn.Module):
    """Wrapper for TorchScript models with predict interface."""
    
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.task = getattr(model, 'task', None) or config.get('model_config', {}).get('task', 'IntersectionStatus_IntersectionVolume')
        self.volume_scale_factor = getattr(model, 'volume_scale_factor', None) or config.get('model_config', {}).get('volume_scale_factor', 1000.0)
    
    def predict(self, x):
        """Run inference and return predictions based on task type."""
        with torch.no_grad():
            output = self.model(x)
            
            if self.task == 'IntersectionStatus':
                return (output > 0.5).int().squeeze()
            elif self.task == 'IntersectionVolume':
                return output.squeeze() / self.volume_scale_factor
            elif self.task == 'IntersectionStatus_IntersectionVolume':
                cls_pred = (output[:, 0] > 0.5).int().squeeze()
                reg_pred = output[:, 1].squeeze() / self.volume_scale_factor
                return torch.stack([cls_pred, reg_pred], dim=1)
            
            raise ValueError(f"Unknown task: {self.task}")
    
    def forward(self, x):
        return self.model(x)


def load_model(device):
    """Load and prepare model for inference."""
    import yaml
    
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        # Try loading as TorchScript
        model = torch.jit.load(MODEL_PATH, map_location=device)
        model = ModelWrapper(model, config)
    except Exception:
        # Fallback to state_dict loading
        arch_manager = CArchitectureManager(config['model_config'])
        model = arch_manager.get_model()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    model.to(device)
    # Convert model to the appropriate dtype
    if DTYPE == torch.float64:
        model.double()
    else:
        model.float()
    model.eval()
    return model


def get_eval_files():
    """Get sorted list of evaluation CSV files."""
    files = []
    for root, _, filenames in os.walk(EVAL_DATA_DIR):
        for fn in filenames:
            if fn.endswith('.csv'):
                files.append(os.path.join(root, fn))
    return sorted(files)


def load_raw_data(files):
    """Load raw coordinate data and labels from CSV files (I/O not timed)."""
    raw_data = {}
    np_dtype = np.float64 if DTYPE == torch.float64 else np.float32
    
    for file in files:
        df = pd.read_csv(file)
        raw_features = df.iloc[:, :24].values.astype(np_dtype)
        
        # Load ground truth labels (column 24: IntersectionVolume, column 25: HasIntersection)
        gt_volume = df.iloc[:, 24].values.astype(np_dtype)
        gt_status = df.iloc[:, 25].values.astype(np.int32)
        
        raw_data[file] = {
            'features': raw_features,
            'gt_status': gt_status,
            'gt_volume': gt_volume,
            'n': df.shape[0]
        }
    return raw_data


def preprocess_on_device(raw_tensor, device):
    """
    Apply preprocessing transformations on specified device.
    Transformations: principal_axis -> unitary_tetrahedron
    
    Args:
        raw_tensor: Tensor of shape (n, 24) on target device
        device: torch.device (cpu or cuda)
    
    Returns:
        Transformed tensor of shape (n, 12)
    """
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    transformed = apply_principal_axis_transformation_batch(raw_tensor)
    transformed = apply_unitary_tetrahedron_transformation_batch(transformed)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    return transformed


def time_preprocessing(raw_tensor, device, repeat):
    """Time preprocessing transformations."""
    # Warmup
    _ = preprocess_on_device(raw_tensor, device)
    
    # Timed runs
    times = []
    for _ in range(repeat):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        preprocessed = preprocess_on_device(raw_tensor, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        times.append(t1 - t0)
    
    return preprocessed, np.mean(times), np.std(times)


def time_inference(model, batches, device, repeat):
    """Time model inference."""
    # Warmup
    with torch.no_grad():
        for batch in batches:
            _ = model.predict(batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(repeat):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        with torch.no_grad():
            for batch in batches:
                _ = model.predict(batch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        times.append(t1 - t0)
    
    return np.mean(times), np.std(times)


def run_benchmark(raw_data, device, model, batch_size=BATCH_SIZE, repeat=REPEAT):
    """
    Run full benchmark: preprocessing + inference on specified device.
    
    Returns dict with timing results and accuracy metrics.
    """
    results = {}
    total_samples = 0
    total_preprocess_time = 0.0
    total_inference_time = 0.0
    
    # For accuracy calculation
    all_gt_status = []
    all_gt_volume = []
    all_pred_status = []
    all_pred_volume = []
    
    for file, data in raw_data.items():
        n = data['n']
        total_samples += n
        
        # Convert to tensor on target device
        raw_tensor = torch.tensor(data['features'], dtype=DTYPE, device=device)
        
        # Time preprocessing
        preprocessed, avg_pre_time, std_pre_time = time_preprocessing(raw_tensor, device, repeat)
        
        # Prepare batches for inference
        loader = torch.utils.data.DataLoader(preprocessed, batch_size=batch_size, shuffle=False)
        batches = list(loader)
        
        # Time inference
        avg_inf_time, std_inf_time = time_inference(model, batches, device, repeat)
        
        # Get predictions for accuracy (single pass, not timed)
        predictions = []
        with torch.no_grad():
            for batch in batches:
                pred = model.predict(batch)
                predictions.append(pred.cpu())
        predictions = torch.cat(predictions, dim=0)
        
        # Store ground truth and predictions
        all_gt_status.append(data['gt_status'])
        all_gt_volume.append(data['gt_volume'])
        all_pred_status.append(predictions[:, 0].numpy())
        all_pred_volume.append(predictions[:, 1].numpy())
        
        total_preprocess_time += avg_pre_time
        total_inference_time += avg_inf_time
        
        # Store per-file results
        total_file_time = avg_pre_time + avg_inf_time
        results[os.path.basename(file)] = {
            "samples": int(n),
            "preprocessing_time_s": float(avg_pre_time),
            "preprocessing_time_std_s": float(std_pre_time),
            "preprocessing_latency_ms": float(1000 * avg_pre_time / n),
            "inference_time_s": float(avg_inf_time),
            "inference_time_std_s": float(std_inf_time),
            "inference_latency_ms": float(1000 * avg_inf_time / n),
            "total_time_s": float(total_file_time),
            "total_latency_ms": float(1000 * total_file_time / n),
            "throughput_samples_per_s": float(n / total_file_time)
        }
        
        print(f"  {os.path.basename(file)}: preprocess={avg_pre_time:.4f}s, inference={avg_inf_time:.4f}s")
    
    # Calculate accuracy metrics
    all_gt_status = np.concatenate(all_gt_status)
    all_gt_volume = np.concatenate(all_gt_volume)
    all_pred_status = np.concatenate(all_pred_status)
    all_pred_volume = np.concatenate(all_pred_volume)
    
    # Classification accuracy (all samples)
    classification_accuracy = np.mean(all_gt_status == all_pred_status) * 100
    
    # Regression metrics (only for polyhedron/volumetric intersections)
    # Find indices from polyhedron intersection file
    polyhedron_indices = []
    sample_idx = 0
    for file, data in raw_data.items():
        file_size = data['n']
        if 'polyhedron' in os.path.basename(file).lower():
            polyhedron_indices.extend(range(sample_idx, sample_idx + file_size))
        sample_idx += file_size
    
    if len(polyhedron_indices) > 0:
        polyhedron_indices = np.array(polyhedron_indices)
        gt_vol_polyhedron = all_gt_volume[polyhedron_indices]
        pred_vol_polyhedron = all_pred_volume[polyhedron_indices]
        gt_status_polyhedron = all_gt_status[polyhedron_indices]
        
        # Filter for positive samples (actual intersections)
        positive_mask = gt_status_polyhedron == 1
        if positive_mask.sum() > 0:
            gt_vol_positive = gt_vol_polyhedron[positive_mask]
            pred_vol_positive = pred_vol_polyhedron[positive_mask]
            
            # Both GT and predictions are in original space
            # Compare them directly in original space
            
            # Mean Absolute Error (original space)
            mae = np.mean(np.abs(gt_vol_positive - pred_vol_positive))
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((gt_vol_positive - pred_vol_positive) / (gt_vol_positive + 1e-10))) * 100
        else:
            mae = 0.0
            mape = 0.0
    else:
        mae = 0.0
        mape = 0.0
    
    # Calculate overall summary
    total_time = total_preprocess_time + total_inference_time
    dtype_name = "float64" if DTYPE == torch.float64 else "float32"
    
    results["summary"] = {
        "total_samples": int(total_samples),
        "preprocessing_time_s": float(total_preprocess_time),
        "preprocessing_latency_ms": float(1000 * total_preprocess_time / total_samples),
        "preprocessing_throughput_samples_per_s": float(total_samples / total_preprocess_time),
        "inference_time_s": float(total_inference_time),
        "inference_latency_ms": float(1000 * total_inference_time / total_samples),
        "inference_throughput_samples_per_s": float(total_samples / total_inference_time),
        "total_time_s": float(total_time),
        "total_latency_ms": float(1000 * total_time / total_samples),
        "total_throughput_samples_per_s": float(total_samples / total_time),
        "device": str(device),
        "batch_size": int(batch_size),
        "repeat_measurements": int(repeat),
        "dtype": dtype_name,
        "classification_accuracy_pct": float(classification_accuracy),
        "regression_mae": float(mae),
        "regression_mape_pct": float(mape)
    }
    
    return results


def print_comparison_table(cpu_summary, cuda_summary):
    """Print comparison table between CPU and GPU results."""
    print(f"{'Metric':<35} {'CPU':>18} {'GPU':>18} {'GPU Speedup':>12}")
    print(f"{'-'*83}")
    
    # Preprocessing
    cpu_pre = cpu_summary["preprocessing_latency_ms"]
    cpu_pre_tp = cpu_summary["preprocessing_throughput_samples_per_s"]
    cuda_pre = cuda_summary["preprocessing_latency_ms"]
    cuda_pre_tp = cuda_summary["preprocessing_throughput_samples_per_s"]
    pre_speedup = cpu_pre / cuda_pre if cuda_pre > 0 else 0
    
    print(f"{'Preprocessing latency (ms)':<35} {cpu_pre:>17.4f} {cuda_pre:>17.4f} {pre_speedup:>11.2f}x")
    print(f"{'Preprocessing throughput (s/s)':<35} {cpu_pre_tp:>17,.0f} {cuda_pre_tp:>17,.0f}")
    print()
    
    # Inference
    cpu_inf = cpu_summary["inference_latency_ms"]
    cpu_inf_tp = cpu_summary["inference_throughput_samples_per_s"]
    cuda_inf = cuda_summary["inference_latency_ms"]
    cuda_inf_tp = cuda_summary["inference_throughput_samples_per_s"]
    inf_speedup = cpu_inf / cuda_inf if cuda_inf > 0 else 0
    
    print(f"{'Inference latency (ms)':<35} {cpu_inf:>17.4f} {cuda_inf:>17.4f} {inf_speedup:>11.2f}x")
    print(f"{'Inference throughput (s/s)':<35} {cpu_inf_tp:>17,.0f} {cuda_inf_tp:>17,.0f}")
    print()
    
    # Total
    cpu_total = cpu_summary["total_latency_ms"]
    cpu_total_tp = cpu_summary["total_throughput_samples_per_s"]
    cuda_total = cuda_summary["total_latency_ms"]
    cuda_total_tp = cuda_summary["total_throughput_samples_per_s"]
    total_speedup = cpu_total / cuda_total if cuda_total > 0 else 0
    
    print(f"{'TOTAL latency (ms)':<35} {cpu_total:>17.4f} {cuda_total:>17.4f} {total_speedup:>11.2f}x")
    print(f"{'TOTAL throughput (samples/s)':<35} {cpu_total_tp:>17,.0f} {cuda_total_tp:>17,.0f}")
    print()
    
    # Accuracy metrics
    print(f"{'-'*83}")
    print(f"{'ACCURACY METRICS':<35} {'CPU':>18} {'GPU':>18} {'Difference':>12}")
    print(f"{'-'*83}")
    
    cpu_cls_acc = cpu_summary["classification_accuracy_pct"]
    cuda_cls_acc = cuda_summary["classification_accuracy_pct"]
    cls_diff = abs(cpu_cls_acc - cuda_cls_acc)
    
    print(f"{'Classification Accuracy (%)':<35} {cpu_cls_acc:>17.2f} {cuda_cls_acc:>17.2f} {cls_diff:>11.4f}")
    
    cpu_mae = cpu_summary["regression_mae"]
    cuda_mae = cuda_summary["regression_mae"]
    mae_diff = abs(cpu_mae - cuda_mae)
    
    print(f"{'Regression MAE':<35} {cpu_mae:>17.6f} {cuda_mae:>17.6f} {mae_diff:>11.8f}")
    
    cpu_mape = cpu_summary["regression_mape_pct"]
    cuda_mape = cuda_summary["regression_mape_pct"]
    mape_diff = abs(cpu_mape - cuda_mape)
    
    print(f"{'Regression MAPE (%)':<35} {cpu_mape:>17.2f} {cuda_mape:>17.2f} {mape_diff:>11.4f}")
    
    return {
        'cpu_pre': cpu_pre, 'cpu_pre_tp': cpu_pre_tp,
        'cpu_inf': cpu_inf, 'cpu_inf_tp': cpu_inf_tp,
        'cpu_total': cpu_total, 'cpu_total_tp': cpu_total_tp,
        'cuda_pre': cuda_pre, 'cuda_pre_tp': cuda_pre_tp,
        'cuda_inf': cuda_inf, 'cuda_inf_tp': cuda_inf_tp,
        'cuda_total': cuda_total, 'cuda_total_tp': cuda_total_tp,
        'pre_speedup': pre_speedup,
        'inf_speedup': inf_speedup,
        'total_speedup': total_speedup,
        'cpu_cls_acc': cpu_cls_acc,
        'cuda_cls_acc': cuda_cls_acc,
        'cpu_mae': cpu_mae,
        'cuda_mae': cuda_mae,
        'cpu_mape': cpu_mape,
        'cuda_mape': cuda_mape
    }


def save_summary(metrics, total_samples):
    """Save human-readable summary with LaTeX table."""
    dtype_name = "float64" if DTYPE == torch.float64 else "float32"
    
    lines = []
    lines.append("=" * 70)
    lines.append(f"TETRAHEDRONPAIRNET BENCHMARK RESULTS ({dtype_name.upper()})")
    lines.append("=" * 70)
    lines.append(f"Batch size: {BATCH_SIZE}, Repeat: {REPEAT}")
    lines.append(f"Total samples: {total_samples}")
    lines.append("")
    
    lines.append("-" * 70)
    lines.append(f"{'Metric':<40} {'CPU':>14} {'GPU':>14}")
    lines.append("-" * 70)
    
    lines.append(f"{'Preprocessing latency (ms/sample)':<40} {metrics['cpu_pre']:>13.4f} {metrics['cuda_pre']:>13.4f}")
    lines.append(f"{'Preprocessing throughput (samples/s)':<40} {metrics['cpu_pre_tp']:>13,.0f} {metrics['cuda_pre_tp']:>13,.0f}")
    lines.append("")
    
    lines.append(f"{'Inference latency (ms/sample)':<40} {metrics['cpu_inf']:>13.4f} {metrics['cuda_inf']:>13.4f}")
    lines.append(f"{'Inference throughput (samples/s)':<40} {metrics['cpu_inf_tp']:>13,.0f} {metrics['cuda_inf_tp']:>13,.0f}")
    lines.append("")
    
    lines.append(f"{'TOTAL latency (ms/sample)':<40} {metrics['cpu_total']:>13.4f} {metrics['cuda_total']:>13.4f}")
    lines.append(f"{'TOTAL throughput (samples/s)':<40} {metrics['cpu_total_tp']:>13,.0f} {metrics['cuda_total_tp']:>13,.0f}")
    lines.append("-" * 70)
    
    lines.append("")
    lines.append("GPU SPEEDUPS:")
    lines.append(f"  Preprocessing: {metrics['pre_speedup']:.2f}x faster")
    lines.append(f"  Inference:     {metrics['inf_speedup']:.2f}x faster")
    lines.append(f"  Total:         {metrics['total_speedup']:.2f}x faster")
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("ACCURACY METRICS:")
    lines.append("-" * 70)
    lines.append(f"{'Metric':<40} {'CPU':>14} {'GPU':>14}")
    lines.append("-" * 70)
    lines.append(f"{'Classification Accuracy (%)':<40} {metrics['cpu_cls_acc']:>13.2f} {metrics['cuda_cls_acc']:>13.2f}")
    lines.append(f"{'Regression MAE':<40} {metrics['cpu_mae']:>13.6f} {metrics['cuda_mae']:>13.6f}")
    lines.append(f"{'Regression MAPE (%)':<40} {metrics['cpu_mape']:>13.2f} {metrics['cuda_mape']:>13.2f}")
    lines.append("-" * 70)
    lines.append("")
    
    # LaTeX format
    lines.append("=" * 70)
    lines.append("LATEX TABLE VALUES:")
    lines.append("=" * 70)
    lines.append("")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{l}{\textit{TetrahedronPairNet Performance (Batch Size: " + str(BATCH_SIZE) + r", Float32)}} \\")
    lines.append(r"\hspace{3mm} Preprocessing (CPU) & \multicolumn{3}{c|}{" + f"{metrics['cpu_pre']:.4f}" + r" ms} & Data transformation \\")
    lines.append(r"\hspace{3mm} Preprocessing (GPU) & \multicolumn{3}{c|}{" + f"{metrics['cuda_pre']:.4f}" + r" ms} & " + f"{metrics['pre_speedup']:.1f}x speedup" + r" \\")
    lines.append(r"\hspace{3mm} Inference (CPU) & \multicolumn{3}{c|}{" + f"{metrics['cpu_inf']:.4f}" + r" ms} & Forward pass only \\")
    lines.append(r"\hspace{3mm} Inference (GPU) & \multicolumn{3}{c|}{\textbf{" + f"{metrics['cuda_inf']:.4f}" + r" ms (" + f"{metrics['cuda_inf_tp']:,.0f}" + r" samples/s)}} & \textbf{" + f"{metrics['inf_speedup']:.1f}x" + r" speedup} \\")
    lines.append(r"\hspace{3mm} Total (GPU) & \multicolumn{3}{c|}{\textbf{" + f"{metrics['cuda_total']:.4f}" + r" ms (" + f"{metrics['cuda_total_tp']:,.0f}" + r" samples/s)}} & \textbf{" + f"{metrics['total_speedup']:.1f}x" + r" speedup vs CPU} \\")
    lines.append(r"\hspace{3mm} Accuracy & \multicolumn{3}{c|}{" + f"{metrics['cuda_cls_acc']:.2f}" + r"\% classification} & " + f"MAE: {metrics['cuda_mae']:.6f}" + r" \\")
    lines.append(r"\bottomrule")
    lines.append("")
    lines.append("=" * 70)
    
    with open("benchmark_summary.txt", "w") as f:
        f.write("\n".join(lines))
    
    print("\n" + "\n".join(lines))


def main():
    dtype_name = "float64" if DTYPE == torch.float64 else "float32"
    
    print(f"\n{'='*70}")
    print(f"TETRAHEDRONPAIRNET BENCHMARK ({dtype_name.upper()})")
    print(f"{'='*70}")
    print(f"Batch size: {BATCH_SIZE}, Repeat: {REPEAT}")
    
    # Load raw data
    print("\nLoading raw data from CSV files...")
    files = get_eval_files()
    raw_data = load_raw_data(files)
    total_samples = sum(d['n'] for d in raw_data.values())
    print(f"Loaded {total_samples} samples from {len(files)} files\n")
    
    # CPU benchmark
    print(f"{'='*70}")
    print("CPU BENCHMARK (Preprocessing + Inference on CPU)")
    print(f"{'='*70}")
    
    device_cpu = torch.device("cpu")
    model_cpu = load_model(device_cpu)
    results_cpu = run_benchmark(raw_data, device_cpu, model_cpu)
    
    with open("inference_metrics_cpu.json", "w") as f:
        json.dump(results_cpu, f, indent=2)
    
    # GPU benchmark
    if not torch.cuda.is_available():
        print("\nCUDA not available, skipping GPU benchmark")
        return
    
    print(f"\n{'='*70}")
    print("GPU BENCHMARK (Preprocessing + Inference on GPU)")
    print(f"{'='*70}")
    
    device_cuda = torch.device("cuda")
    model_cuda = load_model(device_cuda)
    results_cuda = run_benchmark(raw_data, device_cuda, model_cuda)
    
    with open("inference_metrics_cuda.json", "w") as f:
        json.dump(results_cuda, f, indent=2)
    
    del model_cuda
    torch.cuda.empty_cache()
    
    # Comparison and summary
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}\n")
    
    metrics = print_comparison_table(results_cpu["summary"], results_cuda["summary"])
    
    print(f"\n{'='*70}")
    
    # Save all results
    all_results = {"cpu": results_cpu, "cuda": results_cuda}
    with open("inference_metrics_all.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\nAll results saved to inference_metrics_all.json")
    
    save_summary(metrics, total_samples)
    print("Summary saved to benchmark_summary.txt")


if __name__ == "__main__":
    main()
