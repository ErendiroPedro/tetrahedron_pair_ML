"""
TetrahedronPairNet Inference Benchmark
======================================
Measures preprocessing and inference performance on CPU and GPU.

Usage:
    python scripts/benchmark_inference_speed.py
"""
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Benchmark configuration - edit these values as needed."""
    # Paths
    model_path: str = "models_to_eval/model.pt"
    config_path: str = "config/config.yaml"
    test_data_file: str = "data/processed/train/train_data.csv"
    
    # Benchmark settings
    batch_size: int = 2048
    num_repeats: int = 30
    dtype: str = "float32"  # "float32" or "float64"
    
    # Model settings (must match the trained model)
    input_dim: int = 12  # After preprocessing: 12 features
    volume_scale_factor: float = 1000.0
    task: str = "IntersectionStatus_IntersectionVolume"
    
    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        return torch.float64 if self.dtype == "float64" else torch.float32
    
    def get_numpy_dtype(self):
        """Convert string dtype to numpy dtype."""
        return np.float64 if self.dtype == "float64" else np.float32


# Edit this configuration as needed
CONFIG = Config(
    model_path="/home/sei/tetrahedron_pair_ML/models_to_eval/model.pt",
    config_path="/home/sei/tetrahedron_pair_ML/config/config.yaml",
    test_data_file="/home/sei/tetrahedron_pair_ML/src/evaluator/test_data/polyhedron_intersection/tetrahedron_pair_polyhedron_intersection_p16_s100k_uniform_volums_0-0.01.csv",
    batch_size=2048,
    num_repeats=30,
    dtype="float32",  # Change to "float64" for double precision
    input_dim=12,
    volume_scale_factor=1000.0,
    task="IntersectionStatus_IntersectionVolume",
)


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def apply_principal_axis_transform(data: torch.Tensor) -> torch.Tensor:
    """
    Center data on T1's centroid and rotate to align with principal axes.
    
    Args:
        data: Tensor of shape (batch, 24) - raw tetrahedra coordinates
        
    Returns:
        Tensor of shape (batch, 24) - transformed coordinates
    """
    batch_size = data.shape[0]
    
    # Reshape to (batch, 2 tetrahedra, 4 vertices, 3 coords)
    tetrahedra = data.reshape(batch_size, 2, 4, 3)
    t1, t2 = tetrahedra[:, 0], tetrahedra[:, 1]
    
    # Center on T1's centroid
    centroid = t1.mean(dim=1, keepdim=True)
    t1_centered = t1 - centroid
    t2_centered = t2 - centroid
    
    # Compute rotation matrix from covariance eigendecomposition
    cov = torch.bmm(t1_centered.transpose(1, 2), t1_centered)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # Sort eigenvectors by eigenvalues (descending)
    idx = torch.argsort(eigenvalues, dim=1, descending=True)
    R = torch.gather(eigenvectors, 2, idx.unsqueeze(1).expand(-1, 3, -1))
    
    # Apply rotation
    R_T = R.transpose(1, 2)
    t1_rot = torch.bmm(R_T, t1_centered.transpose(1, 2)).transpose(1, 2)
    t2_rot = torch.bmm(R_T, t2_centered.transpose(1, 2)).transpose(1, 2)
    
    return torch.stack([t1_rot, t2_rot], dim=1).reshape(batch_size, 24)


def apply_unitary_tetrahedron_transform(data: torch.Tensor) -> torch.Tensor:
    """
    Transform T2 coordinates using T1 as the coordinate basis.
    
    Args:
        data: Tensor of shape (batch, 24) - principal-axis transformed data
        
    Returns:
        Tensor of shape (batch, 12) - T2 in T1's coordinate system
    """
    batch_size = data.shape[0]
    device, dtype = data.device, data.dtype
    
    # Reshape to (batch, 2 tetrahedra, 4 vertices, 3 coords)
    tetrahedra = data.reshape(batch_size, 2, 4, 3)
    t1, t2 = tetrahedra[:, 0], tetrahedra[:, 1]
    
    # Build transformation matrix from T1's edges
    v0 = t1[:, 0, :]
    edges = t1[:, 1:, :] - v0.unsqueeze(1)  # (batch, 3, 3)
    
    # Compute inverse (with fallback for singular matrices)
    try:
        inv_edges = torch.linalg.inv(edges)
    except RuntimeError:
        inv_edges = torch.zeros_like(edges)
        for i in range(batch_size):
            try:
                inv_edges[i] = torch.linalg.inv(edges[i])
            except RuntimeError:
                inv_edges[i] = torch.eye(3, dtype=dtype, device=device)
    
    # Transform T2
    t2_translated = t2 - v0.unsqueeze(1)
    t2_transformed = torch.bmm(t2_translated, inv_edges)
    
    return t2_transformed.reshape(batch_size, 12)


def preprocess(raw_data: torch.Tensor) -> torch.Tensor:
    """
    Full preprocessing pipeline: principal_axis -> unitary_tetrahedron.
    
    Args:
        raw_data: Tensor of shape (n, 24)
        
    Returns:
        Tensor of shape (n, 12)
    """
    data = apply_principal_axis_transform(raw_data)
    data = apply_unitary_tetrahedron_transform(data)
    return data


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(config: Config, device: torch.device) -> torch.nn.Module:
    """
    Load the trained model.
    
    Tries TorchScript first, falls back to manual architecture creation.
    """
    model_path = config.model_path
    torch_dtype = config.get_torch_dtype()
    
    # Try TorchScript loading first
    try:
        model = torch.jit.load(model_path, map_location=device)
        print(f"  Loaded TorchScript model from {model_path}")
    except Exception as e:
        print(f"  TorchScript loading failed: {e}")
        print(f"  Falling back to state_dict loading...")
        
        # Import and create model manually
        from src.CArchitectureManager import ArchitectureRegistry
        
        # Load config to get architecture settings
        with open(config.config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        model_config = yaml_config['model_config']
        arch_name = model_config['architecture']['use_model'].lower()
        arch_params = model_config['architecture'].get(arch_name, {})
        common = model_config['common_parameters']
        
        # Build full config for model creation
        full_config = {
            'input_dim': config.input_dim,
            'task': common['task'],
            'activation': common['activation_function'],
            'volume_scale_factor': common['volume_scale_factor'],
            'dropout_rate': common.get('dropout_rate', 0.0),
            **arch_params
        }
        
        # Create and load model
        model = ArchitectureRegistry.create(arch_name, full_config)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"  Loaded state_dict model ({arch_name}) from {model_path}")
    
    # Move to device and set dtype
    model.to(device=device, dtype=torch_dtype)
    model.eval()
    print(f"  Model dtype: {torch_dtype}, device: {device}")
    return model


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_data(file_path: str, config: Config) -> dict:
    """
    Load test data from CSV file.
    
    Expected format: 24 feature columns + gt_volume + gt_status
    
    Returns:
        Dictionary with 'features', 'gt_volume', 'gt_status', 'n_samples'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test data not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    if df.shape[1] < 26:
        raise ValueError(
            f"Expected 26 columns (24 features + 2 labels), got {df.shape[1]}"
        )
    
    np_dtype = config.get_numpy_dtype()
    
    return {
        'features': df.iloc[:, :24].values.astype(np_dtype),
        'gt_volume': df.iloc[:, 24].values.astype(np_dtype),
        'gt_status': df.iloc[:, 25].values.astype(np.int32),
        'n_samples': len(df),
    }


# =============================================================================
# INFERENCE
# =============================================================================

def predict(model: torch.nn.Module, x: torch.Tensor, config: Config) -> torch.Tensor:
    """Run model prediction with proper output handling."""
    with torch.no_grad():
        output = model(x)
        
        if config.task == 'IntersectionStatus':
            return (output > 0.5).int().squeeze()
        elif config.task == 'IntersectionVolume':
            return output.squeeze() / config.volume_scale_factor
        else:  # IntersectionStatus_IntersectionVolume
            cls_pred = (output[:, 0] > 0.5).int()
            vol_pred = output[:, 1] / config.volume_scale_factor
            return torch.stack([cls_pred.float(), vol_pred], dim=1)


# =============================================================================
# TIMING FUNCTIONS
# =============================================================================

def time_function(func, *args, num_repeats: int = 30, device: torch.device = None):
    """
    Time a function with warmup and multiple repeats.
    
    Returns:
        Tuple of (result, mean_time, std_time)
    """
    # Warmup
    result = func(*args)
    if device and device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(num_repeats):
        if device and device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = func(*args)
        if device and device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
    
    return result, np.mean(times), np.std(times)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark(test_data: dict, model: torch.nn.Module, 
                  device: torch.device, config: Config) -> dict:
    """
    Run full benchmark on given device.
    
    Returns:
        Dictionary with timing and accuracy metrics
    """
    n = test_data['n_samples']
    torch_dtype = config.get_torch_dtype()
    print(f"  Benchmarking {n:,} samples on {device} ({config.dtype})...")
    
    # Move data to device with correct dtype
    raw_tensor = torch.tensor(test_data['features'], device=device, dtype=torch_dtype)
    
    # Time preprocessing
    print("  - Timing preprocessing...")
    preprocessed, pre_time, pre_std = time_function(
        preprocess, raw_tensor, 
        num_repeats=config.num_repeats, device=device
    )
    print(f"    {pre_time:.4f}s ± {pre_std:.4f}s")
    
    # Prepare batches
    batches = list(torch.split(preprocessed, config.batch_size))
    
    # Time inference
    print("  - Timing inference...")
    def run_inference():
        preds = []
        for batch in batches:
            preds.append(predict(model, batch, config))
        return torch.cat(preds, dim=0)
    
    predictions, inf_time, inf_std = time_function(
        run_inference, 
        num_repeats=config.num_repeats, device=device
    )
    print(f"    {inf_time:.4f}s ± {inf_std:.4f}s")
    
    # Calculate accuracy
    print("  - Calculating accuracy...")
    predictions = predictions.cpu().numpy()
    pred_status = predictions[:, 0]
    pred_volume = predictions[:, 1]
    
    gt_status = test_data['gt_status']
    gt_volume = test_data['gt_volume']
    
    cls_accuracy = np.mean(gt_status == pred_status) * 100
    
    # Regression metrics only for intersecting pairs
    mask = gt_status == 1
    if mask.sum() > 0:
        mae = np.mean(np.abs(gt_volume[mask] - pred_volume[mask]))
        mape = np.mean(np.abs((gt_volume[mask] - pred_volume[mask]) / 
                              (gt_volume[mask] + 1e-10))) * 100
    else:
        mae, mape = 0.0, 0.0
    
    # Build results (convert all to native Python types for JSON serialization)
    total_time = pre_time + inf_time
    
    return {
        'device': str(device),
        'dtype': config.dtype,
        'n_samples': int(n),
        'batch_size': int(config.batch_size),
        'num_repeats': int(config.num_repeats),
        # Preprocessing metrics
        'preprocessing_time_s': float(pre_time),
        'preprocessing_std_s': float(pre_std),
        'preprocessing_throughput': float(n / pre_time),
        'preprocessing_latency_ms': float(1000 * pre_time / n),
        # Inference metrics
        'inference_time_s': float(inf_time),
        'inference_std_s': float(inf_std),
        'inference_throughput': float(n / inf_time),
        'inference_latency_ms': float(1000 * inf_time / n),
        # Total metrics
        'total_time_s': float(total_time),
        'total_throughput': float(n / total_time),
        'total_latency_ms': float(1000 * total_time / n),
        # Accuracy metrics
        'classification_accuracy_pct': float(cls_accuracy),
        'regression_mae': float(mae),
        'regression_mape_pct': float(mape),
    }


# =============================================================================
# REPORTING
# =============================================================================

def print_results(cpu_results: dict, gpu_results: dict = None):
    """Print benchmark results in a nice table format."""
    
    def speedup(cpu_val, gpu_val):
        return cpu_val / gpu_val if gpu_val > 0 else 0
    
    print("\n" + "=" * 75)
    print("BENCHMARK RESULTS")
    print("=" * 75)
    
    if gpu_results:
        print(f"\n{'Metric':<35} {'CPU':>15} {'GPU':>15} {'Speedup':>8}")
        print("-" * 75)
        
        # Preprocessing
        cpu_pre = cpu_results['preprocessing_latency_ms']
        gpu_pre = gpu_results['preprocessing_latency_ms']
        print(f"{'Preprocessing latency (ms)':<35} {cpu_pre:>15.4f} {gpu_pre:>15.4f} {speedup(cpu_pre, gpu_pre):>7.1f}x")
        
        cpu_pre_tp = cpu_results['preprocessing_throughput']
        gpu_pre_tp = gpu_results['preprocessing_throughput']
        print(f"{'Preprocessing throughput (s/s)':<35} {cpu_pre_tp:>15,.0f} {gpu_pre_tp:>15,.0f}")
        
        # Inference
        cpu_inf = cpu_results['inference_latency_ms']
        gpu_inf = gpu_results['inference_latency_ms']
        print(f"{'Inference latency (ms)':<35} {cpu_inf:>15.4f} {gpu_inf:>15.4f} {speedup(cpu_inf, gpu_inf):>7.1f}x")
        
        cpu_inf_tp = cpu_results['inference_throughput']
        gpu_inf_tp = gpu_results['inference_throughput']
        print(f"{'Inference throughput (s/s)':<35} {cpu_inf_tp:>15,.0f} {gpu_inf_tp:>15,.0f}")
        
        # Total
        cpu_tot = cpu_results['total_latency_ms']
        gpu_tot = gpu_results['total_latency_ms']
        print(f"{'TOTAL latency (ms)':<35} {cpu_tot:>15.4f} {gpu_tot:>15.4f} {speedup(cpu_tot, gpu_tot):>7.1f}x")
        
        cpu_tot_tp = cpu_results['total_throughput']
        gpu_tot_tp = gpu_results['total_throughput']
        print(f"{'TOTAL throughput (samples/s)':<35} {cpu_tot_tp:>15,.0f} {gpu_tot_tp:>15,.0f}")
        
        # Accuracy
        print("-" * 75)
        print(f"{'Classification Accuracy (%)':<35} {cpu_results['classification_accuracy_pct']:>15.2f} {gpu_results['classification_accuracy_pct']:>15.2f}")
        print(f"{'Regression MAE':<35} {cpu_results['regression_mae']:>15.6f} {gpu_results['regression_mae']:>15.6f}")
        print(f"{'Regression MAPE (%)':<35} {cpu_results['regression_mape_pct']:>15.2f} {gpu_results['regression_mape_pct']:>15.2f}")
    else:
        # CPU only
        print(f"\n{'Metric':<40} {'Value':>20}")
        print("-" * 62)
        print(f"{'Preprocessing latency (ms)':<40} {cpu_results['preprocessing_latency_ms']:>20.4f}")
        print(f"{'Preprocessing throughput (s/s)':<40} {cpu_results['preprocessing_throughput']:>20,.0f}")
        print(f"{'Inference latency (ms)':<40} {cpu_results['inference_latency_ms']:>20.4f}")
        print(f"{'Inference throughput (s/s)':<40} {cpu_results['inference_throughput']:>20,.0f}")
        print(f"{'TOTAL latency (ms)':<40} {cpu_results['total_latency_ms']:>20.4f}")
        print(f"{'TOTAL throughput (samples/s)':<40} {cpu_results['total_throughput']:>20,.0f}")
        print("-" * 62)
        print(f"{'Classification Accuracy (%)':<40} {cpu_results['classification_accuracy_pct']:>20.2f}")
        print(f"{'Regression MAE':<40} {cpu_results['regression_mae']:>20.6f}")
        print(f"{'Regression MAPE (%)':<40} {cpu_results['regression_mape_pct']:>20.2f}")
    
    print("=" * 75 + "\n")


def save_results(cpu_results: dict, gpu_results: dict = None, config: Config = None):
    """Save results to JSON files."""
    # Save CPU results
    with open("inference_metrics_cpu.json", "w") as f:
        json.dump(cpu_results, f, indent=2)
    
    # Save GPU results if available
    if gpu_results:
        with open("inference_metrics_cuda.json", "w") as f:
            json.dump(gpu_results, f, indent=2)
        
        # Save combined
        with open("inference_metrics_all.json", "w") as f:
            json.dump({"cpu": cpu_results, "cuda": gpu_results}, f, indent=2)
    
    print("Results saved to inference_metrics_*.json")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    config = CONFIG
    
    # Print header
    print("\n" + "=" * 70)
    print("TETRAHEDRONPAIRNET INFERENCE BENCHMARK")
    print("=" * 70)
    print(f"Model:      {config.model_path}")
    print(f"Test data:  {config.test_data_file}")
    print(f"Batch size: {config.batch_size}")
    print(f"Repeats:    {config.num_repeats}")
    print(f"Dtype:      {config.dtype}")
    
    # Load test data
    print("\n[1/4] Loading test data...")
    try:
        test_data = load_test_data(config.test_data_file, config)
        print(f"  Loaded {test_data['n_samples']:,} samples ({config.dtype})")
    except Exception as e:
        print(f"  ERROR: {e}")
        return
    
    # CPU benchmark
    print("\n[2/4] CPU Benchmark...")
    device_cpu = torch.device("cpu")
    model_cpu = load_model(config, device_cpu)
    cpu_results = run_benchmark(test_data, model_cpu, device_cpu, config)
    
    # GPU benchmark (if available)
    gpu_results = None
    if torch.cuda.is_available():
        print(f"\n[3/4] GPU Benchmark ({torch.cuda.get_device_name(0)})...")
        device_cuda = torch.device("cuda")
        model_cuda = load_model(config, device_cuda)
        gpu_results = run_benchmark(test_data, model_cuda, device_cuda, config)
        
        del model_cuda
        torch.cuda.empty_cache()
    else:
        print("\n[3/4] GPU Benchmark... SKIPPED (CUDA not available)")
    
    # Print and save results
    print("\n[4/4] Saving results...")
    print_results(cpu_results, gpu_results)
    save_results(cpu_results, gpu_results, config)
    
    print("Done!")


if __name__ == "__main__":
    main()
