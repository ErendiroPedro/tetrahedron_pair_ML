"""
Benchmark script to create a 2D heatmap of Model Size vs Batch Size with throughput as color.
Tests 10 model configurations across 5 batch sizes.
Each model is trained for 20 epochs before evaluation.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Also add src directory for local imports
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import yaml
import time
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path


# Batch sizes to test
BATCH_SIZES = [4086, 2048, 1024, 512, 256, 128]

# Benchmark configuration
DTYPE = torch.float32  # torch.float32 or torch.float64
INFERENCE_DEVICE = 'cuda'  # 'cpu' or 'cuda'


def generate_layer_configs(num_configs=10):
    """Generate network configurations with varying sizes."""
    configs = []
    
    # Scale from tiny to large networks
    multipliers = np.linspace(0.5, 8, num_configs)
    base_size = 8
    
    for i, mult in enumerate(multipliers):
        layer_size = max(4, int(base_size * mult))
        
        # Scale depth for larger networks
        if mult < 2:
            depth = 2
        elif mult < 4:
            depth = 3
        else:
            depth = 4
        
        config = {
            'config_id': i + 1,
            'multiplier': round(mult, 2),
            'per_vertex_layers': [layer_size] * depth,
            'per_tetrahedron_layers': [layer_size] * depth,
            'per_two_tetrahedra_layers': [layer_size] * depth,
            'shared_layers': [layer_size],
            'classification_head': [layer_size, 1],
            'regression_head': [layer_size, 1],
            'vertices_aggregation_function': 'max',
            'tetrahedra_aggregation_function': 'max',
        }
        configs.append(config)
    
    return configs


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_test_datasets(test_data_path):
    """Load the test datasets for inference benchmarking."""
    import pandas as pd
    
    dataset_folders = [
        'no_intersection', 
        'polyhedron_intersection',
        'point_intersection', 
        'segment_intersection', 
        'polygon_intersection'
    ]
    
    all_features = []
    for folder in dataset_folders:
        folder_path = os.path.join(test_data_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found")
            continue
            
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                filepath = os.path.join(folder_path, file)
                df = pd.read_csv(filepath)
                # Extract features (all columns except last 2)
                features = df.iloc[:, :-2].values
                all_features.append(features)
    
    if all_features:
        # Concatenate all features
        combined = np.vstack(all_features)
        np_dtype = np.float64 if DTYPE == torch.float64 else np.float32
        return torch.tensor(combined.astype(np_dtype), dtype=DTYPE)
    return None


def measure_inference_speed(model, test_data, batch_size=2048, 
                            warmup_runs=10, benchmark_runs=100):
    """Measure inference speed using real test data."""
    model.eval()
    device = next(model.parameters()).device
    
    # Prepare batch of real test data
    n_samples = min(batch_size, len(test_data))
    indices = np.random.choice(len(test_data), n_samples, replace=False)
    x = test_data[indices].to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return {
        'batch_size': n_samples,
        'total_time_ms': round(avg_time, 4),
        'std_ms': round(std_time, 4),
        'time_per_sample_ms': round(avg_time / n_samples, 6),
        'throughput_samples_per_sec': round(n_samples / (avg_time / 1000), 2)
    }


def get_model_size_mb(model):
    """Calculate model size in megabytes."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def run_benchmark(config_id, tetra_config, base_config_path, results_dir, test_data, train_epochs=20):
    """Run benchmark for a single configuration across all batch sizes."""
    from src.CArchitectureManager import CArchitectureManager
    from src.CModelTrainer import CModelTrainer
    
    print(f"\n{'='*60}")
    print(f"Config {config_id}: multiplier={tetra_config['multiplier']}")
    print(f"Layer sizes: {tetra_config['per_vertex_layers']}")
    print(f"{'='*60}")
    
    # Load base config to get paths
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    home = base_config.get('home')
    processed_data_path = base_config['processor_config']['dataset_paths']['processed_data']
    if not os.path.isabs(processed_data_path):
        processed_data_path = os.path.join(home, processed_data_path)
    
    # Build model config
    model_config = {
        'processed_data_path': processed_data_path,
        'architecture': {
            'use_model': 'tetrahedronpairnet',
            'tetrahedronpairnet': tetra_config
        },
        'common_parameters': {
            'activation_function': 'relu',
            'dropout_rate': 0.0,
            'task': 'IntersectionStatus_IntersectionVolume',
            'volume_scale_factor': 1000.0
        }
    }
    
    # Training config
    trainer_config = {
        'processed_data_path': processed_data_path,
        'task': 'IntersectionStatus_IntersectionVolume',
        'epochs': train_epochs,
        'batch_size': 1024,
        'learning_rate': 1e-3,
        'early_stopping_patience': 10,
        'loss_function': 'default'
    }
    
    try:
        # Create model
        arch_manager = CArchitectureManager(model_config)
        model = arch_manager.get_model()
        
        # Use GPU for training if available
        train_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Measure parameters and size before training
        total_params, trainable_params = count_parameters(model)
        model_size_mb = get_model_size_mb(model)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Model size: {model_size_mb:.4f} MB")
        
        # TRAIN THE MODEL (on GPU if available)
        print(f"\n--- Training for {train_epochs} epochs on {train_device} ---")
        train_start = time.time()
        
        trainer = CModelTrainer(trainer_config, device=train_device)
        trained_model, report, training_metrics, loss_curve = trainer.train_and_validate(model)
        
        train_time = time.time() - train_start
        print(f"Training completed in {train_time:.1f}s")
        
        # Move model to inference device and convert dtype
        inference_device = torch.device(INFERENCE_DEVICE if INFERENCE_DEVICE == 'cpu' or torch.cuda.is_available() else 'cpu')
        print(f"\n--- Moving model to {inference_device} for inference benchmarking ---")
        model = trained_model.to(inference_device)
        
        # Convert model to appropriate dtype
        if DTYPE == torch.float64:
            model.double()
        else:
            model.float()
        
        # Ensure test data is on correct device
        test_data_device = test_data.to(inference_device)
        
        # Measure inference speed for ALL batch sizes with real test data
        print(f"--- Benchmarking inference on {inference_device} across {len(BATCH_SIZES)} batch sizes ---")
        batch_results = {}
        for batch_size in BATCH_SIZES:
            inference_result = measure_inference_speed(model, test_data_device, batch_size=batch_size)
            batch_results[batch_size] = inference_result
            print(f"  Batch {batch_size}: {inference_result['throughput_samples_per_sec']:,.0f} samples/sec")
        
        result = {
            'config_id': config_id,
            'multiplier': tetra_config['multiplier'],
            'layer_config': {
                'per_vertex_layers': tetra_config['per_vertex_layers'],
                'per_tetrahedron_layers': tetra_config['per_tetrahedron_layers'],
                'per_two_tetrahedra_layers': tetra_config['per_two_tetrahedra_layers'],
                'shared_layers': tetra_config['shared_layers'],
            },
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': round(model_size_mb, 4),
            'train_device': str(train_device),
            'inference_device': str(inference_device),
            'dtype': str(DTYPE).split('.')[-1],  # 'float32' or 'float64'
            'training': {
                'epochs': train_epochs,
                'train_time_seconds': round(train_time, 2),
                'final_val_loss': training_metrics.get('final_val_loss') if training_metrics else None,
            },
            'inference_by_batch_size': batch_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save individual result
        result_file = results_dir / f'config_{config_id:02d}_results.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Results saved to: {result_file}")
        
        del model
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        import traceback
        print(f"Error benchmarking config {config_id}: {e}")
        traceback.print_exc()
        return {
            'config_id': config_id,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Main benchmark runner."""
    dtype_name = "float64" if DTYPE == torch.float64 else "float32"
    
    print("="*60)
    print("TetrahedronPairNet: Model Size vs Batch Size Heatmap Benchmark")
    print("="*60)
    print(f"Dtype: {dtype_name}")
    print(f"Inference device: {INFERENCE_DEVICE}")
    
    # Setup paths
    base_dir = Path('/home/sei/tetrahedron_pair_ML')
    base_config_path = base_dir / 'config' / 'config.yaml'
    results_dir = base_dir / 'artifacts' / 'benchmark_results'
    test_data_path = base_dir / 'src' / 'evaluator' / 'test_data'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Benchmark settings
    num_configs = 10
    train_epochs = 20
    
    # Load test data for inference benchmarking
    print("\nLoading test datasets for inference benchmarking...")
    test_data = load_test_datasets(str(test_data_path))
    if test_data is not None:
        print(f"Loaded {len(test_data):,} samples from test sets ({dtype_name})")
    else:
        print("Warning: No test data found, using random data")
        test_data = torch.randn(10000, 24, dtype=DTYPE)
    
    # Generate configurations
    configs = generate_layer_configs(num_configs)
    
    print(f"\nGenerated {len(configs)} model configurations")
    print(f"Batch sizes to test: {BATCH_SIZES}")
    print(f"Each model will be trained for {train_epochs} epochs")
    
    train_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference_device = torch.device(INFERENCE_DEVICE if INFERENCE_DEVICE == 'cpu' or torch.cuda.is_available() else 'cpu')
    
    print(f"Training device: {train_device}")
    if train_device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"Inference device: {inference_device}")
    print(f"Results will be saved to: {results_dir}")
    
    # Run benchmarks
    all_results = []
    
    for config in configs:
        result = run_benchmark(
            config_id=config['config_id'],
            tetra_config=config,
            base_config_path=base_config_path,
            results_dir=results_dir,
            test_data=test_data,
            train_epochs=train_epochs
        )
        all_results.append(result)
    
    # Save combined results
    combined_results = {
        'benchmark_info': {
            'num_configs': num_configs,
            'train_epochs': train_epochs,
            'train_device': str(train_device),
            'inference_device': str(inference_device),
            'dtype': dtype_name,
            'batch_sizes': BATCH_SIZES,
            'test_samples': len(test_data),
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        },
        'results': all_results
    }
    
    combined_file = results_dir / 'benchmark_combined_results.json'
    with open(combined_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Benchmark Complete!")
    print(f"Combined results saved to: {combined_file}")
    print(f"{'='*60}")
    
    # Print summary table
    print("\nSummary (Model Size x Batch Size -> Throughput):")
    print("-" * 100)
    header = f"{'Config':>6} | {'Params':>10} | {'Size(MB)':>8} |"
    for bs in BATCH_SIZES:
        header += f" {bs:>8} |"
    print(header)
    print("-" * 100)
    
    successful_results = [r for r in all_results if 'error' not in r]
    successful_results.sort(key=lambda x: x['total_parameters'])
    
    for r in successful_results:
        row = f"{r['config_id']:>6} | {r['total_parameters']:>10,} | {r['model_size_mb']:>8.4f} |"
        for bs in BATCH_SIZES:
            throughput = r['inference_by_batch_size'].get(str(bs), {}).get('throughput_samples_per_sec', 0)
            if throughput == 0:
                throughput = r['inference_by_batch_size'].get(bs, {}).get('throughput_samples_per_sec', 0)
            row += f" {throughput/1000:>7.1f}k |"
        print(row)
    
    print("-" * 100)
    
    return combined_results


if __name__ == '__main__':
    results = main()
