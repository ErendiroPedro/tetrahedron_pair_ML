"""
Benchmark script to create a 2D heatmap of Training Data Size vs Model Size with Mean Accuracy as color.

X-Axis: Training Data Size (Log scale: 10k, 50k, 100k, 500k, 1M)
Y-Axis: Model Size (Smallest 3,042 params, Medium 19,610 params, Largest 59,165 params)
Color: Mean Accuracy over the 5 test datasets

Batch size fixed at 2048.
"""

import sys
import os

# Add project root to Python path
# Since this script is in scripts/, go up one level to reach project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import yaml
import time
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import shutil
import gc


# Training data sizes to test (log scale)
DATA_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]
DATA_SIZE_LABELS = ['10k', '50k', '100k', '500k', '1M']

# Fixed batch size
BATCH_SIZE = 2048

# Training epochs
TRAIN_EPOCHS = 50


def get_model_configs():
    """
    Define 3 model configurations: Smallest, Medium, Largest
    Based on benchmark results:
    - Smallest: ~3,042 params (multiplier ~1.33, layer size 10-11)
    - Medium: ~19,610 params (multiplier 3.0, layer size 24)
    - Largest: ~59,165 params (multiplier ~5.5, layer size 44)
    """
    configs = [
        {
            'name': 'Smallest',
            'multiplier': 1.5,
            'per_vertex_layers': [6, 6],
            'per_tetrahedron_layers': [6, 6],
            'per_two_tetrahedra_layers': [6, 6],
            'shared_layers': [6],
            'classification_head': [6, 1],
            'regression_head': [6, 1],
            'vertices_aggregation_function': 'max',
            'tetrahedra_aggregation_function': 'max',
        },
        {
            'name': 'Medium',
            'multiplier': 3.0,
            'per_vertex_layers': [24, 24, 24],
            'per_tetrahedron_layers': [24, 24, 24],
            'per_two_tetrahedra_layers': [24, 24, 24],
            'shared_layers': [24],
            'classification_head': [24, 1],
            'regression_head': [24, 1],
            'vertices_aggregation_function': 'max',
            'tetrahedra_aggregation_function': 'max',
        },
        {
            'name': 'Largest',
            'multiplier': 5.5,
            'per_vertex_layers': [96, 96, 96],
            'per_tetrahedron_layers': [96, 96, 96],
            'per_two_tetrahedra_layers': [96, 96, 96],
            'shared_layers': [96],
            'classification_head': [96, 1],
            'regression_head': [96, 1],
            'vertices_aggregation_function': 'max',
            'tetrahedra_aggregation_function': 'max',
        },
    ]
    return configs


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_test_datasets(test_data_path):
    """Load all test datasets for evaluation."""
    dataset_folders = [
        'no_intersection', 
        'polyhedron_intersection',
        'point_intersection', 
        'segment_intersection', 
        'polygon_intersection'
    ]
    
    datasets = []
    for folder in dataset_folders:
        folder_path = os.path.join(test_data_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found")
            continue
            
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                filepath = os.path.join(folder_path, file)
                df = pd.read_csv(filepath)
                
                # Extract features and labels
                X = df.iloc[:, :-2].values
                y_status = df['HasIntersection'].values
                y_volume = df['IntersectionVolume'].values
                
                datasets.append({
                    'name': f"{folder}/{file}",
                    'type': folder,
                    'X': torch.tensor(X, dtype=torch.float32),
                    'y_status': y_status,
                    'y_volume': y_volume,
                    'n_samples': len(df)
                })
                print(f"  Loaded {folder}/{file}: {len(df)} samples")
    
    return datasets


def evaluate_model_accuracy(model, test_datasets, device='cpu'):
    """
    Evaluate model accuracy on all test datasets.
    Returns mean accuracy across all datasets and per-dataset metrics.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    model = model.float().to(device)
    model.eval()
    
    dataset_metrics = []
    all_accuracies = []
    
    with torch.no_grad():
        for dataset in test_datasets:
            X = dataset['X'].to(device)
            y_true = dataset['y_status']
            
            # Get predictions
            output = model(X).cpu().numpy()
            
            # For combined task, first column is classification logits
            if output.shape[1] == 2:
                y_pred_proba = 1 / (1 + np.exp(-output[:, 0]))  # sigmoid
            else:
                y_pred_proba = 1 / (1 + np.exp(-output.flatten()))
            
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            all_accuracies.append(accuracy)
            
            dataset_metrics.append({
                'name': dataset['name'],
                'type': dataset['type'],
                'n_samples': dataset['n_samples'],
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
    
    mean_accuracy = np.mean(all_accuracies)
    
    return {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': np.std(all_accuracies),
        'per_dataset': dataset_metrics
    }


def create_training_data(base_config_path, num_train_samples, num_val_samples, output_dir):
    """
    Create training data with specified number of samples.
    Uses the CDataProcessor to sample from raw data.
    """
    from src.CDataProcessor import CDataProcessor
    
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create processor config
    processor_config = base_config['processor_config'].copy()
    processor_config['num_train_samples'] = num_train_samples
    processor_config['num_val_samples'] = num_val_samples
    processor_config['skip_processing'] = False
    
    # Set paths - use project_root instead of home from config
    raw_data_path = processor_config['dataset_paths']['raw_data']
    if not os.path.isabs(raw_data_path):
        raw_data_path = os.path.join(project_root, raw_data_path)
    processor_config['dataset_paths']['raw_data'] = raw_data_path
    processor_config['dataset_paths']['processed_data'] = output_dir
    
    print(f"\n  Creating dataset with {num_train_samples:,} train + {num_val_samples:,} val samples...")
    
    # Process data
    processor = CDataProcessor(processor_config)
    processor.process()
    
    return output_dir


def train_model(model_config, processed_data_path, base_config_path, device):
    """Train a model with the given configuration and data."""
    from src.CArchitectureManager import CArchitectureManager
    from src.CModelTrainer import CModelTrainer
    
    # Load base config for common parameters
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Build architecture config
    arch_config = {
        'processed_data_path': processed_data_path,
        'architecture': {
            'use_model': 'tetrahedronpairnet',
            'tetrahedronpairnet': model_config
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
        'epochs': TRAIN_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': 1e-3,
        'early_stopping_patience': 10,
        'loss_function': 'default',
        'device': device  # Add device to config
    }
    
    # Create model
    arch_manager = CArchitectureManager(arch_config)
    model = arch_manager.get_model()
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"    Model parameters: {total_params:,}")
    
    # Train model
    trainer = CModelTrainer(trainer_config)
    trained_model, report, training_metrics, loss_curve = trainer.train_and_validate(model)
    
    return trained_model, total_params, training_metrics


def run_single_experiment(model_config, data_size, base_config_path, results_dir, 
                          test_datasets, train_device, temp_base_dir):
    """Run a single experiment for one model size and one data size."""
    
    model_name = model_config['name']
    data_label = DATA_SIZE_LABELS[DATA_SIZES.index(data_size)]
    
    print(f"\n{'='*70}")
    print(f"Experiment: Model={model_name}, Data Size={data_label} ({data_size:,})")
    print(f"{'='*70}")
    
    # Create temporary directory for this experiment's processed data
    temp_data_dir = os.path.join(temp_base_dir, f"{model_name}_{data_label}")
    os.makedirs(temp_data_dir, exist_ok=True)
    
    try:
        # Create training data
        val_size = max(10000, data_size // 10)  # 10% for validation, min 10k
        create_training_data(base_config_path, data_size, val_size, temp_data_dir)
        
        # Train model
        print(f"\n  Training model on {train_device}...")
        train_start = time.time()
        trained_model, total_params, training_metrics = train_model(
            model_config, temp_data_dir, base_config_path, train_device
        )
        train_time = time.time() - train_start
        print(f"  Training completed in {train_time:.1f}s")
        
        # Evaluate on test datasets (on CPU)
        print(f"\n  Evaluating on {len(test_datasets)} test datasets...")
        eval_results = evaluate_model_accuracy(trained_model, test_datasets, device='cpu')
        
        print(f"  Mean Accuracy: {eval_results['mean_accuracy']:.4f} "
              f"(±{eval_results['std_accuracy']:.4f})")
        
        # Per-dataset results
        print(f"  Per-dataset accuracies:")
        for ds in eval_results['per_dataset']:
            print(f"    {ds['type']}: {ds['accuracy']:.4f}")
        
        result = {
            'model_name': model_name,
            'model_config': {
                'per_vertex_layers': model_config['per_vertex_layers'],
                'per_tetrahedron_layers': model_config['per_tetrahedron_layers'],
                'per_two_tetrahedra_layers': model_config['per_two_tetrahedra_layers'],
                'shared_layers': model_config['shared_layers'],
            },
            'total_parameters': total_params,
            'data_size': data_size,
            'data_size_label': data_label,
            'val_size': val_size,
            'batch_size': BATCH_SIZE,
            'train_epochs': TRAIN_EPOCHS,
            'train_time_seconds': round(train_time, 2),
            'train_device': str(train_device),
            'evaluation': eval_results,
            'mean_accuracy': eval_results['mean_accuracy'],
            'final_val_loss': training_metrics.get('final_val_loss'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save individual result
        result_file = results_dir / f'{model_name}_{data_label}_results.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Results saved to: {result_file}")
        
        # Cleanup
        del trained_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return result
        
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return {
            'model_name': model_name,
            'data_size': data_size,
            'data_size_label': data_label,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
    
    finally:
        # Clean up temporary data directory
        if os.path.exists(temp_data_dir):
            shutil.rmtree(temp_data_dir, ignore_errors=True)


def main():
    """Main benchmark runner."""
    print("="*70)
    print("TetrahedronPairNet: Data Size vs Model Size Accuracy Benchmark")
    print("="*70)
    
    # Setup paths - use project_root instead of hardcoding
    base_dir = Path(project_root)
    base_config_path = base_dir / 'config' / 'config.yaml'
    results_dir = base_dir / 'artifacts' / 'data_size_benchmark_results'
    test_data_path = base_dir / 'src' / 'evaluator' / 'test_data'
    temp_base_dir = base_dir / 'artifacts' / 'temp_benchmark_data'
    
    results_dir.mkdir(parents=True, exist_ok=True)
    temp_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment settings
    model_configs = get_model_configs()
    
    print(f"\nExperiment Configuration:")
    print(f"  Training Data Sizes: {DATA_SIZE_LABELS}")
    print(f"  Model Sizes: {[c['name'] for c in model_configs]}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Training Epochs: {TRAIN_EPOCHS}")
    print(f"  Results Directory: {results_dir}")
    
    # Detect device
    train_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Training Device: {train_device}")
    if train_device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load test datasets
    print(f"\nLoading test datasets from: {test_data_path}")
    test_datasets = load_test_datasets(str(test_data_path))
    print(f"Loaded {len(test_datasets)} test datasets")
    
    # Run all experiments
    all_results = []
    total_experiments = len(model_configs) * len(DATA_SIZES)
    experiment_num = 0
    
    for model_config in model_configs:
        for data_size in DATA_SIZES:
            experiment_num += 1
            print(f"\n[Experiment {experiment_num}/{total_experiments}]")
            
            result = run_single_experiment(
                model_config=model_config,
                data_size=data_size,
                base_config_path=base_config_path,
                results_dir=results_dir,
                test_datasets=test_datasets,
                train_device=train_device,
                temp_base_dir=str(temp_base_dir)
            )
            all_results.append(result)
    
    # Save combined results
    combined_results = {
        'benchmark_info': {
            'model_configs': [c['name'] for c in model_configs],
            'data_sizes': DATA_SIZES,
            'data_size_labels': DATA_SIZE_LABELS,
            'batch_size': BATCH_SIZE,
            'train_epochs': TRAIN_EPOCHS,
            'train_device': str(train_device),
            'n_test_datasets': len(test_datasets),
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        },
        'results': all_results
    }
    
    combined_file = results_dir / 'data_size_benchmark_combined.json'
    with open(combined_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Benchmark Complete!")
    print(f"Combined results saved to: {combined_file}")
    print(f"{'='*70}")
    
    # Print summary table
    print("\nSummary: Mean Accuracy (Data Size x Model Size)")
    print("-" * 80)
    
    # Header
    header = f"{'Model':>12} | {'Params':>10} |"
    for label in DATA_SIZE_LABELS:
        header += f" {label:>8} |"
    print(header)
    print("-" * 80)
    
    # Group results by model
    for model_config in model_configs:
        model_name = model_config['name']
        model_results = [r for r in all_results if r.get('model_name') == model_name and 'error' not in r]
        
        if not model_results:
            continue
        
        params = model_results[0].get('total_parameters', 0)
        row = f"{model_name:>12} | {params:>10,} |"
        
        for data_size in DATA_SIZES:
            data_label = DATA_SIZE_LABELS[DATA_SIZES.index(data_size)]
            matching = [r for r in model_results if r.get('data_size') == data_size]
            if matching:
                accuracy = matching[0].get('mean_accuracy', 0)
                row += f" {accuracy:>7.4f} |"
            else:
                row += f" {'N/A':>7} |"
        
        print(row)
    
    print("-" * 80)
    
    # Cleanup temp directory
    if temp_base_dir.exists():
        shutil.rmtree(temp_base_dir, ignore_errors=True)
    
    return combined_results


if __name__ == '__main__':
    results = main()
