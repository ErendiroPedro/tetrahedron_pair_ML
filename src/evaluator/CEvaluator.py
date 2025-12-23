import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import platform
import psutil
import subprocess
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import mean_absolute_error, cohen_kappa_score, r2_score
import warnings
warnings.filterwarnings('ignore')

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import GeometryUtils as gu
import src.CDataProcessor as dp

torch.set_default_dtype(torch.float32)

class CEvaluator:
    def __init__(self, evaluator_config, processor_config, device=None):
        self.config = evaluator_config
        self.processor_config = processor_config
        self.device = torch.device(self.config['device'])
        
        
        print(f"-- Evaluator using: {self.device} (float32) --")
        
        self.n_evaluation_runs = evaluator_config.get('n_evaluation_runs', 50)
        self.volume_bins = evaluator_config.get('volume_bins', 10)
        self.volume_range = evaluator_config.get('volume_range', None)
        
        self.task_type = evaluator_config.get('task', 'IntersectionStatus')
        self.test_data_loaded = False
        self.datasets = []
        self.intersection_status_column_name = 'HasIntersection'
        self.intersection_volume_column_name = 'IntersectionVolume'
        
        processor_config_eval = self._remove_augmentations(processor_config)
        self.dp = dp.CDataProcessor(processor_config_eval)
        
        self.test_registry = {
            'IntersectionStatus': [self.classification_performance, self.point_wise_permutation_consistency, self.tetrahedron_wise_permutation_consistency, self.inference_speed],
            'IntersectionVolume': [self.intersection_volume_performance, self.point_wise_permutation_consistency, self.tetrahedron_wise_permutation_consistency, self.inference_speed],
            'IntersectionStatus_IntersectionVolume': [self.classification_performance, self.point_wise_permutation_consistency, self.tetrahedron_wise_permutation_consistency, self.intersection_volume_performance, self.inference_speed]
        }
        
        self.dataset_folders = (
            ['polyhedron_intersection', 'point_intersection', 'segment_intersection', 'polygon_intersection'] 
            if self.task_type == 'IntersectionVolume'
            else [
                'no_intersection', 
                'polyhedron_intersection','point_intersection', 
                'segment_intersection', 'polygon_intersection'
            ]
        )
        
        print(f"Task Type: {self.task_type}")
        print(f"Dataset Folders: {self.dataset_folders}")

    def _remove_augmentations(self, config):
        config_copy = config.copy()
        if 'augmentations' in config_copy:
            augmentations = config_copy['augmentations'].copy() if config_copy['augmentations'] else {}
            
            # Remove data augmentation that changes the data
            augmentations['point_wise_permutation_augmentation_pct'] = 0
            augmentations['tetrahedron_wise_permutation_augmentation_pct'] = 0
            
            # Keep sorting augmentations (they provide consistent ordering)
            # x_sorting and y_sorting should be preserved
            
            config_copy['augmentations'] = augmentations
        return config_copy

    def evaluate(self, model, training_metrics=None):
        try:
            if isinstance(model, dict):
                raise ValueError("Model state dict passed instead of model object.")
            
            if not isinstance(model, torch.nn.Module):
                raise ValueError(f"Expected torch.nn.Module, got {type(model)}")
            
            model = model.float().to(self.device)
            model.eval()
            
            print("=" * 80)
            print("COMPREHENSIVE MODEL EVALUATION")
            print("=" * 80)
            
            system_info = self._get_system_info()
            model_info = self._get_model_info(model)
            
            self._load_test_data()
            
            report = {
                'system_info': system_info,
                'model_info': model_info,
                'task_type': self.task_type,
                'device': str(self.device),
                'n_evaluation_runs': self.n_evaluation_runs,
                'dataset_reports': {}
            }
            
            if training_metrics:
                report['training_metrics'] = training_metrics
                print(f"Training metrics included in evaluation report")
            
            for dataset in self.datasets:
                dataset_report = {}
                dataset_name = dataset['name']
                
                print(f"\n--- Evaluating {dataset_name} ---")
                
                for test in self.test_registry.get(self.task_type, []):
                    test_name = test.__name__
                    try:
                        print(f"  Running {test_name}...")
                        dataset_report[test_name] = test(model, dataset)
                    except Exception as e:
                        print(f"  Error in {test_name}: {e}")
                        dataset_report[test_name] = {'error': str(e)}
                
                report['dataset_reports'][dataset_name] = dataset_report
            
            if self.task_type in ['IntersectionStatus', 'IntersectionStatus_IntersectionVolume']:
                report['overall_classification_metrics'] = self._calculate_overall_classification_metrics(report)
                overall_auc, total_auc_samples = self._calculate_overall_auc(model)
                report['overall_classification_metrics']['overall_auc'] = overall_auc
                report['overall_classification_metrics']['total_auc_samples'] = total_auc_samples
            if self.task_type in ['IntersectionVolume', 'IntersectionStatus_IntersectionVolume']:
                report['overall_regression_metrics'] = self._calculate_overall_regression_metrics(report)
            
            report['inference_speed_summary'] = self._calculate_inference_speed_summary(report)
            
            self._print_final_report(report)
            
            print("---- Finished Evaluation ----")
            return report
            
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
        
    def _load_test_data(self):
        if self.test_data_loaded:
            return
        
        base_path = self.config['test_data_path']
        
        if not os.path.exists(base_path):
            raise ValueError(f"Test data directory not found: {base_path}")
        
        self.datasets = []
        
        for folder in self.dataset_folders:
            dataset_dir = os.path.join(base_path, folder)
            
            if not os.path.exists(dataset_dir):
                print(f"Warning: Dataset directory not found: {dataset_dir}")
                continue
            
            for file in os.listdir(dataset_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(dataset_dir, file)
                    print(f"Loading {file_path}")
                    
                    try:
                        df = pd.read_csv(file_path)
                        
                        if self.intersection_status_column_name not in df.columns:
                            print(f"Warning: {self.intersection_status_column_name} not found in {file_path}")
                            continue
                        
                        if self.intersection_volume_column_name not in df.columns:
                            print(f"Warning: {self.intersection_volume_column_name} not found in {file_path}")
                            continue
                        
                        transformation_time_ms = 0
                        augmentation_time_ms = 0

                        # Apply sorting augmentations (these should be applied during evaluation)
                        if self.processor_config.get('augmentations', None):
                            start = time.time()
                            # Only apply sorting augmentations, skip geometric augmentations
                            augmentation_config = self.processor_config.get('augmentations', {})
                            
                            from CDataProcessor import AugmentationEngine
                            df = AugmentationEngine._apply_sorting_augmentations(df, augmentation_config)
                            
                            augmentation_time_ms = (time.time() - start) * 1000
                            print(f"  Applied sorting augmentations: {augmentation_time_ms:.2f}ms")
                        if self.processor_config.get('transformations', None):
                            start = time.time()
                            df = self.dp.transform_data(df, self.processor_config)
                            transformation_time_ms = (time.time() - start) * 1000
                            print(f"  Applied transformations: {transformation_time_ms:.2f}ms")
                        
                        X = df.drop(columns=[self.intersection_status_column_name, self.intersection_volume_column_name])
                        
                        self.datasets.append({
                            'name': f"{folder}/{file}",
                            'type': folder,
                            'intersection_status': df[self.intersection_status_column_name],
                            'intersection_volume': df[self.intersection_volume_column_name],
                            'X': X,
                            'transformation_time_ms': transformation_time_ms,
                            'augmentation_time_ms': augmentation_time_ms,
                            'file_path': file_path
                        })
                        
                        print(f"  Loaded {len(df)} samples")
                        
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        continue
        
        if not self.datasets:
            raise ValueError("No valid datasets found")
        
        self.test_data_loaded = True
        print(f"Loaded {len(self.datasets)} datasets total")
        
    def classification_performance(self, model, dataset):
        try:
            X = torch.tensor(dataset['X'].values, dtype=torch.float32).to(self.device)
            y_true = dataset['intersection_status'].values
            
            print(f"    Classification on {len(X)} samples...")
            
            metrics_list = []
            
            with torch.no_grad():
                for run in range(self.n_evaluation_runs):
                    if hasattr(model, 'predict'):
                        y_pred = model.predict(X).cpu().numpy()
                    else:
                        y_pred = model(X).cpu().numpy()
                    
                    if self.task_type == 'IntersectionStatus_IntersectionVolume':
                        y_pred = y_pred[:, 0]
                    
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    
                    run_metrics = {
                        'accuracy': accuracy_score(y_true, y_pred_binary),
                        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
                        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
                        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
                        'confusion_matrix': confusion_matrix(y_true, y_pred_binary).tolist()
                    }
                    
                    try:
                        auc = roc_auc_score(y_true, y_pred)
                        run_metrics['auc'] = auc
                    except:
                        run_metrics['auc'] = 0.5
                    
                    try:
                        kappa = cohen_kappa_score(y_true, y_pred_binary)
                        run_metrics['kappa'] = kappa
                    except:
                        run_metrics['kappa'] = 0.0
                    
                    metrics_list.append(run_metrics)
            
            avg_metrics = {
                'repetitions': self.n_evaluation_runs,
                'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
                'precision': np.mean([m['precision'] for m in metrics_list]),
                'recall': np.mean([m['recall'] for m in metrics_list]),
                'f1': np.mean([m['f1'] for m in metrics_list]),
                'auc': np.mean([m['auc'] for m in metrics_list]),
                'kappa': np.mean([m['kappa'] for m in metrics_list]),
                'confusion_matrix': np.mean([m['confusion_matrix'] for m in metrics_list], axis=0).tolist(),
                'n_samples': len(y_true)
            }
            
            return avg_metrics
            
        except Exception as e:
            return {'error': str(e)}

    def _calculate_mape(self, y_true, y_pred, epsilon=1e-8):
        y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
        return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    def calculate_volume_binning_kappa(self, y_true_volumes, y_pred_volumes, num_bins=10):
        """
        Calculate Cohen's Kappa for volume predictions binned into discrete categories.
        
        Args:
            y_true_volumes: Actual intersection volumes
            y_pred_volumes: Predicted intersection volumes  
            num_bins: Number of bins for discretization (default: 10)
        
        Returns:
            kappa_score: Cohen's Kappa coefficient
            bin_accuracy: Overall binning accuracy
            confusion_matrix: Confusion matrix for debugging
            bin_info: Dictionary with binning information (NEW)
        """
        import numpy as np
        from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
        
        # Handle edge cases
        if len(y_true_volumes) == 0:
            return 0.0, 0.0, None, None
        
        # Define bin edges
        min_vol = min(y_true_volumes.min(), y_pred_volumes.min())
        max_vol = max(y_true_volumes.max(), y_pred_volumes.max())
        
        # Determine binning strategy and create bins
        use_log_scale = min_vol > 0
        if use_log_scale:
            bin_edges = np.logspace(np.log10(min_vol), np.log10(max_vol), num_bins + 1)
            binning_method = "log_scale"
            volume_span_orders = np.log10(max_vol / min_vol)
        else:
            bin_edges = np.linspace(min_vol, max_vol, num_bins + 1)
            binning_method = "linear"
            volume_span_orders = None
        
        # Discretize volumes into bins
        y_true_discrete = np.digitize(y_true_volumes, bin_edges) - 1
        y_pred_discrete = np.digitize(y_pred_volumes, bin_edges) - 1
        
        # Ensure bins are in valid range [0, num_bins-1]
        y_true_discrete = np.clip(y_true_discrete, 0, num_bins - 1)
        y_pred_discrete = np.clip(y_pred_discrete, 0, num_bins - 1)
        
        # Create bin information for interpretability
        bin_info = {
            'binning_method': binning_method,
            'num_bins': num_bins,
            'volume_range_orders_of_magnitude': volume_span_orders,
            'min_volume': float(min_vol),
            'max_volume': float(max_vol)
        }
        
        try:
            # Calculate Cohen's Kappa and accuracy
            kappa_score = cohen_kappa_score(y_true_discrete, y_pred_discrete)
            bin_accuracy = accuracy_score(y_true_discrete, y_pred_discrete)
            conf_matrix = confusion_matrix(y_true_discrete, y_pred_discrete, labels=range(num_bins))
            
            return kappa_score, bin_accuracy, conf_matrix, bin_info
            
        except Exception as e:
            print(f"Warning: Kappa calculation failed: {e}")
            try:
                bin_accuracy = accuracy_score(y_true_discrete, y_pred_discrete)
                return 0.0, bin_accuracy, None, bin_info
            except:
                return 0.0, 0.0, None, bin_info

    def intersection_volume_performance(self, model, dataset):
        try:
            X_values = dataset['X'].values
            if isinstance(X_values[0], (list, np.ndarray)):
                X_tensor = torch.tensor(np.stack(X_values), dtype=torch.float32)
            else:
                X_tensor = torch.tensor(X_values, dtype=torch.float32)
                if len(X_tensor.shape) == 1:
                    X_tensor = X_tensor.unsqueeze(1)

            y_true = dataset['intersection_volume'].values.astype(np.float32)

            model.eval()
            X_tensor = X_tensor.to(self.device)

            with torch.no_grad():
                # Always use raw model output, then scale consistently
                raw_output = model(X_tensor)
                
                if self.task_type == 'IntersectionStatus_IntersectionVolume':
                    if raw_output.ndim > 1 and raw_output.shape[1] > 1:
                        volume_predictions = raw_output[:, 1]  # Get volume predictions
                    else:
                        print("Warning: IntersectionStatus_IntersectionVolume task type selected, but model output seems 1D. Using as is.")
                        volume_predictions = raw_output
                else:
                    volume_predictions = raw_output.squeeze()
                
                # Apply volume scaling consistently
                volume_scale_factor = self.config.get('volume_scale_factor', 1.0)
                y_pred = (volume_predictions / volume_scale_factor).cpu().numpy().astype(np.float32)

            if y_true.shape != y_pred.shape:
                return {'error': f"Shape mismatch between y_true and y_pred: {y_true.shape} vs {y_pred.shape}"}
            if len(y_true) == 0:
                return {'error': "Dataset is empty."}

            # Calculate the overall metrics
            overall_mae = mean_absolute_error(y_true, y_pred)
            overall_r2 = r2_score(y_true, y_pred)
            overall_relative_mae = self._calculate_relative_mae(y_true, y_pred)
            overall_smape = self._calculate_smape(y_true, y_pred)
            
            # Enhanced log-scale kappa calculation
            overall_kappa, overall_bin_accuracy_10bins, conf_matrix, bin_info = self.calculate_volume_binning_kappa(
                y_true, y_pred, num_bins=10
            )


            volume_range = self.config.get('volume_range', (0, 0.01))
            if volume_range[0] < 0 or volume_range[1] > 0.16666667:
                raise ValueError("Volume range must be between 0 and 0.16666667.")
            intervals = np.linspace(volume_range[0], volume_range[1], self.config.get('evaluation_n_bins', 5))
            interval_metrics = {}
            total_correct_bin_predictions = 0
            total_samples_in_bins = 0

            for i in range(len(intervals) - 1):
                low = intervals[i]
                high = intervals[i+1]
                interval_key = f'{low:.8f}-{high:.8f}'

                true_bin_mask = (y_true >= low) & (y_true < high)
                true_bin_samples = int(np.sum(true_bin_mask))

                if true_bin_samples > 0:
                    pred_bin_mask = (y_pred >= low) & (y_pred < high)
                    correct_bin_predictions = int(np.sum(true_bin_mask & pred_bin_mask))
                    bin_accuracy = float(correct_bin_predictions / true_bin_samples)

                    # Calculate interval metrics - only MAE, MSE, bin predictions, and bin accuracy
                    y_true_interval = y_true[true_bin_mask]
                    y_pred_interval = y_pred[true_bin_mask]
                    
                    mae_interval = mean_absolute_error(y_true_interval, y_pred_interval)
                    mse_interval = np.mean((y_true_interval - y_pred_interval) ** 2)  # Calculate MSE

                    interval_metrics[interval_key] = {
                        'mae': float(mae_interval),
                        'mse': float(mse_interval),  # Added: MSE for interval
                        'samples': true_bin_samples,
                        'correct_bin_predictions': correct_bin_predictions,
                        'bin_accuracy': bin_accuracy
                    }
                    total_correct_bin_predictions += correct_bin_predictions
                    total_samples_in_bins += true_bin_samples
                else:
                    interval_metrics[interval_key] = {
                        'mae': None,
                        'mse': None,  # Added: MSE placeholder for empty intervals
                        'samples': 0,
                        'correct_bin_predictions': 0,
                        'bin_accuracy': None
                    }

            # Calculate bin accuracy using the original method for interval-based binning
            overall_bin_accuracy_intervals = float(total_correct_bin_predictions / total_samples_in_bins) if total_samples_in_bins > 0 else 0.0

            results = {
                'overall_IntersectionVolume_metrics': {
                    'mae': float(overall_mae),
                    'r2_score': float(overall_r2),
                    'relative_mae': float(overall_relative_mae),
                    'smape': float(overall_smape),
                    'kappa': float(overall_kappa),
                    'samples': len(y_true),
                    'overall_bin_accuracy': overall_bin_accuracy_10bins,  # Now log-scale
                    'overall_bin_accuracy_intervals': overall_bin_accuracy_intervals,  # Original interval accuracy
                    'samples_in_binned_range': total_samples_in_bins,
                    'confusion_matrix': conf_matrix.tolist() if conf_matrix is not None else None,
                    'log_scale_binning_info': bin_info  # NEW: Add binning details
                },
                'interval_metrics': interval_metrics
            }

        except Exception as e:
            import traceback
            print(f"Error during IntersectionVolume performance calculation: {e}")
            traceback.print_exc()
            results = {'error': str(e)}

        return results

    def inference_speed(self, model, dataset):
        try:
            X = torch.tensor(dataset['X'].values, dtype=torch.float32).to(self.device)
            num_samples = X.shape[0]
            
            print(f"    Speed test on {num_samples} samples...")
            
            predict_method = model.predict if hasattr(model, 'predict') else lambda x: model(x)

            with torch.no_grad():
                for _ in range(10):
                    _ = predict_method(X)

            repetitions = 50
            timings = np.zeros(repetitions)

            device_info = {
                'device_type': str(self.device),
            }

            if self.device.type == 'cuda':
                device_info.update({
                    'cuda_device_name': torch.cuda.get_device_name(self.device),
                    'cuda_driver_version': torch.version.cuda,
                    'cuda_capability': torch.cuda.get_device_capability(self.device),
                    'cuda_memory': f"{torch.cuda.get_device_properties(self.device).total_memory/1e9:.2f} GB"
                })

                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                for rep in range(repetitions):
                    starter.record()
                    _ = predict_method(X)
                    ender.record()
                    torch.cuda.synchronize()
                    timings[rep] = starter.elapsed_time(ender)
            else:
                device_info.update({
                    'cpu_model': platform.processor(),
                    'system_platform': platform.platform(),
                    'cpu_cores_physical': os.cpu_count()
                })

                for rep in range(repetitions):
                    start_time = time.perf_counter()
                    _ = predict_method(X)
                    end_time = time.perf_counter()
                    timings[rep] = (end_time - start_time) * 1000

            avg_time_ms = np.mean(timings)
            total_time_seconds = (dataset['transformation_time_ms'] / 1000.0) + (dataset['augmentation_time_ms'] / 1000.0) + (avg_time_ms / 1000.0)

            return {
                'device_info': device_info,
                'repetitions': repetitions,
                'augmentation_time_seconds': dataset['augmentation_time_ms'] / 1000.0,
                'transformation_time_seconds': dataset['transformation_time_ms'] / 1000.0,
                'inference_time_seconds': avg_time_ms / 1000.0,
                'total_time_seconds': total_time_seconds,
                'avg_time_per_sample_seconds': total_time_seconds / num_samples,
                'samples_per_second': num_samples / total_time_seconds,
                'total_samples': num_samples
            }
            
        except Exception as e:
            return {'error': str(e)}

    def tetrahedron_wise_permutation_consistency(self, model, dataset):
        try:
            X = torch.tensor(dataset['X'].values, dtype=torch.float32, device=self.device)
            
            X_swapped = gu.swap_tetrahedrons(X)
            
            with torch.no_grad():
                if hasattr(model, 'predict'):
                    pred_original = model.predict(X).cpu().numpy().astype(np.float32)
                    pred_swapped = model.predict(X_swapped).cpu().numpy().astype(np.float32)
                else:
                    pred_original = model(X).cpu().numpy().astype(np.float32)
                    pred_swapped = model(X_swapped).cpu().numpy().astype(np.float32)
            
            rtol = self.config.get('regression_consistency_thresholds', {}).get('rtol', 1e-1)
            atol = self.config.get('regression_consistency_thresholds', {}).get('atol', 1e-4)

            if self.task_type == 'IntersectionStatus':
                consistent = (pred_original == pred_swapped)
                consistency_rate = float(np.mean(consistent))

                result = {
                    "classification_consistency_rate": consistency_rate,
                    "total_samples": X.shape[0]
                }
            elif self.task_type == 'IntersectionVolume':
                consistent = np.isclose(pred_original, pred_swapped, rtol=rtol, atol=atol)
                consistency_rate = float(np.mean(consistent))
                mad = float(np.mean(np.abs(pred_original - pred_swapped)))
                result = {
                    "IntersectionVolume_consistency_rate": consistency_rate,
                    "consistency_thresholds": {"rtol": rtol, "atol": atol},
                    "mean_absolute_difference": mad,
                    "total_samples": X.shape[0]
                }
            elif self.task_type == 'IntersectionStatus_IntersectionVolume':
                cls_original = pred_original[:, 0]
                cls_swapped = pred_swapped[:, 0]
                reg_original = pred_original[:, 1]
                reg_swapped = pred_swapped[:, 1]
                
                cls_consistent = (cls_original == cls_swapped)
                reg_consistent = np.isclose(reg_original, reg_swapped, rtol=rtol, atol=atol)
                
                consistency_rate_cls = float(np.mean(cls_consistent))
                consistency_rate_reg = float(np.mean(reg_consistent))
                mad_reg = float(np.mean(np.abs(reg_original - reg_swapped)))
                
                result = {
                    "classification_consistency_rate": consistency_rate_cls,
                    "IntersectionVolume_consistency_rate": consistency_rate_reg,
                    "mean_absolute_difference": mad_reg,
                    "consistency_thresholds": {"rtol": rtol, "atol": atol},
                    "total_samples": X.shape[0]
                }
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")
            
            return result

        except Exception as e:
            return {"error": str(e)}
            
    def point_wise_permutation_consistency(self, model, dataset):
        try:
            if not hasattr(self, 'task_type') or self.task_type not in ['IntersectionStatus', 'IntersectionVolume', 'IntersectionStatus_IntersectionVolume']:
                raise ValueError(f"Invalid or missing task type: {getattr(self, 'task_type', None)}")

            if not isinstance(dataset, dict) or 'X' not in dataset:
                raise ValueError("Input dataset must be a dictionary with key 'X'.")
            X = dataset['X']
            if not hasattr(X, 'values'):
                raise ValueError("dataset['X'] must have a 'values' attribute (e.g., a pandas DataFrame).")
            X = torch.tensor(X.values, dtype=torch.float32, device=self.device)
            
            X_permuted = gu.permute_points_within_tetrahedrons(X)

            with torch.no_grad():
                if hasattr(model, 'predict'):
                    pred_original = model.predict(X).cpu().numpy().astype(np.float32)
                    pred_permuted = model.predict(X_permuted).cpu().numpy().astype(np.float32)
                else:
                    pred_original = model(X).cpu().numpy().astype(np.float32)
                    pred_permuted = model(X_permuted).cpu().numpy().astype(np.float32)

            rtol = self.config.get('regression_consistency_thresholds', {}).get('rtol', 1e-1)
            atol = self.config.get('regression_consistency_thresholds', {}).get('atol', 1e-4)
            result = {}
            
            if self.task_type == 'IntersectionStatus':
                consistent = (pred_original == pred_permuted)
                result["classification_consistency_rate"] = float(np.mean(consistent))

            elif self.task_type == 'IntersectionVolume':
                consistent = np.isclose(pred_original, pred_permuted, rtol=rtol, atol=atol)
                result["IntersectionVolume_consistency_rate"] = float(np.mean(consistent))
                result["consistency_thresholds"] = {"rtol": rtol, "atol": atol}
                result["mean_absolute_difference"] = float(np.mean(np.abs(pred_original - pred_permuted)))

            elif self.task_type == 'IntersectionStatus_IntersectionVolume':
                cls_original = pred_original[:, 0]
                cls_swapped = pred_permuted[:, 0]
                reg_original = pred_original[:, 1]
                reg_swapped = pred_permuted[:, 1]

                cls_consistent = (cls_original == cls_swapped)
                reg_consistent = np.isclose(reg_original, reg_swapped, rtol=rtol, atol=atol)

                result["classification_consistency_rate"] = float(np.mean(cls_consistent))
                result["IntersectionVolume_consistency_rate"] = float(np.mean(reg_consistent))
                result["mean_absolute_difference"] = float(np.mean(np.abs(pred_original - pred_permuted)))
                result["consistency_thresholds"] = {"rtol": rtol, "atol": atol}

            result["total_samples"] = X.shape[0]

            return result

        except ValueError as ve:
            return {"error": "Data format or validation error", "details": str(ve)}
        except RuntimeError as re:
            return {"error": "CUDA/Device or runtime error", "details": str(re)}
        except AttributeError as ae:
            return {"error": "Missing attribute or function", "details": str(ae)}
        except Exception as e:
            return {"error": "Unexpected error", "details": str(e)}

    def _calculate_overall_classification_metrics(self, report):
        datasets = report['dataset_reports']
        
        # Existing metrics
        all_metrics = {'total_samples': 0, 'weighted_correct': 0}
        
        # New metrics for your requirements
        dataset_accuracies = []  # For mean accuracy calculation
        all_y_true = []  # For overall AUC calculation
        all_y_pred = []  # For overall AUC calculation
        
        priority_group = ['polyhedron_intersection', 'no_intersection']
        secondary_group = ['point_intersection', 'segment_intersection', 'polygon_intersection']
        
        group_metrics = {
            'priority': {'total_samples': 0, 'correct_samples': 0},
            'secondary': {'total_samples': 0, 'correct_samples': 0}
        }
        
        for dataset_name, dataset_report in datasets.items():
            if 'classification_performance' not in dataset_report:
                continue
                
            dataset_type = dataset_name.split('/')[0]
            
            perf = dataset_report['classification_performance']
            if isinstance(perf, dict) and 'accuracy' in perf:
                dataset_index = next((i for i, d in enumerate(self.datasets) if d['name'] == dataset_name), None)
                if dataset_index is None:
                    continue
                    
                dataset = self.datasets[dataset_index]
                n_samples = len(dataset['intersection_status'])
                
                # Existing calculations
                all_metrics['total_samples'] += n_samples
                all_metrics['weighted_correct'] += perf['accuracy'] * n_samples
                
                # New: Collect individual dataset accuracies for mean calculation
                dataset_accuracies.append(perf['accuracy'])
                
                # New: Collect actual labels for overall AUC calculation
                y_true_dataset = dataset['intersection_status'].values
                all_y_true.extend(y_true_dataset)
                
                # Need to get predictions for AUC - we'll need to re-run prediction
                # or store predictions during classification_performance
                
                if dataset_type in priority_group:
                    group_metrics['priority']['total_samples'] += n_samples
                    group_metrics['priority']['correct_samples'] += perf['accuracy'] * n_samples
                elif dataset_type in secondary_group:
                    group_metrics['secondary']['total_samples'] += n_samples
                    group_metrics['secondary']['correct_samples'] += perf['accuracy'] * n_samples
        
        # Calculate overall accuracy (total correct / total samples)
        if all_metrics['total_samples'] > 0:
            overall_accuracy = all_metrics['weighted_correct'] / all_metrics['total_samples']
        else:
            overall_accuracy = None
        
        # Calculate mean accuracy across datasets
        mean_accuracy = np.mean(dataset_accuracies) if dataset_accuracies else None
        
        # Group accuracies
        for group in ['priority', 'secondary']:
            if group_metrics[group]['total_samples'] > 0:
                group_metrics[group]['accuracy'] = (
                    group_metrics[group]['correct_samples'] / 
                    group_metrics[group]['total_samples']
                )
            else:
                group_metrics[group]['accuracy'] = None
        
        # Weighted metrics
        weighted_metrics = {}
        
        if (group_metrics['priority']['accuracy'] is not None and 
            group_metrics['secondary']['accuracy'] is not None):
            weighted_metrics['80_20_accuracy'] = (
                0.8 * group_metrics['priority']['accuracy'] + 
                0.2 * group_metrics['secondary']['accuracy']
            )
        else:
            weighted_metrics['80_20_accuracy'] = None
        
        if (group_metrics['priority']['accuracy'] is not None and 
            group_metrics['secondary']['accuracy'] is not None):
            weighted_metrics['20_80_accuracy'] = (
                0.2 * group_metrics['priority']['accuracy'] + 
                0.8 * group_metrics['secondary']['accuracy']
            )
        else:
            weighted_metrics['20_80_accuracy'] = None
        
        return {
            'overall_accuracy': overall_accuracy,  # Total correct / total samples
            'mean_accuracy': mean_accuracy,  # Mean of individual dataset accuracies
            'total_samples': all_metrics['total_samples'],
            'num_datasets': len(dataset_accuracies),
            'priority_group_accuracy': group_metrics['priority']['accuracy'],
            'priority_group_samples': group_metrics['priority']['total_samples'],
            'secondary_group_accuracy': group_metrics['secondary']['accuracy'],
            'secondary_group_samples': group_metrics['secondary']['total_samples'],
            'weighted_80_20_accuracy': weighted_metrics['80_20_accuracy'],
            'weighted_20_80_accuracy': weighted_metrics['20_80_accuracy'],
            # Note: overall_auc will need to be calculated separately 
            # because we need the actual predictions, not just accuracy
        }

    def _calculate_overall_auc(self, model):
        """Calculate AUC on all datasets combined"""
        all_y_true = []
        all_y_pred = []
        
        with torch.no_grad():
            for dataset in self.datasets:
                X = torch.tensor(dataset['X'].values, dtype=torch.float32).to(self.device)
                y_true = dataset['intersection_status'].values
                
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X).cpu().numpy()
                else:
                    y_pred = model(X).cpu().numpy()
                
                if self.task_type == 'IntersectionStatus_IntersectionVolume':
                    y_pred = y_pred[:, 0]
                
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
        
        try:
            overall_auc = roc_auc_score(all_y_true, all_y_pred)
            return overall_auc, len(all_y_true)
        except:
            return 0.5, len(all_y_true)

    def _calculate_overall_regression_metrics(self, report):
        datasets = report['dataset_reports']
        
        total_samples = 0
        weighted_mae_sum = 0
        weighted_r2_sum = 0  # Added: R² score tracking
        weighted_relative_mae_sum = 0  # Added: Relative MAE tracking
        weighted_smape_sum = 0  # Added: SMAPE tracking
        weighted_kappa_sum = 0
        weighted_bin_accuracy_sum = 0
        total_samples_in_bins = 0

        for dataset_name, dataset_report in datasets.items():
            if 'intersection_volume_performance' not in dataset_report or 'error' in dataset_report['intersection_volume_performance']:
                continue

            perf = dataset_report['intersection_volume_performance']
            overall_metrics = perf.get('overall_IntersectionVolume_metrics', {})

            mae = overall_metrics.get('mae')
            r2 = overall_metrics.get('r2_score')  # Added: Extract R² score
            relative_mae = overall_metrics.get('relative_mae')  # Added: Extract relative MAE
            smape = overall_metrics.get('smape')  # Added: Extract SMAPE
            kappa = overall_metrics.get('kappa')
            samples = overall_metrics.get('samples')
            bin_accuracy = overall_metrics.get('overall_bin_accuracy')
            samples_in_binned_range = overall_metrics.get('samples_in_binned_range')

            if samples is not None and samples > 0:
                total_samples += samples
                if mae is not None:
                    weighted_mae_sum += mae * samples
                if r2 is not None:  # Added: Weight R² score by sample count
                    weighted_r2_sum += r2 * samples
                if relative_mae is not None:  # Added: Weight relative MAE by sample count
                    weighted_relative_mae_sum += relative_mae * samples
                if smape is not None:  # Added: Weight SMAPE by sample count
                    weighted_smape_sum += smape * samples
                if kappa is not None:
                    weighted_kappa_sum += kappa * samples
                
                if bin_accuracy is not None and samples_in_binned_range is not None and samples_in_binned_range > 0:
                    weighted_bin_accuracy_sum += bin_accuracy * samples_in_binned_range
                    total_samples_in_bins += samples_in_binned_range

        # Calculate weighted averages
        overall_mae = weighted_mae_sum / total_samples if total_samples > 0 else None
        overall_r2 = weighted_r2_sum / total_samples if total_samples > 0 else None  # Added: Calculate overall R²
        overall_relative_mae = weighted_relative_mae_sum / total_samples if total_samples > 0 else None  # Added: Calculate overall relative MAE
        overall_smape = weighted_smape_sum / total_samples if total_samples > 0 else None  # Added: Calculate overall SMAPE
        overall_kappa = weighted_kappa_sum / total_samples if total_samples > 0 else None
        overall_bin_accuracy = weighted_bin_accuracy_sum / total_samples_in_bins if total_samples_in_bins > 0 else None

        return {
            'overall_mae': overall_mae,
            'overall_r2_score': overall_r2,  # Added: R² score in return dict
            'overall_relative_mae': overall_relative_mae,  # Added: Relative MAE in return dict  
            'overall_smape': overall_smape,  # Added: SMAPE in return dict
            'overall_kappa': overall_kappa,
            'overall_bin_accuracy': overall_bin_accuracy,
            'total_samples': total_samples,
            'total_samples_in_binned_range': total_samples_in_bins
        }

    def _calculate_inference_speed_summary(self, report):
        all_speeds = []
        total_samples = 0
        device_info = None
        
        for dataset_name, dataset_report in report['dataset_reports'].items():
            if 'inference_speed' in dataset_report:
                speed = dataset_report['inference_speed']
                if 'error' not in speed:
                    all_speeds.append(speed['samples_per_second'])
                    total_samples += speed.get('total_samples', 0)
                    if device_info is None:
                        device_info = speed.get('device_info', {}).get('device_type', str(self.device))
        
        return {
            'device': device_info,
            'total_samples': total_samples,
            'avg_throughput_samples_per_second': np.mean(all_speeds) if all_speeds else 0,
            'avg_latency_per_sample_ms': (1000 / np.mean(all_speeds)) if all_speeds else float('inf')
        }

    def _calculate_smape(self, y_true, y_pred, epsilon=1e-8):
        """Calculate Symmetric Mean Absolute Percentage Error"""
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon))

    def _calculate_relative_mae(self, y_true, y_pred):
        """Calculate Relative Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred)) / np.mean(y_true) if np.mean(y_true) != 0 else float('inf')

    def _get_system_info(self):
        system_info = {
            'operating_system': platform.system(),
            'os_version': platform.version(),
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'total_memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'available_memory_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
        }
        
        if torch.cuda.is_available():
            try:
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
                
                system_info.update({
                    'gpu_available': True,
                    'gpu_count': gpu_count,
                    'current_gpu': current_device,
                    'gpu_name': gpu_name,
                    'gpu_memory_gb': round(gpu_memory, 2),
                    'cuda_version': torch.version.cuda
                })
                
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        system_info['nvidia_driver_version'] = result.stdout.strip()
                except:
                    pass
                    
            except Exception as e:
                system_info.update({
                    'gpu_available': False,
                    'gpu_error': str(e)
                })
        else:
            system_info['gpu_available'] = False
        
        return system_info

    def _get_model_info(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024**2)
        
        return {
            'model_class': type(model).__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': round(model_size_mb, 2),
            'parameter_dtype': str(next(model.parameters()).dtype),
            'device': str(next(model.parameters()).device)
        }

    def _discretize_values(self, values, n_bins=10):
        try:
            return pd.cut(values, bins=n_bins, labels=False, duplicates='drop')
        except:
            return np.zeros_like(values, dtype=int)
    
    def _print_final_report(self, results):
        print("\n" + "=" * 80)
        print("FINAL EVALUATION REPORT")
        print("=" * 80)
        
        print("\n--- SYSTEM INFORMATION ---")
        sys_info = results['system_info']
        print(f"Operating System: {sys_info['operating_system']} {sys_info['os_version']}")
        print(f"Platform: {sys_info['platform']}")
        print(f"Processor: {sys_info['processor']}")
        print(f"CPU Cores: {sys_info['cpu_count']} logical, {sys_info['cpu_count_physical']} physical")
        print(f"Total Memory: {sys_info['total_memory_gb']} GB")
        print(f"Python Version: {sys_info['python_version']}")
        print(f"PyTorch Version: {sys_info['pytorch_version']}")
        
        if sys_info.get('gpu_available', False):
            print(f"GPU: {sys_info['gpu_name']}")
            print(f"GPU Memory: {sys_info['gpu_memory_gb']} GB")
            print(f"CUDA Version: {sys_info['cuda_version']}")
        else:
            print("GPU: Not available")
        
        print(f"\n--- MODEL INFORMATION ---")
        model_info = results['model_info']
        print(f"Model Class: {model_info['model_class']}")
        print(f"Total Parameters: {model_info['total_parameters']:,}")
        print(f"Trainable Parameters: {model_info['trainable_parameters']:,}")
        print(f"Model Size: {model_info['model_size_mb']} MB")
        print(f"Parameter Type: {model_info['parameter_dtype']}")
        
        print(f"\n--- DATASET RESULTS ---")
        for dataset_name, dataset_results in results['dataset_reports'].items():
            print(f"\n{dataset_name}:")
            
            if 'classification_performance' in dataset_results:
                perf = dataset_results['classification_performance']
                if 'error' not in perf:
                    print(f"  Accuracy: {perf['accuracy']:.4f}")
                    print(f"  AUC: {perf['auc']:.4f}")
                    print(f"  Kappa: {perf['kappa']:.4f}")
                else:
                    print(f"  Classification Error: {perf['error']}")
            
            if 'intersection_volume_performance' in dataset_results:
                perf = dataset_results['intersection_volume_performance']
                if 'error' not in perf:
                    overall = perf['overall_IntersectionVolume_metrics']
                    print(f"  MAE: {overall['mae']:.6e}")
                    print(f"  R²: {overall.get('r2_score', 'N/A'):.4f}")
                    print(f"  Relative MAE: {overall.get('relative_mae', 'N/A'):.4f}")
                    print(f"  SMAPE: {overall.get('smape', 'N/A'):.2f}%")
                    print(f"  Kappa (log-scale): {overall['kappa']:.4f}")  # Updated label
                    print(f"  Bin Accuracy (log-scale): {overall['overall_bin_accuracy']:.4f}")  # Updated label
                    
                    # NEW: Show binning information
                    if 'log_scale_binning_info' in overall:
                        bin_info = overall['log_scale_binning_info']
                        binning_method = bin_info.get('binning_method', 'unknown')
                        volume_orders = bin_info.get('volume_range_orders_of_magnitude', None)
                        
                        if volume_orders is not None:
                            print(f"  Volume Range: {volume_orders:.1f} orders of magnitude ({binning_method})")
                        else:
                            print(f"  Binning Method: {binning_method}")
                    
                    if 'interval_metrics' in perf:
                        print(f"  Volume Bin Analysis:")
                        for interval_key, metrics in perf['interval_metrics'].items():
                            if metrics['samples'] > 0:
                                # Fixed: Use new metrics instead of MAPE
                                mae_val = metrics.get('mae', 'N/A')
                                r2_val = metrics.get('r2_score', 'N/A')
                                rel_mae_val = metrics.get('relative_mae', 'N/A')
                                smape_val = metrics.get('smape', 'N/A')
                                samples = metrics['samples']
                                
                                mae_str = f"{mae_val:.2e}" if mae_val != 'N/A' else 'N/A'
                                r2_str = f"{r2_val:.3f}" if r2_val != 'N/A' else 'N/A'
                                rel_mae_str = f"{rel_mae_val:.3f}" if rel_mae_val != 'N/A' else 'N/A'
                                smape_str = f"{smape_val:.1f}%" if smape_val != 'N/A' else 'N/A'
                                
                                print(f"    {interval_key}: MAE={mae_str}, R²={r2_str}, RelMAE={rel_mae_str}, SMAPE={smape_str}, n={samples}")
                else:
                    print(f"  Regression Error: {perf['error']}")
            
            if 'point_wise_permutation_consistency' in dataset_results:
                consistency = dataset_results['point_wise_permutation_consistency']
                if 'error' not in consistency:
                    if 'classification_consistency_rate' in consistency:
                        print(f"  Point-wise Classification Consistency: {consistency['classification_consistency_rate']:.4f}")
                    if 'IntersectionVolume_consistency_rate' in consistency:
                        print(f"  Point-wise Volume Consistency: {consistency['IntersectionVolume_consistency_rate']:.4f}")
            
            if 'tetrahedron_wise_permutation_consistency' in dataset_results:
                consistency = dataset_results['tetrahedron_wise_permutation_consistency']
                if 'error' not in consistency:
                    if 'classification_consistency_rate' in consistency:
                        print(f"  Tetrahedron-wise Classification Consistency: {consistency['classification_consistency_rate']:.4f}")
                    if 'IntersectionVolume_consistency_rate' in consistency:
                        print(f"  Tetrahedron-wise Volume Consistency: {consistency['IntersectionVolume_consistency_rate']:.4f}")
        
        print(f"\n--- OVERALL METRICS ---")
        
        if 'overall_classification_metrics' in results:
            cls_metrics = results['overall_classification_metrics']
            print(f"Overall Accuracy: {cls_metrics['overall_accuracy']:.4f}")
            print(f"Weighted 80/20 Accuracy: {cls_metrics['weighted_80_20_accuracy']:.4f}")
            print(f"Weighted 20/80 Accuracy: {cls_metrics['weighted_20_80_accuracy']:.4f}")
        
        if 'overall_regression_metrics' in results:
            reg_metrics = results['overall_regression_metrics']
            print(f"Overall MAE: {reg_metrics['overall_mae']:.6e}")
            print(f"Overall R²: {reg_metrics.get('overall_r2_score', 'N/A'):.4f}")  # Fixed: Use new metrics
            print(f"Overall Relative MAE: {reg_metrics.get('overall_relative_mae', 'N/A'):.4f}")  # Fixed: Use new metrics  
            print(f"Overall SMAPE: {reg_metrics.get('overall_smape', 'N/A'):.2f}%")  # Fixed: Use new metrics
            print(f"Overall Kappa: {reg_metrics['overall_kappa']:.4f}")
            print(f"Overall Bin Accuracy: {reg_metrics['overall_bin_accuracy']:.4f}")
        
        print(f"\n--- INFERENCE SPEED ---")
        speed = results['inference_speed_summary']
        print(f"Device: {speed['device']}")
        print(f"Total Samples: {speed['total_samples']:,}")
        print(f"Throughput: {speed['avg_throughput_samples_per_second']:.3f} samples/second")
        print(f"Latency: {speed['avg_latency_per_sample_ms']:.6f} ms/sample")
        
        print("\n" + "=" * 80)