import pandas as pd
import numpy as np
import torch
import time
import platform
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import GeometryUtils as gu
import src.CDataProcessor as dp

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix

class CEvaluator:
    def __init__(self, config):
        self.config = config
        self.task_type = config.get('task', 'IntersectionStatus')
        self.test_data_loaded = False
        self.datasets = []
        self.intersection_status_column_name = 'HasIntersection'
        self.intersection_volume_column_name = 'IntersectionVolume'
        self.dp = dp.CDataProcessor(config)
        
        self.test_registry = {
            'IntersectionStatus': [self.classification_performance, self.point_wise_permutation_consistency, self.tetrahedron_wise_permutation_consistency, self.inference_speed],
            'IntersectionVolume': [self.IntersectionVolume_performance, self.point_wise_permutation_consistency, self.tetrahedron_wise_permutation_consistency, self.inference_speed],
            'IntersectionStatus_IntersectionVolume': [self.classification_performance, self.point_wise_permutation_consistency, self.tetrahedron_wise_permutation_consistency, self.IntersectionVolume_performance, self.inference_speed]
        }

        self.dataset_folders = (
            ['polyhedron_intersection', 'big_dataset'] 
            if self.task_type == 'IntersectionVolume'
            else [
                'no_intersection', 'point_intersection', 
                'segment_intersection', 'polygon_intersection', 
                'polyhedron_intersection', 'big_dataset'
            ]
        )

    def evaluate(self, model):
        
        print("-- Evaluating --")

        model.eval()

        self._load_test_data()
        
        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        report = {
            'task_type': self.task_type,
            'model_parameters': {
                'total': total_params,
                'trainable': trainable_params,
                'non_trainable': total_params - trainable_params
            },
            'dataset_reports': {}
        }

        # Run tests for the specific task type
        for dataset in self.datasets:
            dataset_report = {}
            for test in self.test_registry.get(self.task_type, []):
                test_name = test.__name__
                try:
                    dataset_report[test_name] = test(model, dataset)
                except Exception as e:
                    dataset_report[test_name] = {'error': str(e)}
            
            report['dataset_reports'][dataset['name']] = dataset_report

        # Calculate aggregate metrics based on task type
        if self.task_type in ['IntersectionStatus', 'IntersectionStatus_IntersectionVolume']:
            report['aggregate_IntersectionStatus_metrics'] = self._calculate_aggregate_IntersectionStatus_metrics(report)
        
        if self.task_type in ['IntersectionVolume', 'IntersectionStatus_IntersectionVolume']:
             report['aggregate_IntersectionVolume_metrics'] = self._calculate_aggregate_IntersectionVolume_metrics(report)

        print("---- Finished Evaluation ----")

        return report

    def _load_test_data(self):
        """Load all test datasets from configured directory structure"""

        if self.test_data_loaded:
            return

        base_path = self.config['test_data_path']

        assert os.path.exists(base_path), f"Directory not found: {base_path}"

        for folder in self.dataset_folders:
            dataset_dir = os.path.join(base_path, folder)

            assert os.path.exists(dataset_dir), f"Directory not found: {dataset_dir}"

            for file in os.listdir(dataset_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(dataset_dir, file)
                    df = pd.read_csv(file_path)
                    transformation_time_ms = 0
                    augmentation_time_ms = 0

                    if self.config.get('augmentations', None) is not None:
                        start = time.time()
                        df = self.dp.augment_data(df, self.config)
                        augmentation_time_ms = (time.time() - start) * 1000

                    if self.config.get('transformations', None) is not None:
                        start = time.time()
                        df = self.dp.transform_data(df, self.config)
                        transformation_time_ms = (time.time() - start) * 1000

                    
                    self.datasets.append({
                        'name': f"{folder}/{file}",
                        'type': folder,
                        'intersection_status': df[self.intersection_status_column_name],
                        'intersection_volume': df[self.intersection_volume_column_name],
                        'X': df.drop(columns=[self.intersection_status_column_name, self.intersection_volume_column_name]),
                        'transformation_time_ms': transformation_time_ms,
                        'augmentation_time_ms': augmentation_time_ms
                    })

        self.test_data_loaded = True

    def classification_performance(self, model, dataset):
        X = torch.tensor(dataset['X'].values, dtype=torch.float32)
        y_true = dataset['intersection_status'].values

        device = next(model.parameters()).device
        X = X.to(device)

        num_repeats = 50
        metrics_list = []

        try:
            with torch.no_grad():
                for _ in range(num_repeats):
                    y_pred = model.predict(X).cpu().numpy()
                    if self.task_type == 'IntersectionStatus_IntersectionVolume':
                        y_pred = y_pred[:, 0]

                    # Compute metrics for the current run
                    run_metrics = {
                        'accuracy': accuracy_score(y_true, y_pred),
                        'precision': precision_score(y_true, y_pred),
                        'recall': recall_score(y_true, y_pred),
                        'f1': f1_score(y_true, y_pred),
                        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
                    }
                    metrics_list.append(run_metrics)
        except Exception as e:
            return {'error': str(e)}

        # Average metrics across all runs
        avg_metrics = {
            'repetitions': num_repeats,
            'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
            'precision': np.mean([m['precision'] for m in metrics_list]),
            'recall': np.mean([m['recall'] for m in metrics_list]),
            'f1': np.mean([m['f1'] for m in metrics_list]),
            'confusion_matrix': np.mean([m['confusion_matrix'] for m in metrics_list], axis=0).tolist()
        }

        return avg_metrics

    def IntersectionVolume_performance(self, model, dataset):
        """
        Evaluates IntersectionVolume model performance overall and within specific intervals.

        Args:
            model: The trained PyTorch model.
            dataset: A dictionary or DataFrame containing 'X' and 'intersection_volume'.

        Returns:
            A dictionary containing overall and interval-specific performance metrics.
        """
 
        try:
            X_values = dataset['X'].values
            # Handle cases where X might already be stacked or needs stacking
            if isinstance(X_values[0], (list, np.ndarray)):
                 X_tensor = torch.tensor(np.stack(X_values), dtype=torch.float32)
            else:
                 X_tensor = torch.tensor(X_values, dtype=torch.float32)
                 if len(X_tensor.shape) == 1: # Ensure X is 2D (N_samples, N_features)
                    X_tensor = X_tensor.unsqueeze(1)

            y_true = dataset['intersection_volume'].values.astype(np.float32)

            model.eval()
            device = next(model.parameters()).device
            X_tensor = X_tensor.to(device)

            with torch.no_grad():
                # Perform prediction once
                y_pred_tensor = model.predict(X_tensor) # Assuming model.predict exists and returns tensor
                y_pred = y_pred_tensor.cpu().numpy().astype(np.float32)

            # Handle potential multi-output if task includes classification
            if self.task_type == 'IntersectionStatus_IntersectionVolume':
                 if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                     y_pred = y_pred[:, 1] # Assuming IntersectionVolume target is the second column
                 else:
                     # Handle cases where output might be unexpectedly 1D
                     print("Warning: IntersectionStatus_IntersectionVolume task type selected, but model output seems 1D. Using as is.")


            if y_true.shape != y_pred.shape:
                 return {'error': f"Shape mismatch between y_true and y_pred: {y_true.shape} vs {y_pred.shape}"}
            if len(y_true) == 0:
                 return {'error': "Dataset is empty."}


            # --- Calculate Overall Metrics ---
            overall_mae = mean_absolute_error(y_true, y_pred)
            overall_mse = mean_squared_error(y_true, y_pred)

            # --- Calculate Interval Metrics ---
            volume_range = self.config.get('volume_range', (0, 0.01))
            if volume_range[0] < 0 or volume_range[1] > 0.3333:
                raise ValueError("Volume range must be between 0 and 0.3333.")
            intervals = np.linspace(volume_range[0], volume_range[1], self.config.get('evaluation_n_bins', 5))
            interval_metrics = {}
            total_correct_bin_predictions = 0
            total_samples_in_bins = 0

            for i in range(len(intervals) - 1):
                low = intervals[i]
                high = intervals[i+1]
                interval_key = f'{low:.3f}-{high:.3f}'

                # Find samples where the TRUE value falls into this bin
                true_bin_mask = (y_true >= low) & (y_true < high)
                true_bin_samples = int(np.sum(true_bin_mask))

                if true_bin_samples > 0:
                    # Find samples where the PREDICTED value also falls into this bin
                    pred_bin_mask = (y_pred >= low) & (y_pred < high)

                    # Count samples correctly placed in this bin (true AND predicted are in the bin)
                    correct_bin_predictions = int(np.sum(true_bin_mask & pred_bin_mask))

                    # Calculate bin accuracy: % of samples truly in this bin that were also predicted in this bin
                    bin_accuracy = float(correct_bin_predictions / true_bin_samples)

                    # Calculate MAE/MSE for samples truly belonging to this bin
                    mae_interval = mean_absolute_error(y_true[true_bin_mask], y_pred[true_bin_mask])
                    mse_interval = mean_squared_error(y_true[true_bin_mask], y_pred[true_bin_mask])

                    interval_metrics[interval_key] = {
                        'mae': float(mae_interval),
                        'mse': float(mse_interval),
                        'samples': true_bin_samples,
                        'correct_bin_predictions': correct_bin_predictions,
                        'bin_accuracy': bin_accuracy
                    }
                    total_correct_bin_predictions += correct_bin_predictions
                    total_samples_in_bins += true_bin_samples
                else:
                    # No true samples in this bin
                    interval_metrics[interval_key] = {
                        'mae': None,
                        'mse': None,
                        'samples': 0,
                        'correct_bin_predictions': 0,
                        'bin_accuracy': None
                    }

            # Calculate overall bin accuracy across defined intervals
            overall_bin_accuracy = float(total_correct_bin_predictions / total_samples_in_bins) if total_samples_in_bins > 0 else 0.0

            # --- Assemble Final Results ---
            results = {
                'overall_IntersectionVolume_metrics': {
                    'mae': float(overall_mae),
                    'mse': float(overall_mse),
                    'samples': len(y_true),
                    'overall_bin_accuracy': overall_bin_accuracy,
                    'samples_in_binned_range': total_samples_in_bins
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

        X = torch.tensor(dataset['X'].values, dtype=torch.float32)
        device = next(model.parameters()).device
        X = X.to(device)
        num_samples = X.shape[0]

        predict_method = model.predict if hasattr(model, 'predict') else lambda x: model(x)

        # Warmup runs to avoid initialization overhead
        with torch.no_grad():
            for _ in range(10):
                _ = predict_method(X)

        repetitions = 50
        timings = np.zeros(repetitions)

        device_info = {
            'device_type': str(device),
        }

        if device.type == 'cuda':

            device_info.update({
                'cuda_device_name': torch.cuda.get_device_name(device),
                'cuda_driver_version': torch.version.cuda,
                'cuda_capability': torch.cuda.get_device_capability(device),
                'cuda_memory': f"{torch.cuda.get_device_properties(device).total_memory/1e9:.2f} GB"
            })

            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for rep in range(repetitions):
                starter.record()
                _ = predict_method(X)
                ender.record()
                torch.cuda.synchronize()
                timings[rep] = starter.elapsed_time(ender)  # Milliseconds
        else:

            device_info.update({
                'cpu_model': platform.processor(),
                'system_platform': platform.platform(),
                'cpu_cores_physical': os.cpu_count()  # logical cores
            })

            for rep in range(repetitions):
                start_time = time.perf_counter()
                _ = predict_method(X)
                end_time = time.perf_counter()
                timings[rep] = (end_time - start_time) * 1000  # Convert to milliseconds

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
            'samples_per_second': num_samples / total_time_seconds
        }

    def tetrahedron_wise_permutation_consistency(self, model, dataset):
        """
        Test prediction consistency when swapping tetrahedron order in pairs.
        Returns a dictionary of consistency metrics.
        """
        try:
            device = next(model.parameters()).device
            X = torch.tensor(dataset['X'].values, dtype=torch.float32, device=device)
            
            X_swapped = gu.swap_tetrahedrons(X)
            
            # Get predictions without gradient tracking
            with torch.no_grad():
                pred_original = model.predict(X).cpu().numpy().astype(np.float32)
                pred_swapped = model.predict(X_swapped).cpu().numpy().astype(np.float32)
            
            rtol = 1e-1
            atol = 1e-2

            # Process predictions based on task type
            if self.task_type == 'IntersectionStatus':
                consistent = (pred_original == pred_swapped)
                consistency_rate = float(np.mean(consistent))

                result = {
                    "classification_consistency_rate": consistency_rate,
                    "total_samples": X.shape[0]
                }
            elif self.task_type == 'IntersectionVolume':
                consistent = np.isclose(pred_original, pred_swapped, rtol=rtol, atol = atol)
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
        """
        Test prediction consistency when randomly permuting points within each of
        the 2 tetrahedrons (24 total features) while preserving coordinate groupings.
        Returns a dictionary with consistency metrics.
        """
        try:
            # Validate task type
            if not hasattr(self, 'task_type') or self.task_type not in ['IntersectionStatus', 'IntersectionVolume', 'IntersectionStatus_IntersectionVolume']:
                raise ValueError(f"Invalid or missing task type: {getattr(self, 'task_type', None)}")

            # Determine device
            if hasattr(model, 'device'):
                device = model.device
            elif next(model.parameters(), None) is not None:
                device = next(model.parameters()).device
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Validate and prepare input data
            if not isinstance(dataset, dict) or 'X' not in dataset:
                raise ValueError("Input dataset must be a dictionary with key 'X'.")
            X = dataset['X']
            if not hasattr(X, 'values'):
                raise ValueError("dataset['X'] must have a 'values' attribute (e.g., a pandas DataFrame).")
            X = torch.tensor(X.values, dtype=torch.float32, device=device)
            
            X_permuted = gu.permute_points_within_tetrahedrons(X)

            # Get predictions
            pred_original = model.predict(X).cpu().numpy().astype(np.float32)
            pred_permuted = model.predict(X_permuted).cpu().numpy().astype(np.float32)

            # Calculate consistency metrics
            rtol = 1e-1
            atol = 1e-2
            result = {}

            mad = float(np.mean(np.abs(pred_original - pred_permuted)))
            
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

    def _calculate_aggregate_IntersectionStatus_metrics(self, report):
        """Calculate aggregate classification metrics across all datasets"""
        datasets = report['dataset_reports']
        
        # Prepare containers for metrics
        all_metrics = {'total_samples': 0, 'weighted_correct': 0}
        
        # Dataset groups for weighted metrics
        priority_group = ['polyhedron_intersection', 'no_intersection']
        secondary_group = ['point_intersection', 'segment_intersection', 'polygon_intersection']
        
        group_metrics = {
            'priority': {'total_samples': 0, 'correct_samples': 0},
            'secondary': {'total_samples': 0, 'correct_samples': 0}
        }
        
        # Collect metrics from all classification datasets
        for dataset_name, dataset_report in datasets.items():
            if 'classification_performance' not in dataset_report:
                continue
                
            # Get dataset type from dataset folder (e.g., 'no_intersection/file.csv' -> 'no_intersection')
            dataset_type = dataset_name.split('/')[0]
            
            perf = dataset_report['classification_performance']
            if isinstance(perf, dict) and 'accuracy' in perf:
                # Get total sample count
                dataset_index = next((i for i, d in enumerate(self.datasets) if d['name'] == dataset_name), None)
                if dataset_index is None:
                    continue
                    
                dataset = self.datasets[dataset_index]
                n_samples = len(dataset['intersection_status'])
                
                # Add to overall metrics
                all_metrics['total_samples'] += n_samples
                all_metrics['weighted_correct'] += perf['accuracy'] * n_samples
                
                # Add to group metrics
                if dataset_type in priority_group:
                    group_metrics['priority']['total_samples'] += n_samples
                    group_metrics['priority']['correct_samples'] += perf['accuracy'] * n_samples
                elif dataset_type in secondary_group:
                    group_metrics['secondary']['total_samples'] += n_samples
                    group_metrics['secondary']['correct_samples'] += perf['accuracy'] * n_samples
        
        # Calculate overall accuracy
        if all_metrics['total_samples'] > 0:
            all_metrics['overall_accuracy'] = all_metrics['weighted_correct'] / all_metrics['total_samples']
        else:
            all_metrics['overall_accuracy'] = None
        
        # Calculate group accuracies
        for group in ['priority', 'secondary']:
            if group_metrics[group]['total_samples'] > 0:
                group_metrics[group]['accuracy'] = (
                    group_metrics[group]['correct_samples'] / 
                    group_metrics[group]['total_samples']
                )
            else:
                group_metrics[group]['accuracy'] = None
        
        # Calculate weighted metrics (80/20 and 20/80)
        weighted_metrics = {}
        
        # 80/20 weighting (80% priority, 20% secondary)
        if (group_metrics['priority']['accuracy'] is not None and 
            group_metrics['secondary']['accuracy'] is not None):
            weighted_metrics['80_20_accuracy'] = (
                0.8 * group_metrics['priority']['accuracy'] + 
                0.2 * group_metrics['secondary']['accuracy']
            )
        else:
            weighted_metrics['80_20_accuracy'] = None
        
        # 20/80 weighting (20% priority, 80% secondary)
        if (group_metrics['priority']['accuracy'] is not None and 
            group_metrics['secondary']['accuracy'] is not None):
            weighted_metrics['20_80_accuracy'] = (
                0.2 * group_metrics['priority']['accuracy'] + 
                0.8 * group_metrics['secondary']['accuracy']
            )
        else:
            weighted_metrics['20_80_accuracy'] = None
        
        # Combine results
        return {
            'overall_accuracy': all_metrics['overall_accuracy'],
            'total_samples': all_metrics['total_samples'],
            'priority_group_accuracy': group_metrics['priority']['accuracy'],
            'priority_group_samples': group_metrics['priority']['total_samples'],
            'secondary_group_accuracy': group_metrics['secondary']['accuracy'],
            'secondary_group_samples': group_metrics['secondary']['total_samples'],
            'weighted_80_20_accuracy': weighted_metrics['80_20_accuracy'],
            'weighted_20_80_accuracy': weighted_metrics['20_80_accuracy']
        }
    
    def _calculate_aggregate_IntersectionVolume_metrics(self, report):
        """Calculate aggregate IntersectionVolume metrics across all datasets"""
        datasets = report['dataset_reports']
        
        total_samples = 0
        weighted_mae_sum = 0
        weighted_mse_sum = 0
        weighted_bin_accuracy_sum = 0
        total_samples_in_bins = 0

        for dataset_name, dataset_report in datasets.items():
            # Skip if IntersectionVolume performance data is missing or has errors
            if 'IntersectionVolume_performance' not in dataset_report or 'error' in dataset_report['IntersectionVolume_performance']:
                continue

            perf = dataset_report['IntersectionVolume_performance']
            overall_metrics = perf.get('overall_IntersectionVolume_metrics', {})

            # Get metrics for this dataset
            mae = overall_metrics.get('mae')
            mse = overall_metrics.get('mse')
            samples = overall_metrics.get('samples')
            bin_accuracy = overall_metrics.get('overall_bin_accuracy')
            samples_in_binned_range = overall_metrics.get('samples_in_binned_range')

            # Ensure we have valid numbers to work with
            if samples is not None and samples > 0:
                total_samples += samples
                if mae is not None:
                    weighted_mae_sum += mae * samples
                if mse is not None:
                    weighted_mse_sum += mse * samples
                
                if bin_accuracy is not None and samples_in_binned_range is not None and samples_in_binned_range > 0:
                    weighted_bin_accuracy_sum += bin_accuracy * samples_in_binned_range
                    total_samples_in_bins += samples_in_binned_range

        # Calculate overall weighted averages
        overall_mae = weighted_mae_sum / total_samples if total_samples > 0 else None
        overall_mse = weighted_mse_sum / total_samples if total_samples > 0 else None
        overall_bin_accuracy = weighted_bin_accuracy_sum / total_samples_in_bins if total_samples_in_bins > 0 else None

        return {
            'overall_mae': overall_mae,
            'overall_mse': overall_mse,
            'overall_bin_accuracy': overall_bin_accuracy,
            'total_samples': total_samples,
            'total_samples_in_binned_range': total_samples_in_bins
        }
