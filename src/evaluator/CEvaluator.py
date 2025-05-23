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
        self.task_type = config.get('task', 'binary_classification')
        self.test_data_loaded = False
        self.datasets = []
        self.intersection_status_column_name = 'HasIntersection'
        self.intersection_volume_column_name = 'IntersectionVolume'
        self.dp = dp.CDataProcessor(config)
        
        self.test_registry = {
            'binary_classification': [self.classification_performance, self.point_wise_permutation_consistency, self.tetrahedron_wise_permutation_consistency, self.inference_speed],
            'regression': [self.regression_performance, self.point_wise_permutation_consistency, self.tetrahedron_wise_permutation_consistency, self.inference_speed],
            'classification_and_regression': [self.classification_performance, self.point_wise_permutation_consistency, self.tetrahedron_wise_permutation_consistency, self.regression_performance, self.inference_speed]
        }

        self.dataset_folders = (
            ['polyhedron_intersection', 'big_dataset'] 
            if self.task_type == 'regression'
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

        if self.task_type in ['binary_classification', 'classification_and_regression']:
            report['aggregate_binary_classification_metrics'] = self._calculate_aggregate_classification_metrics(report)

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
                    if self.task_type == 'classification_and_regression':
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

    def regression_performance(self, model, dataset):
        X = torch.tensor(dataset['X'].values, dtype=torch.float32)
        y_true = dataset['intersection_volume'].values.astype(np.float32)

        model.eval()
        device = next(model.parameters()).device
        X = X.to(device)

        num_repeats = 50
        all_interval_metrics = []

        try:
            with torch.no_grad():
                for _ in range(num_repeats):
                    y_pred = model.predict(X).cpu().numpy().astype(np.float32)
                    if self.task_type == 'classification_and_regression':
                        y_pred = y_pred[:, 1]

                    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

                    intervals = np.linspace(0, 0.01, 6)  # 5 intervals
                    interval_metrics = {}

                    for i in range(len(intervals)-1):
                        low = intervals[i]
                        high = intervals[i+1]
                        
                        # Samples that truly belong in this bin
                        true_bin_mask = (y_true >= low) & (y_true < high)
                        # Samples predicted to be in this bin
                        pred_bin_mask = (y_pred >= low) & (y_pred < high)
                        
                        # Count samples that are correctly placed in this bin
                        true_bin_samples = np.sum(true_bin_mask)
                        correct_bin_predictions = np.sum(true_bin_mask & pred_bin_mask)
                        
                        # Calculate bin prediction accuracy (percentage of samples correctly predicted in this bin)
                        bin_accuracy = float(correct_bin_predictions / true_bin_samples) if true_bin_samples > 0 else 0.0

                        if np.sum(true_bin_mask) > 0:
                            mae = mean_absolute_error(y_true[true_bin_mask], y_pred[true_bin_mask])
                            mse = mean_squared_error(y_true[true_bin_mask], y_pred[true_bin_mask])
                            interval_metrics[f'{low:.3f}-{high:.3f}'] = {
                                'mae': float(mae),
                                'mse': float(mse),
                                'samples': int(np.sum(true_bin_mask)),
                                'correct_bin_predictions': int(correct_bin_predictions),
                                'bin_accuracy': bin_accuracy
                            }
                        else:
                            interval_metrics[f'{low:.3f}-{high:.3f}'] = {
                                'mae': None,
                                'mse': None,
                                'samples': 0,
                                'correct_bin_predictions': 0,
                                'bin_accuracy': None
                            }

                    all_interval_metrics.append(interval_metrics)
        except Exception as e:
            return {'error': str(e)}

        # Average metrics across all runs for each interval
        avg_interval_metrics = {}
        if not all_interval_metrics:
            return avg_interval_metrics

        interval_keys = all_interval_metrics[0].keys()

        for key in interval_keys:
            maes = []
            mses = []
            accuracies = []
            samples = all_interval_metrics[0][key]['samples']
            correct_preds = 0
            
            for run in all_interval_metrics:
                current = run[key]
                if current['mae'] is not None:
                    maes.append(current['mae'])
                if current['mse'] is not None:
                    mses.append(current['mse'])
                if current['bin_accuracy'] is not None:
                    accuracies.append(current['bin_accuracy'])
                if 'correct_bin_predictions' in current:
                    correct_preds += current['correct_bin_predictions']
            
            # Average over all runs
            avg_mae = np.mean(maes) if maes else None
            avg_mse = np.mean(mses) if mses else None
            avg_bin_accuracy = np.mean(accuracies) if accuracies else None
            
            # For the average number of correct predictions, we divide by num_repeats
            avg_correct_preds = correct_preds / num_repeats if num_repeats > 0 else 0

            avg_interval_metrics[key] = {
                'repetitions': num_repeats,
                'mae': avg_mae,
                'mse': avg_mse,
                'samples': samples,
                'correct_bin_predictions': int(avg_correct_preds),
                'bin_accuracy': avg_bin_accuracy
            }

        # Calculate overall bin accuracy across all intervals
        total_samples = sum(m['samples'] for m in avg_interval_metrics.values())
        total_correct = sum(m.get('correct_bin_predictions', 0) for m in avg_interval_metrics.values())
        overall_bin_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # Add overall bin accuracy to results
        avg_interval_metrics['overall_bin_accuracy'] = overall_bin_accuracy

        return avg_interval_metrics

    def _calculate_aggregate_regression_metrics(self, report):
        """Calculate aggregate regression metrics across all datasets using existing results"""
        aggregate_metrics = {
            'total_samples': 0,
            'bin_accuracy': {},
            'overall_mae': 0,
            'overall_mse': 0,
            'weighted_samples': 0,
            'error_datasets': 0,  # Track how many datasets had errors
            'overall_bin_accuracy': 0  # New field for overall bin prediction accuracy
        }
        
        # Define the intervals as used in regression_performance
        intervals = np.linspace(0, 0.01, 6)  # 5 intervals
        for i in range(len(intervals)-1):
            interval_key = f'{intervals[i]:.3f}-{intervals[i+1]:.3f}'
            aggregate_metrics['bin_accuracy'][interval_key] = {
                'total_samples': 0,
                'mae': 0,
                'mse': 0,
                'weighted_mae': 0,
                'weighted_mse': 0,
                'correct_bin_predictions': 0,
                'weighted_bin_accuracy': 0  # For weighted average calculation
            }
        
        # Variables to track overall bin accuracy
        total_bin_correct = 0
        
        # Process all regression datasets
        for dataset_name, dataset_report in report['dataset_reports'].items():
            # Skip datasets without regression performance
            if 'regression_performance' not in dataset_report:
                continue
                
            perf = dataset_report['regression_performance']
            
            # Handle error datasets gracefully
            if not isinstance(perf, dict) or 'error' in perf:
                aggregate_metrics['error_datasets'] += 1
                continue
            
            # Process each interval
            for interval_key, interval_metrics in perf.items():
                # Skip overall bin accuracy key
                if interval_key == 'overall_bin_accuracy':
                    continue
                    
                # Skip invalid or empty intervals
                if not isinstance(interval_metrics, dict) or 'samples' not in interval_metrics:
                    continue
                    
                samples = interval_metrics.get('samples', 0)
                mae = interval_metrics.get('mae')
                mse = interval_metrics.get('mse')
                bin_accuracy = interval_metrics.get('bin_accuracy')
                correct_bin_predictions = interval_metrics.get('correct_bin_predictions', 0)
                
                if samples > 0:
                    # Update interval metrics
                    aggregate_metrics['bin_accuracy'][interval_key]['total_samples'] += samples
                    
                    if mae is not None and mse is not None:
                        aggregate_metrics['bin_accuracy'][interval_key]['weighted_mae'] += mae * samples
                        aggregate_metrics['bin_accuracy'][interval_key]['weighted_mse'] += mse * samples
                        
                        # Update overall error metrics
                        aggregate_metrics['weighted_samples'] += samples
                        aggregate_metrics['overall_mae'] += mae * samples
                        aggregate_metrics['overall_mse'] += mse * samples
                    
                    # Update bin accuracy metrics
                    aggregate_metrics['bin_accuracy'][interval_key]['correct_bin_predictions'] += correct_bin_predictions
                    if bin_accuracy is not None:
                        aggregate_metrics['bin_accuracy'][interval_key]['weighted_bin_accuracy'] += bin_accuracy * samples
                    
                    # Update overall bin accuracy metrics
                    total_bin_correct += correct_bin_predictions
                    
                    # Update total samples count
                    aggregate_metrics['total_samples'] += samples
        
        # Calculate aggregated metrics
        if aggregate_metrics['weighted_samples'] > 0:
            aggregate_metrics['overall_mae'] /= aggregate_metrics['weighted_samples']
            aggregate_metrics['overall_mse'] /= aggregate_metrics['weighted_samples']
        
        # Calculate overall bin accuracy
        if aggregate_metrics['total_samples'] > 0:
            aggregate_metrics['overall_bin_accuracy'] = total_bin_correct / aggregate_metrics['total_samples']
        
        # Calculate per-interval average metrics
        for interval_key, metrics in aggregate_metrics['bin_accuracy'].items():
            if metrics['total_samples'] > 0:
                # Error metrics
                if 'weighted_mae' in metrics and 'weighted_mse' in metrics:
                    metrics['mae'] = metrics['weighted_mae'] / metrics['total_samples']
                    metrics['mse'] = metrics['weighted_mse'] / metrics['total_samples']
                    
                # Bin accuracy metrics
                if 'weighted_bin_accuracy' in metrics:
                    metrics['bin_accuracy'] = metrics['weighted_bin_accuracy'] / metrics['total_samples']
                else:
                    metrics['bin_accuracy'] = metrics['correct_bin_predictions'] / metrics['total_samples']
                
                # Remove temporary weighted values
                if 'weighted_mae' in metrics:
                    metrics.pop('weighted_mae')
                if 'weighted_mse' in metrics:
                    metrics.pop('weighted_mse')
                if 'weighted_bin_accuracy' in metrics:
                    metrics.pop('weighted_bin_accuracy')
        
        return aggregate_metrics

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
            
            rtol = 1e-2
            atol = 1e-8

            # Process predictions based on task type
            if self.task_type == 'binary_classification':
                consistent = (pred_original == pred_swapped)
                consistency_rate = float(np.mean(consistent))

                result = {
                    "classification_consistency_rate": consistency_rate,
                    "total_samples": X.shape[0]
                }
            elif self.task_type == 'regression':
                consistent = np.isclose(pred_original, pred_swapped, rtol=rtol, atol = atol)
                consistency_rate = float(np.mean(consistent))
                mad = float(np.mean(np.abs(pred_original - pred_swapped)))
                result = {
                    "regression_consistency_rate": consistency_rate,
                    "consistency_thresholds": {"rtol": rtol, "atol": atol},
                    "mean_absolute_difference": mad,
                    "total_samples": X.shape[0]
                }
            elif self.task_type == 'classification_and_regression':
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
                    "regression_consistency_rate": consistency_rate_reg,
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
            if not hasattr(self, 'task_type') or self.task_type not in ['binary_classification', 'regression', 'classification_and_regression']:
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

            # Permute points within tetrahedrons
            if not hasattr(gu, 'permute_points_within_tetrahedrons'):
                raise AttributeError("The function 'permute_points_within_tetrahedrons' is missing.")
            X_permuted = gu.permute_points_within_tetrahedrons(X)

            # Get predictions
            pred_original = model.predict(X).cpu().numpy().astype(np.float32)
            pred_permuted = model.predict(X_permuted).cpu().numpy().astype(np.float32)

            # Calculate consistency metrics
            rtol = 1e-2
            atol = 1e-8
            result = {}
            if self.task_type == 'binary_classification':
                consistent = (pred_original == pred_permuted)
                result["classification_consistency_rate"] = float(np.mean(consistent))

            elif self.task_type == 'regression':
                consistent = np.isclose(pred_original, pred_permuted, rtol=rtol, atol=atol)
                result["regression_consistency_rate"] = float(np.mean(consistent))
                result["consistency_thresholds"] = {"rtol": rtol, "atol": atol}
                result["mean_absolute_difference"] = float(np.mean(np.abs(pred_original - pred_permuted)))

            elif self.task_type == 'classification_and_regression':
                cls_original = pred_original[:, 0]
                cls_swapped = pred_permuted[:, 0]
                reg_original = pred_original[:, 1]
                reg_swapped = pred_permuted[:, 1]

                cls_consistent = (cls_original == cls_swapped)
                reg_consistent = np.isclose(reg_original, reg_swapped, rtol=rtol, atol=atol)

                result["classification_consistency_rate"] = float(np.mean(cls_consistent))
                result["regression_consistency_rate"] = float(np.mean(reg_consistent))
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

    def _calculate_aggregate_classification_metrics(self, report):
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
                
            # Get dataset type from dataset name (e.g., 'no_intersection/file.csv' -> 'no_intersection')
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
