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
import DataProcessor as dp

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.task_type = config.get('task', 'binary_classification')
        self.test_data_loaded = False
        self.datasets = []
        self.intersection_status_column_name = 'HasIntersection'
        self.intersection_volume_column_name = 'IntersectionVolume'
        self.dp = dp.DataProcessor(config)
        
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
        
        report = {
            'task_type': self.task_type,
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

                    if self.config.get('transformations', None) is not None:
                        start = time.time()
                        df = self.dp.transform_data(df, self.config)
                        transformation_time_ms = (time.time() - start) * 1000
                    
                    if self.config.get('augmentations', None) is not None:
                        start = time.time()
                        df = self.dp.augment_data(df, self.config)
                        augmentation_time_ms = (time.time() - start) * 1000
                    
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

        # Ensure data is on the same device as the model
        device = next(model.parameters()).device
        X = X.to(device)

        with torch.no_grad(): # Disables gradient computation for efficiency
            try:
                y_pred = model.predict(X).cpu().numpy()
                if self.task_type == 'classification_and_regression':
                    y_pred = y_pred[:, 0]
            except Exception as e:
                return {'error': str(e)}
            
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

        return metrics

    def regression_performance(self, model, dataset):

        X = torch.tensor(dataset['X'].values, dtype=torch.float32)
        y_true = dataset['intersection_volume'].values.astype(np.float32)
        
        model.eval()
        device = next(model.parameters()).device
        X = X.to(device)
        
        with torch.no_grad():  # Disables gradient computation for efficiency
            try:
                y_pred = model.predict(X).cpu().numpy().astype(np.float32)
                if self.task_type == 'classification_and_regression':
                    y_pred = y_pred[:, 1] 
            except Exception as e:
                return {'error': str(e)}

        assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"
        
        intervals = np.linspace(0, 0.01, 6)  # 5 intervals
        interval_metrics = {}
        
        for i in range(len(intervals)-1):
            low = intervals[i]
            high = intervals[i+1]
            mask = (y_true >= low) & (y_true < high)
            
            if np.sum(mask) > 0:
                interval_metrics[f'{low:.3f}-{high:.3f}'] = {
                    'mae': float(mean_absolute_error(y_true[mask], y_pred[mask])),
                    'mse': float(mean_squared_error(y_true[mask], y_pred[mask])),
                    'samples': int(np.sum(mask))
                }
            else:
                interval_metrics[f'{low:.3f}-{high:.3f}'] = {
                    'mae': None,
                    'mse': None,
                    'samples': 0
                }
        
        return interval_metrics

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