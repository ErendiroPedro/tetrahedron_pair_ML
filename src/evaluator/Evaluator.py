import os
import pandas as pd
import numpy as np
import torch
import time
import platform

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
        
        self.test_registry = {
            'binary_classification': [self.classification_performance, self.inference_speed],
            'regression': [self.regression_performance, self.inference_speed],
            'classification_and_regression': [self.classification_performance, self.regression_performance, self.inference_speed]
        }

    def evaluate(self, model):
        """
        Evaluate the model across different test datasets.
        
        :param model: Trained model to evaluate
        :return: Comprehensive evaluation report
        """

        if self.config.get('skip_evaluation', False):
            print("-- Skipped Evaluation --")
            return {'evaluation_status': 'skipped'}
        
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

        intersection_types = (
            ['polyhedron_intersection'] 
            if self.task_type == 'regression'
            else [
                'no_intersection', 'point_intersection', 
                'segment_intersection', 'polygon_intersection', 
                'polyhedron_intersection'
            ]
        )

        for itype in intersection_types:
            type_dir = os.path.join(base_path, itype)
            if not os.path.exists(type_dir):
                continue

            for file in os.listdir(type_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(type_dir, file)
                    df = pd.read_csv(file_path)
                    
                    self.datasets.append({
                        'name': f"{itype}/{file}",
                        'type': itype,
                        'intersection_status': df[self.intersection_status_column_name],
                        'intersection_volume': df[self.intersection_volume_column_name],
                        'X': df.drop(columns=[self.intersection_status_column_name, self.intersection_volume_column_name]),
                    })

        self.test_data_loaded = True

    def classification_performance(self, model, dataset):
      """
      Comprehensive classification performance evaluation.

      :param model: Trained model
      :param dataset: Dataset dictionary
      :return: Dictionary of performance metrics
      """
      # Convert DataFrame to PyTorch tensor
      X = torch.tensor(dataset['X'].values, dtype=torch.float32)
      y_true = dataset['intersection_status'].values

      # Ensure the tensor is on the same device as the model
      device = next(model.parameters()).device
      X = X.to(device)

      # Generate predictions
      y_pred = model.predict(X).cpu().numpy()  # Convert predictions back to numpy for metric computation

      # Compute metrics
      metrics = {
         'accuracy': accuracy_score(y_true, y_pred),
         'precision': precision_score(y_true, y_pred),
         'recall': recall_score(y_true, y_pred),
         'f1': f1_score(y_true, y_pred),
         'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
      }

      return metrics

    def regression_performance(self, model, dataset):
        """
        Simplified regression evaluation with interval analysis.
        Focuses on core metrics and 0-0.01 value range performance.
        """
        # Data preparation remains the same
        X = torch.tensor(dataset['X'].values, dtype=torch.float32)
        y_true = dataset['intersection_volume'].values.astype(np.float32)
        
        model.eval()
        device = next(model.parameters()).device
        X = X.to(device)
        
        # Prediction handling
        with torch.no_grad():
            try:
                y_pred = model(X).cpu().numpy().reshape(-1).astype(np.float32)
            except Exception as e:
                return {'error': str(e)}
        
        # Validation check
        if y_true.shape != y_pred.shape:
            return {'error': f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"}
        
        # Interval analysis (0-0.01 range)
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
        """
        Measure inference speed of the model on the dataset including device info.
        
        :param model: Trained model
        :param dataset: Dataset dictionary
        :return: Dictionary with inference speed metrics and device info
        """
        X = torch.tensor(dataset['X'].values, dtype=torch.float32)
        device = next(model.parameters()).device
        X = X.to(device)
        num_samples = X.shape[0]

        predict_method = model.predict if hasattr(model, 'predict') else lambda x: model(x)

        # Warmup runs to avoid initialization overhead
        with torch.no_grad():
            for _ in range(10):
                _ = predict_method(X)

        # Configure repetitions based on device
        repetitions = 30
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
        avg_time_seconds = avg_time_ms / 1000.0

        return {
            'device_info': device_info,
            'total_time_seconds': avg_time_seconds,
            'avg_time_per_sample_seconds': avg_time_seconds / num_samples,
            'samples_per_second': num_samples / avg_time_seconds
        }