import os
import pandas as pd
from functools import wraps
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix)

class Evaluator:
    _test_registry = {
        'binary_classification': [],
        'regression': [],
        'classification_and_regression': []
    }

    def __init__(self, config):
        self.config = config
        self.task_type = config.get('task', 'binary_classification')
        self.test_data_loaded = False
        self.datasets = []
        self.intersection_status_column = config['IntersectionStatus']
        self.intersection_volume_column = config['IntersectionVolume']

    @classmethod
    def register_test(cls, task_type):
        def decorator(func):
            cls._test_registry[task_type].append(func)
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def evaluate(self, model):
        if self.config.get('skip_evaluation', False):
            return {'evaluation_status': 'skipped'}

        self._load_test_data()
        report = {
            'task_type': self.task_type,
            'dataset_reports': {}
        }

        for dataset in self.datasets:
            dataset_report = {}
            for test in self._test_registry[self.task_type]:
                test_name = test.__name__
                try:
                    dataset_report[test_name] = test(self, model, dataset)
                except Exception as e:
                    dataset_report[test_name] = {'error': str(e)}
            
            report['dataset_reports'][dataset['name']] = dataset_report

        return report

    def _load_test_data(self):
        """Load all test datasets from configured directory structure"""
        if self.test_data_loaded:
            return

        base_path = self.config['test_data_path']
        intersection_types = [
            'no_intersection', 'point_intersection', 'segment_intersection',
            'polygon_intersection', 'polyhedron_intersection'
        ]

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
                        'X': df.drop(columns=[self.intersection_status_column, self.intersection_volume_column]),
                        'y': df[self.intersection_status_column],
                        'volume': df[self.intersection_volume_column]
                    })

        self.test_data_loaded = True

    @register_test('binary_classification')
    def classification_performance(self, model, dataset):
        """Comprehensive classification evaluation"""
        y_pred = model.predict(dataset['X'])
        y_true = dataset['y']
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

        # Add probabilistic metrics if available
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(dataset['X'])[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except Exception as e:
                metrics['probability_error'] = str(e)

        # Add class distribution
        metrics['class_distribution'] = {
            'positive': y_true.mean(),
            'negative': 1 - y_true.mean()
        }

        return metrics

    # Add new tests using the decorator