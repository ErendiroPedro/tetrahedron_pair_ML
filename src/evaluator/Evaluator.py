import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

class Evaluator:
    def __init__(self, config):
        """
        Initialize the Evaluator with configuration.
        
        :param config: Dictionary containing evaluation configuration
        """
        self.config = config
        self.task_type = config.get('task', 'binary_classification')
        self.test_data_loaded = False
        self.datasets = []
        self.intersection_status_column_name = 'HasIntersection'
        self.intersection_volume_column_name = 'IntersectionVolume'
        
        self.test_registry = {
            'binary_classification': [self.classification_performance],
            'regression': [self.regression_performance],
            'classification_and_regression': [self.classification_performance, self.regression_performance]
        }

    def evaluate(self, model):
        """
        Evaluate the model across different test datasets.
        
        :param model: Trained model to evaluate
        :return: Comprehensive evaluation report
        """
        if self.config.get('skip_evaluation', False):
            return {'evaluation_status': 'skipped'}

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
                        'y': df[self.intersection_status_column_name],
                        'volume': df[self.intersection_volume_column_name],
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
        y_pred = model.predict(dataset['X'])
        y_true = dataset['y']
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(dataset['X'])[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except Exception as e:
                metrics['probability_error'] = str(e)

        metrics['class_distribution'] = {
            'positive': y_true.mean(),
            'negative': 1 - y_true.mean()
        }

        return metrics
    
    def regression_performance(self, model, dataset):
        """
        Comprehensive regression performance evaluation.
        
        :param model: Trained model
        :param dataset: Dataset dictionary
        :return: Dictionary of performance metrics
        """
        pass