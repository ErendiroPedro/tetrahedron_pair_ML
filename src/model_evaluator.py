import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.model_architecture import GraphPairClassifier
from src.data_processor import DataProcessor

class ModelEvaluator:
    def __init__(self, raw_data_root, model_path, batch_size=10000):
        self.batch_size = batch_size
        self.raw_data_root = raw_data_root
        self.model_path = model_path
        self.data_processor = DataProcessor(raw_data_root)
        self.model = self.load_model()

    def load_model(self):
        model = GraphPairClassifier(num_input_features=3)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model

    def process_and_load_data(self, subset):
        dataset = self.data_processor.process_entries(subset='val', batch_size=self.batch_size)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, follow_batch=['x_1', 'x_2'])

    def evaluate(self):
        test_loader = self.process_and_load_data(subset='val')
        y_true = []
        y_pred = []

        with torch.no_grad():
            for data in test_loader:
                outputs = self.model(data)
                predictions = (outputs > 0.5).float()
                y_true.extend(data.y.float().unsqueeze(1).cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        self.print_confusion_matrix(y_true, y_pred)
        self.print_accuracy(y_true, y_pred)

    def print_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')

    def print_accuracy(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Test Accuracy: {accuracy}")

