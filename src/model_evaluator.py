import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.model_architecture import TabularClassifier, GraphPairClassifier
from src.data_processor import TabularProcessor, GraphProcessor

class ModelEvaluator:
    def __init__(self, raw_data_root, model_path, batch_size=750):
        self.batch_size = batch_size
        self.raw_data_root = raw_data_root
        self.model_path = model_path
        self.data_processor = GraphProcessor(raw_data_root)
        self.model = GraphPairClassifier()
        self.test_loader = self._get_graph_loader()


    def evaluate(self):
        # setup model
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        # start evaluation
        y_true = []
        y_pred = []

        with torch.no_grad():
            
            for i in range(200):
                self.test_loader = self._get_graph_loader()
                for data in self.test_loader:
                    outputs = self.model(data)
                    predictions = (outputs > 0.5).float()
                    y_true.extend(data.y.float().unsqueeze(1).cpu().numpy())
                    y_pred.extend(predictions.cpu().numpy())

            # for i in range(200):
            #     self.test_loader = self._get_tabular_loader()
            #     for data in self.test_loader:
            #         inputs = data[:, :-1]  # all features except the last one
            #         labels = data[:, -1]  # only the last feature
            #         outputs = self.model(inputs)
            #         predictions = (outputs > 0.5).float()
            #         y_true.extend(labels.float().unsqueeze(1).cpu().numpy())
            #         y_pred.extend(predictions.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        self._print_confusion_matrix(y_true, y_pred)
        self._print_metrics(y_true, y_pred)

######################
## Helper Functions ##
######################
        
    def _get_tabular_loader(self):
        dataset = self.data_processor.process_entries(subset='val', batch_size=self.batch_size)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _get_graph_loader(self):
        dataset = self.data_processor.process_entries(subset='val', batch_size=self.batch_size)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, follow_batch=['x_1', 'x_2'])

    def _print_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')

    def _print_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        metrics = f"Evaluation Metrics\n\nTest Accuracy: {accuracy*100:.2f}%\nPrecision: {precision*100:.2f}%\nRecall: {recall*100:.2f}%\nF1 Score: {f1*100:.2f}%"
        print(metrics)

        with open('evaluation_metrics.txt', 'w') as f:
            f.write(metrics)

