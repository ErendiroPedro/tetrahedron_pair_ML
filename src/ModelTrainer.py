from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

class ModelTrainer():
    def __init__(self, config):
        self.model = None
        self.config = config
        self.loss_function = self._setup_loss_functions(config["loss_function"])
        self.optimizer = self._setup_optimizer(config["optimizer"])
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.learning_rate_schedule = config["learning_rate_schedule"]
        self.early_stopping_patience = config["early_stopping_patience"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_loss_functions(self, task_name):
        if task_name == "binary_classification":
            return F.binary_cross_entropy
        elif task_name == "regression":
            return F.mse_loss
        elif task_name == "classification_and_regression":
            return self._classification_and_regression_loss
        else:
            raise ValueError(f"Unsupported task: {task_name}")

    def _classification_and_regression_loss(self, output, target):
        cls_loss = F.binary_cross_entropy(output[:, 0], target[:, 0])
        reg_loss = F.mse_loss(output[:, 1], target[:, 1])
        return cls_loss + reg_loss
    
    def _setup_optimizer(self, optimizer_name):    
        if optimizer_name == "adam":
            return torch.optim.Adam
        elif optimizer_name == "adamw":
            return torch.optim.AdamW
        elif optimizer_name == "sgd":
            return torch.optim.SGD
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for data in train_loader:
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.binary_cross_entropy(output, data.y.float().unsqueeze(1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        return epoch_loss

    def _validate_epoch(self, val_loader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for data in val_loader:
                output = self.model(data)
                val_loss = F.binary_cross_entropy(output, data.y.float().unsqueeze(1))
                total_val_loss += val_loss.item()

        epoch_val_loss = total_val_loss / len(val_loader)
        return epoch_val_loss

    def _get_training_report(self, train_loss_values, val_loss_values):
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_values, label='Training Loss')
        plt.plot(val_loss_values, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()

        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()

        # Create the training report
            # Add other metrics to the report as needed
        training_report =  plot_base64

        return training_report
