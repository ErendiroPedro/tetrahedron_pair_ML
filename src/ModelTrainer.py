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
        self.loss_function = None
        self.optimizer = None
        self.learning_rate = None
        self.batch_size = None
        self.epochs = None
        self.learning_rate_schedule = None
        self.config = config

        self._configure_training(self.config[self.config["type"]])

    def _configure_training(self):
        """Configure training parameters from config."""
        training_config = self.config[self.config["type"]]
        
        # Set training parameters
        self.batch_size = training_config["batch_size"]
        self.epochs = training_config["epochs"]
        self.learning_rate = training_config["learning_rate"]
        self.early_stopping_patience = training_config.get("early_stopping_patience", 5)
        
        # Loss functions setup
        self.task = self.config["model_config"]["task"]
        self.loss_functions = self._setup_loss_functions()
        
        # Optimizer setup
        optimizer_name = training_config["optimizer"].lower()
        if optimizer_name == "adam":
            self.optimizer_class = torch.optim.Adam
        elif optimizer_name == "adamw":
            self.optimizer_class = torch.optim.AdamW
        elif optimizer_name == "sgd":
            self.optimizer_class = torch.optim.SGD
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Learning rate scheduler setup
        self.scheduler_config = training_config.get("learning_rate_schedule")

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
