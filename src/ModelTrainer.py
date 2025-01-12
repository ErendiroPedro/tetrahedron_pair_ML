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

        self._configure(self.config[self.config["type"]])

    def train_and_validate(self, model):
        self.model = model
        train_loss_values = []
        val_loss_values = []

        train_data_dir = os.path.join(self.config["processed_data_path"], "train")
        val_data_dir = os.path.join(self.config["processed_data_path"], "val")

        train_dataset = CustomDataset(train_data_dir)
        val_dataset = CustomDataset(val_data_dir)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Training and validation loop
        for epoch in range(self.epochs):

            # Train for one epoch
            epoch_train_loss = self._train_epoch(train_loader)
            train_loss_values.append(epoch_train_loss)

            # Validate for one epoch
            epoch_val_loss = self._validate_epoch(val_loader)
            val_loss_values.append(epoch_val_loss)

            print(f"Epoch {epoch + 1}/{self.epochs}, Training Loss: {epoch_train_loss:.4f}")
            print(f"Epoch {epoch + 1}/{self.epochs}, Validation Loss: {epoch_val_loss:.4f}")   

        # Generate and return the training report
        return self.model, self._get_training_report(train_loss_values, val_loss_values)

    def _configure(self, config):
        self.loss_function = config["loss_function"]
        self.optimizer = config["optimizer"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]

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
