from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.model_architecture import GraphPairClassifier, MLPClassifier
from src.data_processor import GraphDataProcessor, TabularDataProcessor
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, model, optimizer, data_processor, batch_size=8, epochs=20):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_processor = data_processor

    @abstractmethod
    def _train_epoch(self, train_loader):
        pass

    @abstractmethod
    def _validate_epoch(self, val_loader):
        pass

    def train_and_validate(self):
        loss_values = []
        val_loss_values = []

        for epoch in range(self.epochs):
            train_data_list = self.data_processor.process_entries(subset='train', batch_size=self.batch_size)
            val_data_list = self.data_processor.process_entries(subset='val', batch_size=self.batch_size)

            train_loader = DataLoader(train_data_list, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_data_list, batch_size=self.batch_size, shuffle=True)

            epoch_loss = self._train_epoch(train_loader)
            print(f'Epoch {epoch+1}, Training Loss: {epoch_loss}')
            loss_values.append(epoch_loss)

            epoch_val_loss = self._validate_epoch(val_loader)
            print(f'Epoch {epoch+1}, Validation Loss: {epoch_val_loss}')
            val_loss_values.append(epoch_val_loss)

        torch.save(self.model.state_dict(), 'model_state_dict.pt')
        self._plot_loss_curve(loss_values, val_loss_values)
  
    def _plot_loss_curve(self, loss_values, val_loss_values):

        plt.figure(figsize=(10, 5))
        plt.plot(loss_values, label='Training Loss')
        plt.plot(val_loss_values, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.savefig('loss_curve.png')

class TabularTrainer(BaseTrainer):
    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for data_batch in train_loader:

            features = data_batch[:, :-1]  # Get features
            labels = data_batch[:, -1]  # Get label

            self.optimizer.zero_grad()
            output = self.model(features)
            loss = F.binary_cross_entropy(output, labels.float().unsqueeze(1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        return epoch_loss

    def _validate_epoch(self, val_loader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for data_batch in val_loader:
                features = data_batch[:, :-1]  # Get features
                labels = data_batch[:, -1]  # Get label

                output = self.model(features)
                val_loss = F.binary_cross_entropy(output, labels.float().unsqueeze(1))
                total_val_loss += val_loss.item()

        epoch_val_loss = total_val_loss / len(val_loader)
        return epoch_val_loss

class GraphTrainer(BaseTrainer):
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