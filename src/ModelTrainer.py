from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from typing import Tuple, List, Dict
import time
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class ModelTrainer:
    def __init__(self, config: Dict):
        """
        Initialize the ModelTrainer with configuration settings.
        
        :param config: Dictionary containing training configuration
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = config
        self.loss_function = self._setup_loss_functions(config.get("loss_function", "binary_classification"))
        self.learning_rate = config.get("learning_rate", 0.001)
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 10)
        self.processed_data_path = config.get("processed_data_path", ".")

        self.model = None # Will be set up during training
        self.optimizer = None # Will be set up during training

    def _setup_loss_functions(self, task_name: str):
        """
        Set up the appropriate loss function based on the task.
        
        :param task_name: Name of the task (classification, regression, etc.)
        :return: Loss function
        """
        loss_functions_map = {
            "binary_classification": F.binary_cross_entropy_with_logits,
            "regression": self._mape_loss,
            "multiclass_classification": F.cross_entropy,
            "classification_and_regression": self._classification_and_regression_loss
        }
        return loss_functions_map.get(task_name, F.binary_cross_entropy_with_logits)
    
    def _mape_loss(self, output, target):
        """
        Mean Absolute Percentage Error (MAPE) loss.
        
        :param output: Model predictions (tensor)
        :param target: Ground truth values (tensor)
        :param epsilon: Small constant to avoid division by zero
        :return: MAPE loss (scalar)
        """
        epsilon=1e-16
        if not isinstance(output, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError(f"Expected tensors, got {type(output)} and {type(target)}.")
        return torch.mean(torch.abs((target - output) / (target + epsilon)))

    def _classification_and_regression_loss(self, output, target):
        """
        Combined loss function for classification and regression tasks.
        
        :param output: Model output
        :param target: Target values
        :return: Combined loss
        """
        cls_loss = F.binary_cross_entropy_with_logits(output[:, 0], target[:, 0])
        reg_loss = self._mape_loss(output[:, 1], target[:, 1])
        return cls_loss + reg_loss

    def _setup_optimizer(self, optimizer_name: str):
        """
        Set up the optimizer based on the configuration.
        
        :param optimizer_name: Name of the optimizer
        :return: Optimizer class
        """
        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD
        }
        return optimizers.get(optimizer_name, torch.optim.Adam)

    def _setup_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Setup training and validation data loaders from CSV files.
        
        :return: Train and validation data loaders
        """
        # Construct full paths to train and validation CSV files
        train_data_path = os.path.join(self.processed_data_path, "train", "train_data.csv")
        val_data_path = os.path.join(self.processed_data_path, "val", "val_data.csv")
        
        # Create datasets
        train_dataset = TetrahedronDataset(train_data_path)
        val_dataset = TetrahedronDataset(val_data_path)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4
        )
        
        return train_loader, val_loader

    def train_and_validate(self, model: torch.nn.Module) -> Tuple[torch.nn.Module, str]:
        """
        Train and validate the model.
        
        :param model: PyTorch model to train
        :return: Trained model and training report
        """
        # Move model to device
        if self.config["skip_training"]:
            print("-- Skipped training model --")
            return
        print(f"-- Training model using device: {self.device} --")
        self.model = model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer(self.config.get("optimizer", "adam"))(
            params=self.model.parameters(),
            lr=self.learning_rate
        )
        
        # Metrics tracking
        train_losses, val_losses = [], []
        
        # Data loaders
        train_loader, val_loader = self._setup_data_loaders()
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Training
            train_loss = self._train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self._validate_epoch(val_loader)
            val_losses.append(val_loss)
            
            # Log progress
            elapsed_time = time.time() - start_time
            self._log_progress(epoch, train_loss, val_loss, elapsed_time)
        print("---- Training Complete ----")
        return self.model, self._generate_training_report(train_losses, val_losses)

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        :param train_loader: Training data loader
        :return: Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_status, batch_volume in train_loader:
            batch_x = batch_x.to(self.device)
            batch_status = batch_status.to(self.device)
            batch_volume = batch_volume.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(batch_x)

            if self.config["loss_function"] == "binary_classification":
                loss = self.loss_function(output, batch_status)
            elif self.config["loss_function"] == "regression":
                loss = self.loss_function(output, batch_volume)
            elif self.config["loss_function"] == "classification_and_regression":
                loss = self.loss_function(output, torch.cat([batch_status, batch_volume], dim=1))
            else:
                raise ValueError("Invalid loss function specified in configuration.")
                
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """
        Validate for one epoch.
        
        :param val_loader: Validation data loader
        :return: Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_status, batch_volume in val_loader:
                batch_x = batch_x.to(self.device)
                batch_status = batch_status.to(self.device)
                batch_volume = batch_volume.to(self.device)

                output = self.model(batch_x)

                # Compute loss based on the task
                if self.config["loss_function"] == "binary_classification":
                    loss = self.loss_function(output, batch_status)
                elif self.config["loss_function"] == "regression":
                    loss = self.loss_function(output, batch_volume)
                elif self.config["loss_function"] == "classification_and_regression":
                    loss = self.loss_function(output, torch.cat([batch_status, batch_volume], dim=1))
                else:
                    raise ValueError("Invalid loss function specified in configuration.")

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def _log_progress(self, epoch: int, train_loss: float, val_loss: float, elapsed_time: float):
        """
        Log training progress.
        
        :param epoch: Current epoch number
        :param train_loss: Training loss for the epoch
        :param val_loss: Validation loss for the epoch
        :param elapsed_time: Time taken for the epoch
        """
        print(f"Epoch {epoch + 1}/{self.epochs} - Time: {elapsed_time:.2f}s")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")

    def _generate_training_report(self, train_losses: List[float], val_losses: List[float]) -> str:
        """
        Generate training report with loss plots.
        
        :param train_losses: List of training losses
        :param val_losses: List of validation losses
        :return: Base64 encoded plot image
        """
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()
        
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
class TetrahedronDataset(Dataset):
    def __init__(self, data_path):
        """
        Load data from a CSV file.
        
        :param data_path: Path to the CSV file
        """
        # Load the CSV file
        df = pd.read_csv(data_path)
        
        # Extract features and targets
        self.features = df.iloc[:, :-2].values
        self.intersection_status = df.iloc[:, -1].values
        self.intersection_volume = df.iloc[:, -2].values
        
        # Convert to tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.intersection_status = torch.tensor(self.intersection_status, dtype=torch.float32).reshape(-1, 1)
        self.intersection_volume = torch.tensor(self.intersection_volume, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.intersection_status[idx], self.intersection_volume[idx]