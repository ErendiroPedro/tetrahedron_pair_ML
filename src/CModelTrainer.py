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

class CModelTrainer:

    def __init__(self, config: Dict):
        """
        Initialize the ModelTrainer with configuration settings.
        
        :param config: Dictionary containing training configuration
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_function = self._setup_loss_functions(config.get("loss_function", "IntersectionStatus"))
        self.learning_rate = config.get("learning_rate", 0.001)
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 5)
        self.processed_data_path = config.get("processed_data_path", ".")

        self.model = None # Will be set up during training
        self.optimizer = None # Will be set up during training

    def train_and_validate(self, model: torch.nn.Module) -> Tuple[torch.nn.Module, str]:
        """
        Train and validate the model.
        
        :param model: PyTorch model to train
        :return: Trained model and training report
        """
        print(f"-- Training model using: {self.device} --")

        self.model = model.to(self.device)
        
        self.optimizer = self._setup_optimizer(self.config.get("optimizer", "adam"))(
            params=self.model.parameters(),
            lr=self.learning_rate
        )
        
        train_losses, val_losses = [], []
        train_loader, val_loader = self._setup_data_loaders()
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Training
            train_loss = self._train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self._validate_epoch(val_loader)
            val_losses.append(val_loss)
            
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

            if self.config["loss_function"] == "IntersectionStatus":
                loss = self.loss_function(output, batch_status)
            elif self.config["loss_function"] == "IntersectionVolume":
                loss = self.loss_function(output, batch_volume)
            elif self.config["loss_function"] == "IntersectionStatus_IntersectionVolume":
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
                if self.config["loss_function"] == "IntersectionStatus":
                    loss = self.loss_function(output, batch_status)
                elif self.config["loss_function"] == "IntersectionVolume":
                    loss = self.loss_function(output, batch_volume)
                elif self.config["loss_function"] == "IntersectionStatus_IntersectionVolume":
                    loss = self.loss_function(output, torch.cat([batch_status, batch_volume], dim=1))
                else:
                    raise ValueError("Invalid loss function specified in configuration.")

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def _setup_loss_functions(self, task_name: str):
        """
        Set up the appropriate loss function based on the task.
        
        :param task_name: Name of the task (classification, IntersectionVolume, etc.)
        :return: Loss function
        """
        loss_functions_map = {
            "IntersectionStatus": F.binary_cross_entropy_with_logits,
            "IntersectionVolume": self._rmlse,# self._mape_loss,
            "IntersectionStatus_IntersectionVolume": self._combined_loss
        }
        return loss_functions_map.get(task_name, F.binary_cross_entropy_with_logits)
    
    def _rmlse(self, output, target):
        log_true = torch.log1p(target)  # log(1 + y)
        log_pred = torch.log1p(output)  # log(1 + y_hat)
        assert log_pred.shape == log_true.shape, f"Shape mismatch: {log_pred.shape} vs {log_true.shape}"
        loss = torch.sqrt(torch.mean((log_true - log_pred) ** 2))
        return loss

    def _mape_loss(self, output, target):
        return torch.mean(torch.abs((target - output) / target))

    def _combined_loss(self, output, target):
        """Balanced combined loss with dynamic weighting"""

        cls_target = target[:, 0].float()
        reg_target = target[:, 1].float()
        
        cls_loss = F.binary_cross_entropy_with_logits(
            output[:, 0], 
            cls_target
        )

        reg_loss = self._rmlse(
            output[:, 1], 
            reg_target
        )
   
        total_loss = 0.5 * cls_loss + 0.5 * reg_loss
        return total_loss

    def _setup_optimizer(self, optimizer_name: str):
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

    def _log_progress(self, epoch: int, train_loss: float, val_loss: float, elapsed_time: float):
        print(f"Epoch {epoch + 1}/{self.epochs} - Time: {elapsed_time:.2f}s")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")

    def _generate_training_report(self, train_losses: List[float], val_losses: List[float]) -> str:
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

    def fine_tune(self, model, fine_tune_data):
        """
        Fine-tune a pre-trained model with new data
        
        Args:
            model: Pre-trained model to fine-tune
            fine_tune_data: New dataset for fine-tuning
            
        Returns:
            Tuple of (fine_tuned_model, training_report)
        """
        print("Starting fine-tuning process...")
        
        # Prepare data loaders for fine-tuning data
        train_loader, val_loader = self._prepare_fine_tune_dataloaders(fine_tune_data)
        
        # Set up optimizer with potentially lower learning rate for fine-tuning
        fine_tune_lr = self.config.get('fine_tune_learning_rate', self.config.get('learning_rate', 0.001) * 0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=fine_tune_lr)
        
        # Fine-tune for fewer epochs
        fine_tune_epochs = self.config.get('fine_tune_epochs', 5)
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
        
        for epoch in range(fine_tune_epochs):
            print(f"Fine-tuning Epoch {epoch+1}/{fine_tune_epochs}")
            
            # Training phase
            model.train()
            train_loss = self._train_epoch_fine_tune(model, train_loader, optimizer)
            
            # Validation phase
            model.eval()
            val_loss = self._validate_epoch_fine_tune(model, val_loader)  # Pass model as parameter
            
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['epoch'].append(epoch + 1)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        training_report = f"Fine-tuning completed. Final train loss: {train_loss:.4f}, Final val loss: {val_loss:.4f}"
        
        return model, training_report

    def _prepare_fine_tune_dataloaders(self, fine_tune_data):
        """Prepare data loaders for fine-tuning"""
        # Split fine-tune data into train/val
        from sklearn.model_selection import train_test_split
        
        train_data, val_data = train_test_split(
            fine_tune_data, 
            test_size=0.2, 
            random_state=42
        )
        
        # Create datasets and data loaders
        train_dataset = TetrahedronDataset(train_data)
        val_dataset = TetrahedronDataset(val_data)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get('batch_size', 32), 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.get('batch_size', 32), 
            shuffle=False
        )
        
        return train_loader, val_loader

    def _train_epoch_fine_tune(self, model, train_loader, optimizer):
        """Training epoch specifically for fine-tuning"""
        total_loss = 0
        num_batches = 0
        
        # Ensure model is on correct device
        model = model.to(self.device)
        
        for batch_x, batch_status, batch_volume in train_loader:
            # Move all tensors to the same device as model
            batch_x = batch_x.to(self.device)
            batch_status = batch_status.to(self.device)
            batch_volume = batch_volume.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch_x)
            
            # Calculate loss based on the task (same as regular training)
            if self.config["loss_function"] == "IntersectionStatus":
                loss = self.loss_function(output, batch_status)
            elif self.config["loss_function"] == "IntersectionVolume":
                loss = self.loss_function(output, batch_volume)
            elif self.config["loss_function"] == "IntersectionStatus_IntersectionVolume":
                loss = self.loss_function(output, torch.cat([batch_status, batch_volume], dim=1))
            else:
                raise ValueError("Invalid loss function specified in configuration.")
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0

    def _validate_epoch_fine_tune(self, model, val_loader):
        """Validation epoch specifically for fine-tuning"""
        total_loss = 0
        num_batches = 0
        
        model.eval()
        with torch.no_grad():
            for batch_x, batch_status, batch_volume in val_loader:
                batch_x = batch_x.to(self.device)
                batch_status = batch_status.to(self.device)
                batch_volume = batch_volume.to(self.device)
                
                # Forward pass
                output = model(batch_x)
                
                # Calculate loss based on the task
                if self.config["loss_function"] == "IntersectionStatus":
                    loss = self.loss_function(output, batch_status)
                elif self.config["loss_function"] == "IntersectionVolume":
                    loss = self.loss_function(output, batch_volume)
                elif self.config["loss_function"] == "IntersectionStatus_IntersectionVolume":
                    loss = self.loss_function(output, torch.cat([batch_status, batch_volume], dim=1))
                else:
                    raise ValueError("Invalid loss function specified in configuration.")
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0

class TetrahedronDataset(Dataset):
    def __init__(self, data_source):
        """
        Load data from a CSV file path or pandas DataFrame.
        
        :param data_source: Path to CSV file (str) or pandas DataFrame
        """
        if isinstance(data_source, str):
            # Load from CSV file
            df = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            # Use DataFrame directly
            df = data_source.copy()
        else:
            raise ValueError("data_source must be either a file path (str) or pandas DataFrame")
        
        # Extract features and targets
        self.features = df.iloc[:, :-2].values
        self.intersection_volume = df.iloc[:, -2].values
        self.intersection_status = df.iloc[:, -1].values
        
        # Convert to tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.intersection_status = torch.tensor(self.intersection_status, dtype=torch.float32).reshape(-1, 1)
        self.intersection_volume = torch.tensor(self.intersection_volume, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.intersection_status[idx], self.intersection_volume[idx]