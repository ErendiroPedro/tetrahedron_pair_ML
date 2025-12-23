import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import base64
import os
from io import BytesIO

torch.set_default_dtype(torch.float64)

# ============================================================================
# LOSS FUNCTION
# ============================================================================


class RMSLELoss(nn.Module):
    def __init__(self, epsilon=1e-8, log_interval=100000):
        super().__init__()
        self.epsilon = epsilon
        self.log_interval = log_interval
        self.sample_count = 0
    
    def forward(self, predictions, targets):
        """
        For IntersectionVolume task:
        predictions: volume_pred shape (batch, 1) 
        targets: intersection_volume shape (batch, 1)
        """
        # Ensure all values are positive for log operation
        pred_safe = torch.clamp(predictions, min=self.epsilon)
        target_safe = torch.clamp(targets, min=self.epsilon)
        
        # Apply RMSLE: sqrt(mean((log(pred + 1) - log(target + 1))^2))
        log_pred = torch.log(pred_safe + 1.0)
        log_target = torch.log(target_safe + 1.0)
        
        squared_log_diff = (log_pred - log_target) ** 2
        msle = torch.mean(squared_log_diff)
        rmsle_loss = torch.sqrt(msle + self.epsilon)
        
        # Clean, focused logging similar to CombinedLoss
        self.sample_count += len(predictions)
        if self.sample_count % self.log_interval < len(predictions):
            self._log_metrics(predictions, targets, rmsle_loss)
        
        return rmsle_loss
    
    def _log_metrics(self, predictions, targets, loss):
        """Clean, focused logging on volume prediction metrics"""
        
        # Volume prediction metrics
        pred_std = predictions.std().item()
        pred_range = f"[{predictions.min().item():.3f}, {predictions.max().item():.3f}]"
        
        target_std = targets.std().item()
        target_range = f"[{targets.min().item():.3f}, {targets.max().item():.3f}]"
        
        print(f"📊 [Sample {self.sample_count:,}] Volume RMSLE Loss:")
        print(f"   📉 Loss: {loss.item():.6f}")
        print(f"   📈 Vol Pred - Range: {pred_range}, Std: {pred_std:.6f}")
        print(f"   📋 Vol Target - Range: {target_range}, Std: {target_std:.6f}")

class CombinedLoss(nn.Module):
    def __init__(self, regression_weight=0.5, classification_weight=0.5, epsilon=1e-8, log_interval=100000):
        super().__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.epsilon = epsilon
        self.log_interval = log_interval
        self.sample_count = 0
    
    def forward(self, predictions, targets, tetrahedron_features=None, model=None):
        # Extract components
        cls_logits = predictions[:, 0]
        vol_pred = predictions[:, 1]
        has_intersection = targets[:, 0]
        vol_target = targets[:, 1]
        
        # Compute base losses
        classification_loss = nn.functional.binary_cross_entropy_with_logits(cls_logits, has_intersection)
        regression_loss = self._compute_regression_loss(vol_pred, vol_target)
        
        
        # Combine all losses
        total_loss = (self.regression_weight * regression_loss + 
                     self.classification_weight * classification_loss)
        
        # Logging
        self.sample_count += len(predictions)
        if self.sample_count % self.log_interval < len(predictions):
            self._log_metrics(cls_logits, vol_pred, has_intersection, vol_target, 
                            classification_loss, regression_loss, total_loss)
        
        return total_loss
    
    
    def _compute_regression_loss(self, predictions, targets):
        """RMSLE loss for volume regression"""
        pred_safe = torch.clamp(predictions, min=self.epsilon)
        target_safe = torch.clamp(targets, min=self.epsilon)
        
        log_pred = torch.log(pred_safe + 1.0)
        log_target = torch.log(target_safe + 1.0)
        
        squared_log_diff = (log_pred - log_target) ** 2
        msle = torch.mean(squared_log_diff)
        
        return torch.sqrt(msle + self.epsilon)
    
    def _log_metrics(self, cls_logits, vol_pred, has_intersection, vol_target, 
                    cls_loss, reg_loss, total_loss):

        # Classification metrics
        pos_ratio = has_intersection.mean().item()
        
        # Volume prediction metrics
        vol_pred_std = vol_pred.std().item()
        vol_pred_range = f"[{vol_pred.min().item():.3f}, {vol_pred.max().item():.3f}]"
        
        vol_target_std = vol_target.std().item()
        vol_target_range = f"[{vol_target.min().item():.3f}, {vol_target.max().item():.3f}]"
        
        # Loss breakdown
        weighted_cls = self.classification_weight * cls_loss.item()
        weighted_reg = self.regression_weight * reg_loss.item()
        
        print(f"📊 [Sample {self.sample_count:,}] Loss Metrics:")
        print(f"   📉 Total: {total_loss.item():.4f} (Cls: {weighted_cls:.4f} + Reg: {weighted_reg:.4f} )")
        print(f"   🎯 Pos ratio: {pos_ratio:.3f}")
        print(f"   📈 Vol Pred - Range: {vol_pred_range}, Std: {vol_pred_std:.4f}")
        print(f"   📋 Vol Target - Range: {vol_target_range}, Std: {vol_target_std:.4f}")


# ============================================================================
# MAIN TRAINER
# ============================================================================


class CModelTrainer:
    def __init__(self, config, device=None):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Training parameters
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 1024)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        
        # Data paths
        self.processed_data_path = config.get('processed_data_path', 'data/processed')
        
        self.task_configs = {
            'IntersectionStatus': {
                'target_builder': lambda status, volume: status,
                'prediction_extractor': lambda pred: pred if pred.dim() == 1 else pred[:, 0:1]
            },
            'IntersectionVolume': {
                'target_builder': lambda status, volume: volume,
                'prediction_extractor': lambda pred: pred if pred.dim() == 1 else pred[:, -1:]
            },
            'IntersectionStatus_IntersectionVolume': {
                'target_builder': lambda status, volume: torch.cat([status, volume], dim=1),
                'prediction_extractor': lambda pred: pred  # Use all predictions
            }
        }

        
        self.LOSS_CONFIGS = {
            'IntersectionVolume': {
                'rmsle': (RMSLELoss, "RMSLE loss for volume-only prediction"),
                'default': (RMSLELoss, "RMSLE loss for volume-only prediction")
            },
            'IntersectionStatus': {
                'bce': (nn.BCEWithLogitsLoss, "BCEWithLogits loss for classification-only prediction"),
                'default': (nn.BCEWithLogitsLoss, "BCEWithLogits loss for classification-only prediction")
            },

            'IntersectionStatus_IntersectionVolume': {
                'combined': (lambda: CombinedLoss(regression_weight=0.5, classification_weight=0.5), 
                            "balanced combined loss with invariance"),
                'default': (lambda: CombinedLoss(regression_weight=0.5, classification_weight=0.5),
                        "balanced combined loss with invariance")
            }
        }

        print(f"🎯 CModelTrainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {self.epochs}, Batch size: {self.batch_size}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Data path: {self.processed_data_path}")

    def train_and_validate(self, model):
        """Main training loop - loads datasets internally"""
        print("🚀 Starting training...")
        
        # Setup model
        model = model.to(self.device)
        model = model.double()

        
        task = self.config['task']
        loss_type = self.config['loss_function']

        print(f"   🎯 Task: {task}")
        print(f"   🔧 Loss function: {loss_type}")

        # Get loss configuration for task
        task_config = self.LOSS_CONFIGS[task]

        # Get specific loss or fall back to default
        if loss_type in task_config:
            loss_class, description = task_config[loss_type]
        else:
            print(f"   ⚠️  '{loss_type}' not available for task '{task}', using default")
            loss_class, description = task_config['default']

        # Create loss function
        loss_fn = loss_class() if callable(loss_class) else loss_class
        print(f"   Using {description}")

        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-5, foreach=False)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

        # Load datasets internally
        train_loader, val_loader = self._setup_data_loaders()
        
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(1, self.epochs + 1):
            print(f"\n=== Epoch {epoch}/{self.epochs} ===")
            epoch_start = time.time()
            
            # Training phase
            model.train()
            train_loss = self._run_epoch(train_loader, model, loss_fn, optimizer, is_training=True, epoch=epoch)
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_loss = self._run_epoch(val_loader, model, loss_fn, None, is_training=False, epoch=epoch)
            
            # Check for invalid losses
            if train_loss is None or val_loss is None or np.isnan(train_loss) or np.isnan(val_loss):
                print(f"⚠️ Invalid losses detected, stopping training")
                break
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Track metrics
            epoch_time = time.time() - epoch_start
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch_time': epoch_time
            })
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Time: {epoch_time:.1f}s")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stopping_patience:
                print(f"🛑 Early stopping after {epoch} epochs")
                model.load_state_dict(best_model_state)
                break
        
        # Generate results
        report = self._create_report(training_history, best_val_loss)
        loss_curve = self._create_loss_plot(training_history)
        
        print(f"🎉 Training complete! Best validation loss: {best_val_loss:.6f}")
        return model, report, {'final_val_loss': best_val_loss}, loss_curve

    def _setup_data_loaders(self):
        train_data_path = os.path.join(self.processed_data_path, "train", "train_data.csv")
        val_data_path = os.path.join(self.processed_data_path, "val", "val_data.csv")
        
        print(f"📂 Loading datasets:")
        print(f"   Training: {train_data_path}")
        print(f"   Validation: {val_data_path}")
        
        train_dataset = TetrahedronDataset(train_data_path)
        val_dataset = TetrahedronDataset(val_data_path)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=16,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=16,
            pin_memory=True
        )
        
        return train_loader, val_loader

    def _run_epoch(self, data_loader, model, loss_fn, optimizer, is_training, epoch=1):

        total_loss = 0.0
        batch_count = 0
        
        task = self.config['task']
        
        for batch_idx, (tetrahedron_features, intersection_status, intersection_volume) in enumerate(data_loader):
            tetrahedron_features = tetrahedron_features.to(device=self.device, dtype=torch.float64)
            intersection_status = intersection_status.to(device=self.device, dtype=torch.float64)
            intersection_volume = intersection_volume.to(device=self.device, dtype=torch.float64)
            
            # Prepare targets based on task
            task_config = self.task_configs[task]
            targets = task_config['target_builder'](intersection_status, intersection_volume)
            
            # Forward pass
            predictions = model(tetrahedron_features)
            predictions = task_config['prediction_extractor'](predictions)
            
            
            loss = loss_fn(predictions, targets)
            
            if is_training:
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        return total_loss / batch_count if batch_count > 0 else None

    def _create_report(self, history, best_val_loss):
        """Create training report"""
        if not history:
            return {"error": "No training history"}
        
        total_time = sum(h['epoch_time'] for h in history)
        return {
            "total_epochs": len(history),
            "best_val_loss": best_val_loss,
            "final_val_loss": history[-1]['val_loss'],
            "total_training_time": total_time,
            "training_history": history
        }
    
    def _create_loss_plot(self, history):
        """Create loss curve visualization"""
        if not history:
            return None
        
        epochs = [h['epoch'] for h in history]
        train_losses = [h['train_loss'] for h in history]
        val_losses = [h['val_loss'] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'o-', label='Training Loss', alpha=0.8)
        plt.plot(epochs, val_losses, 's-', label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ============================================================================
# DATASET
# ============================================================================


class TetrahedronDataset(torch.utils.data.Dataset):
    def __init__(self, data_source):
        if isinstance(data_source, str):
            df = pd.read_csv(data_source)
            print(f"   Loaded {len(df):,} samples from {os.path.basename(data_source)}")
        elif isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
            print(f"   Using DataFrame with {len(df):,} samples")
        else:
            raise ValueError("data_source must be either a file path (str) or pandas DataFrame")
        
        self.tetrahedron_features = df.iloc[:, :-2].values.astype(np.float64)
        self.intersection_volume = df.iloc[:, -2].values.astype(np.float64)
        self.intersection_status = df.iloc[:, -1].values.astype(np.float64)
        

        self.tetrahedron_features = torch.tensor(self.tetrahedron_features, dtype=torch.float64)
        self.intersection_status = torch.tensor(self.intersection_status, dtype=torch.float64).reshape(-1, 1)
        self.intersection_volume = torch.tensor(self.intersection_volume, dtype=torch.float64).reshape(-1, 1)
        
        # Logs
        intersecting_count = (self.intersection_status > 0.5).sum().item()
        non_intersecting_count = len(self.intersection_status) - intersecting_count

        positive_volumes = self.intersection_volume[self.intersection_volume > 0]
        if len(positive_volumes) > 0:
            print(f"   Volume range: [{positive_volumes.min():.2e}, {positive_volumes.max():.2e}]")
        
        print(f"   Features shape: {self.tetrahedron_features.shape}")
        print(f"   Intersecting pairs: {intersecting_count:,} ({intersecting_count/len(self.intersection_status)*100:.1f}%)")
        print(f"   Non-intersecting pairs: {non_intersecting_count:,}")
        
        if intersecting_count > 0:
            intersecting_volumes = self.intersection_volume[self.intersection_status.squeeze() > 0.5]
            positive_volumes = intersecting_volumes[intersecting_volumes > 0]
            print(f"   Positive volumes: {len(positive_volumes):,}/{intersecting_count:,}")

    def __len__(self):
        return len(self.tetrahedron_features)

    def __getitem__(self, idx):
        return (
            self.tetrahedron_features[idx], 
            self.intersection_status[idx], 
            self.intersection_volume[idx]
        )