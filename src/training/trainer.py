import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import src.utils.geometry as gu
from src.utils.file_management import processed_split_path
from src.training.dataset import TetrahedronDataset
from src.training.losses import (
    CombinedLoss,
    DecadeWeightedLoss,
    LogCoshRelativeCombinedLoss,
    QuantileLogCoshRelativeCombinedLoss,
    QuantileRMSLECombinedLoss,
    RMSLELoss,
)

torch.set_default_dtype(torch.float64)

class CModelTrainer:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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

        print(f"🎯 CModelTrainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {self.epochs}, Batch size: {self.batch_size}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Data path: {self.processed_data_path}")
    
        self.LOSS_CONFIGS = {
            'IntersectionVolume': {
                'decade_weighted': (DecadeWeightedLoss, "decade-weighted loss for volume-only prediction"),
                'rmsle': (RMSLELoss, "RMSLE loss for volume-only prediction"),
                'default': (RMSLELoss, "RMSLE loss for volume-only prediction")
            },
            'IntersectionStatus': {
                'bce': (nn.BCEWithLogitsLoss, "BCEWithLogits loss for classification-only prediction"),
                'default': (nn.BCEWithLogitsLoss, "BCEWithLogits loss for classification-only prediction")
            },

            'IntersectionStatus_IntersectionVolume': {
                'quantile_rmsle_aggressive': (lambda: QuantileRMSLECombinedLoss(regression_weight=0.80, classification_weight=0.10, invariance_weight=0.10, tau=0.7),
                                    "Quantile RMSLE aggressive volume-focused loss"),
                'quantile_logcosh_aggressive': (lambda: QuantileLogCoshRelativeCombinedLoss(regression_weight=0.70, classification_weight=0.15, invariance_weight=15, tau=0.7),
                                      "Quantile LogCosh aggressive volume-focused loss (penalizes underpredictions more)"),
                'quantile_logcosh_strong_underpred': (lambda: QuantileLogCoshRelativeCombinedLoss(regression_weight=0.85, classification_weight=0.075, invariance_weight=0.075, tau=0.8),
                                            "Strong underprediction penalty (80% weight on underpredictions)"),
                'logcosh_aggressive': (lambda: LogCoshRelativeCombinedLoss(regression_weight=0.80, classification_weight=0.10, invariance_weight=0.10),
                             "LogCosh Relative aggressive volume-focused loss"),
                'volume_focused': (lambda: CombinedLoss(regression_weight=0.3333, classification_weight=0.3333, invariance_weight=0.3333),
                                "volume-focused combined loss with invariance"),
                'volume_focused_strong_invariance': (lambda: CombinedLoss(regression_weight=1.0, classification_weight=0.5, invariance_weight=0.4),
                                                "volume-focused with strong invariance enforcement"),
                'combined': (lambda: CombinedLoss(regression_weight=0.5, classification_weight=0.5, invariance_weight=0), 
                            "balanced combined loss with invariance"),
                'default': (lambda: CombinedLoss(regression_weight=0.5, classification_weight=0.5, invariance_weight=0.1),
                        "balanced combined loss with invariance")
            }

        }
    def train_and_validate(self, model):
        """Main training loop - loads datasets internally"""
        print("🚀 Starting training...")
        
        # Setup model and training components - ensure proper device and dtype
        # Move to device first, then convert to double to ensure all parameters are properly set
        model = model.to(self.device)
        model = model.double()

        
        # === TASK AND LOSS FUNCTION SELECTION ===
        task = self.config.get('task', 'IntersectionStatus_IntersectionVolume')
        loss_type = self.config.get('loss_function', 'combined')

        print(f"   🎯 Task: {task}")
        print(f"   🔧 Loss function: {loss_type}")

        # Get loss configuration for task
        task_config = self.LOSS_CONFIGS.get(task, self.LOSS_CONFIGS['IntersectionStatus_IntersectionVolume'])

        # Get specific loss or fall back to default
        if loss_type in task_config:
            loss_class, description = task_config[loss_type]
        else:
            print(f"   ⚠️  '{loss_type}' not available for task '{task}', using default")
            loss_class, description = task_config['default']

        # Create loss function
        loss_fn = loss_class() if callable(loss_class) else loss_class
        print(f"   Using {description}")

        # === OPTIMIZER AND SCHEDULER SETUP ===
        # Use foreach=False to avoid PyTorch 2.1 multi-tensor AdamW issues with float64
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-5, foreach=False)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

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

    def _test_model_after_init(self, model):
        """Test model with dummy input to catch issues early"""
        print("🧪 Testing model after initialization...")
        
        # Create dummy input
        dummy_input = torch.randn(4, 24, dtype=torch.float64).to(self.device)
        
        try:
            model.eval()
            with torch.no_grad():
                # Remove the debug temporarily for this test
                original_forward = model.forward
                
                # Create a forward method without debug
                def test_forward(x):
                    features = model._extract_features(x)
                    shared_features = model.shared_layers(features)
                    
                    if model.task == 'IntersectionStatus_IntersectionVolume':
                        intersection_status_logits = model.classification_head(shared_features)
                        regression_raw = model.regression_head(shared_features)
                        regression_output = torch.relu(regression_raw)
                        return torch.cat([intersection_status_logits, regression_output], dim=1)
                    else:
                        return model.classification_head(shared_features)
                
                # Temporarily replace forward
                model.forward = test_forward
                
                # Test
                output = model(dummy_input)
                print(f"   Test output shape: {output.shape}")
                print(f"   Test output range: [{output.min():.6f}, {output.max():.6f}]")
                
                # Check for issues
                if torch.isnan(output).any():
                    print("   🚨 Model produces NaN outputs!")
                elif torch.allclose(output[0], output[1]):
                    print("   🚨 Model produces identical outputs for different inputs!")
                else:
                    print("   ✅ Model test passed")
                
                # Restore original forward
                model.forward = original_forward
                
            model.train()
            
        except Exception as e:
            print(f"   🚨 Model test failed: {e}")

    def _setup_data_loaders(self):
        """Setup training and validation data loaders from CSV files"""
        train_data_path = processed_split_path(self.processed_data_path, "train")
        val_data_path = processed_split_path(self.processed_data_path, "val")
        
        print(f"📂 Loading datasets:")
        print(f"   Training: {train_data_path}")
        print(f"   Validation: {val_data_path}")
        
        # Create datasets WITHOUT normalization
        train_dataset = TetrahedronDataset(train_data_path)
        val_dataset = TetrahedronDataset(val_data_path)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader

    def _debug_model_architecture(self, model, sample_batch):
        """Debug what's happening inside the model"""
        print("🔧 MODEL ARCHITECTURE DEBUG:")
        
        features, status, volume = sample_batch
        features = features[:4].to(self.device)  # Small batch
        
        model.eval()
        with torch.no_grad():
            print(f"   Input shape: {features.shape}")
            print(f"   Input range: [{features.min():.3f}, {features.max():.3f}]")
            
            # Step through the model manually
            # 1. Feature extraction
            if hasattr(model, '_extract_features'):
                extracted_features = model._extract_features(features)
                print(f"   Extracted features shape: {extracted_features.shape}")
                print(f"   Extracted features range: [{extracted_features.min():.3f}, {extracted_features.max():.3f}]")
                print(f"   Extracted features std: {extracted_features.std():.6f}")
                
                # 2. Shared layers
                if hasattr(model, 'shared_layers'):
                    shared_output = model.shared_layers(extracted_features)
                    print(f"   Shared output shape: {shared_output.shape}")
                    print(f"   Shared output range: [{shared_output.min():.3f}, {shared_output.max():.3f}]")
                    print(f"   Shared output std: {shared_output.std():.6f}")
                    
                    # 3. Task heads
                    if hasattr(model, 'classification_head'):
                        cls_output = model.classification_head(shared_output)
                        print(f"   Classification head output: {cls_output.flatten().tolist()}")
                    
                    if hasattr(model, 'regression_head'):
                        reg_raw = model.regression_head(shared_output)
                        reg_sigmoid = torch.sigmoid(reg_raw)
                        reg_scaled = reg_sigmoid * model.volume_scale_factor
                        
                        print(f"   Regression raw: {reg_raw.flatten().tolist()}")
                        print(f"   Regression sigmoid: {reg_sigmoid.flatten().tolist()}")
                        print(f"   Volume scale factor: {model.volume_scale_factor}")
                        print(f"   Regression final: {reg_scaled.flatten().tolist()}")
            
            # Final model output
            final_output = model(features)
            print(f"   Final output shape: {final_output.shape}")
            print(f"   Final output: {final_output.tolist()}")
        
        model.train()

    def _run_epoch(self, data_loader, model, loss_fn, optimizer, is_training, epoch=1):
        """Run one epoch with invariance loss support"""
        if hasattr(loss_fn, 'set_epoch'):
            loss_fn.set_epoch(epoch)

        total_loss = 0.0
        batch_count = 0
        
        # Get task from config
        task = self.config.get('task', 'IntersectionStatus_IntersectionVolume')
        
        for batch_idx, (tetrahedron_features, intersection_status, intersection_volume) in enumerate(data_loader):
            # Move to device AND ensure float64 to match model dtype
            tetrahedron_features = tetrahedron_features.to(device=self.device, dtype=torch.float64)
            intersection_status = intersection_status.to(device=self.device, dtype=torch.float64)
            intersection_volume = intersection_volume.to(device=self.device, dtype=torch.float64)
            
            # Prepare targets based on task
            task_config = self.task_configs[task]
            targets = task_config['target_builder'](intersection_status, intersection_volume)
            
            # Forward pass
            predictions = model(tetrahedron_features)
            predictions = task_config['prediction_extractor'](predictions)
            
            # Compute loss - pass additional parameters for invariance loss
            if hasattr(loss_fn, 'invariance_weight') and loss_fn.invariance_weight > 0:
                # Loss function supports invariance - pass tetrahedron_features and model
                loss = loss_fn(predictions, targets, tetrahedron_features, model)
            else:
                # Standard loss function
                loss = loss_fn(predictions, targets)
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() < 0:
                print(f"⚠️ Invalid loss at batch {batch_idx}: {loss.item()}, skipping")
                continue
            
            if is_training:
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (more aggressive due to invariance loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Enhanced debug for first batch
            if batch_idx == 0 and epoch <= 2:
                print(f"🔍 Epoch {epoch} - Invariance loss debug:")
                if hasattr(loss_fn, 'invariance_weight'):
                    print(f"   Invariance weight: {loss_fn.invariance_weight}")
                    # Test invariance manually
                    with torch.no_grad():
                        self._test_invariance(model, tetrahedron_features[:4])
        
        return total_loss / batch_count if batch_count > 0 else None

    def _test_invariance(self, model, sample_features):
        """Test invariance on a small sample"""
        
        model.eval()
        original_pred = model(sample_features)
        swapped_features = gu.swap_tetrahedrons(sample_features)
        swapped_pred = model(swapped_features)
        
        diff = torch.abs(original_pred - swapped_pred).mean().item()
        print(f"   Mean prediction difference after swap: {diff:.6f}")
        
        model.train()


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
