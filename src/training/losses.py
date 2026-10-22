import torch
import torch.nn as nn


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
