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

class LogMAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.sample_count = 0
    
    def forward(self, predictions, targets):
        self.sample_count += len(predictions)
        should_log = (self.sample_count % 100000) < len(predictions)
        
        # Clamp to avoid division by zero
        pred_safe = torch.clamp(predictions, min=self.epsilon)
        target_safe = torch.clamp(targets, min=self.epsilon)
        
        # Log MAPE: mean(|log(pred) - log(target)| / |log(target)|)
        log_pred = torch.log(pred_safe)
        log_target = torch.log(target_safe)
        
        log_mape = torch.mean(torch.abs(log_pred - log_target) / torch.abs(log_target))
        
        if should_log:
            print(f"      Log MAPE: {log_mape.item():.6f}")
        
        return log_mape

class VariancePenalizedLoss(nn.Module):
    def __init__(self, base_loss_fn, variance_weight=0.1, min_std=1e-3):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.variance_weight = variance_weight
        self.min_std = min_std
        self.sample_count = 0
        
    def forward(self, predictions, targets):
        base_loss = self.base_loss_fn(predictions, targets)
        
        pred_std = torch.std(predictions)
        
        # INVERSE penalty - gets HUGE as std approaches 0
        if pred_std < self.min_std:
            diversity_penalty = self.variance_weight / (pred_std + 1e-8)  # 1/(small number) = big number
        else:
            diversity_penalty = torch.tensor(0.0, device=predictions.device)
        
        total_loss = base_loss + diversity_penalty
        
        self.sample_count += len(predictions)
        should_log = (self.sample_count % 100000) < len(predictions)
        if should_log:
            print(f"      Base loss: {base_loss.item():.6f}, Diversity penalty: {diversity_penalty.item():.6f}")
            print(f"      Penalty calc: {self.variance_weight} / ({pred_std.item():.6f} + 1e-8) = {diversity_penalty.item():.6f}")
            
            if pred_std < self.min_std:
                print(f"      🚨 LOW STD - INVERSE PENALTY: {diversity_penalty.item():.6f}")
                
        return total_loss

class DecadeWeightedLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.sample_count = 0
        
    def forward(self, predictions, targets):
        self.sample_count += len(predictions)
        should_log = (self.sample_count % 100000) < len(predictions)
        
        # Clamp to avoid log(0)
        pred_safe = torch.clamp(predictions, min=self.epsilon)
        target_safe = torch.clamp(targets, min=self.epsilon)
        
        # Calculate which decade each target belongs to
        # For range [1e-7, 0.01], decades are: 1e-7 to 1e-6, 1e-6 to 1e-5, etc.
        log_targets = torch.log10(target_safe)
        decade_indices = torch.floor(log_targets).long()  # -7, -6, -5, -4, -3, -2
        
        # Calculate squared errors
        squared_errors = (pred_safe - target_safe) ** 2
        
        # Weight by inverse of target magnitude to make loss log-uniform
        # Smaller values get higher weights so model pays equal attention
        log_weights = -log_targets  # Higher weight for smaller values
        decade_weights = torch.pow(10.0, log_weights * 0.5)  # Moderate the weighting
        
        # Apply weights
        weighted_errors = squared_errors * decade_weights
        
        loss = torch.mean(weighted_errors)
        
        if should_log:
            unique_decades = torch.unique(decade_indices)
            print(f"      [Sample {self.sample_count:,}] Decade-Weighted Loss: {loss.item():.6f}")
            print(f"      Active decades: {unique_decades.tolist()}")
            print(f"      Prediction range: [{pred_safe.min():.2e}, {pred_safe.max():.2e}]")
            print(f"      Target range: [{targets.min():.2e}, {targets.max():.2e}]")
            print(f"      Weight range: [{decade_weights.min():.2f}, {decade_weights.max():.2f}]")
        
        return loss

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
    def __init__(self, regression_weight=0.5, classification_weight=0.5, 
                 invariance_weight=0.1, epsilon=1e-8, log_interval=100000):
        super().__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.invariance_weight = invariance_weight
        self.epsilon = epsilon
        self.log_interval = log_interval
        self.sample_count = 0
    
    def forward(self, predictions, targets, tetrahedron_features=None, model=None):
        """
        predictions: [intersection_logits, volume_pred] shape (batch, 2)
        targets: [has_intersection, intersection_volume] shape (batch, 2)
        tetrahedron_features: original input for invariance loss
        model: model reference for invariance computation
        """
        # Extract components
        cls_logits = predictions[:, 0]
        vol_pred = predictions[:, 1]
        has_intersection = targets[:, 0]
        vol_target = targets[:, 1]
        
        # Compute base losses
        classification_loss = nn.functional.binary_cross_entropy_with_logits(cls_logits, has_intersection)
        regression_loss = self._compute_regression_loss(vol_pred, vol_target)
        
        # Compute invariance loss if inputs provided
        invariance_loss = torch.tensor(0.0, device=predictions.device)
        if tetrahedron_features is not None and model is not None and self.invariance_weight > 0:
            invariance_loss = self._compute_invariance_loss(model, tetrahedron_features)
        
        # Combine all losses
        total_loss = (self.regression_weight * regression_loss + 
                     self.classification_weight * classification_loss +
                     self.invariance_weight * invariance_loss)
        
        # Logging
        self.sample_count += len(predictions)
        if self.sample_count % self.log_interval < len(predictions):
            self._log_metrics(cls_logits, vol_pred, has_intersection, vol_target, 
                            classification_loss, regression_loss, invariance_loss, total_loss)
        
        return total_loss
    
    def _compute_invariance_loss(self, model, tetrahedron_features):
        """Compute invariance loss using GeometryUtils"""
        import GeometryUtils as gu
        
        # Get predictions for original input
        original_pred = model(tetrahedron_features)
        
        # Swap tetrahedrons
        swapped_features = gu.swap_tetrahedrons(tetrahedron_features)
        
        # Get predictions for swapped input
        swapped_pred = model(swapped_features)
        
        return self._compute_regression_loss(swapped_pred, original_pred)
    
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
                    cls_loss, reg_loss, inv_loss, total_loss):
        """Enhanced logging with invariance loss"""
        
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
        weighted_inv = self.invariance_weight * inv_loss.item()
        
        print(f"📊 [Sample {self.sample_count:,}] Loss Metrics:")
        print(f"   📉 Total: {total_loss.item():.4f} (Cls: {weighted_cls:.4f} + Reg: {weighted_reg:.4f} + Inv: {weighted_inv:.4f})")
        print(f"   🎯 Pos ratio: {pos_ratio:.3f}")
        print(f"   📈 Vol Pred - Range: {vol_pred_range}, Std: {vol_pred_std:.4f}")
        print(f"   📋 Vol Target - Range: {vol_target_range}, Std: {vol_target_std:.4f}")
        print(f"   🔄 Invariance Loss: {inv_loss.item():.6f}")



class LogCoshRelativeLoss(nn.Module):
    def __init__(self, epsilon=1e-6, log_interval=100000):
        super().__init__()
        self.epsilon = epsilon
        self.log_interval = log_interval
        self.sample_count = 0
    
    def forward(self, predictions, targets):
        """
        Custom loss: Log-Cosh with relative scaling
        Loss = 1/N * sum(log(cosh((y_i - ŷ_i) / max(|y_i|, |ŷ_i|, ε))))
        
        Args:
            predictions: model predictions (no clamping - let gradients flow naturally)
            targets: ground truth values
        """
        # Calculate relative error denominator: max(|y_i|, |ŷ_i|, ε)
        abs_pred = torch.abs(predictions)
        abs_target = torch.abs(targets)
        max_magnitude = torch.maximum(
            torch.maximum(abs_pred, abs_target),
            torch.tensor(self.epsilon, device=predictions.device)
        )
        
        # Calculate relative error: (y_i - ŷ_i) / max(|y_i|, |ŷ_i|, ε)
        relative_error = (targets - predictions) / max_magnitude
        
        # Apply log(cosh(x)) - smooth, differentiable, less sensitive to outliers
        log_cosh_loss = torch.log(torch.cosh(relative_error))
        
        # Mean over batch
        loss = torch.mean(log_cosh_loss)
        
        # Logging
        self.sample_count += len(predictions)
        if self.sample_count % self.log_interval < len(predictions):
            self._log_metrics(predictions, targets, relative_error, loss)
        
        return loss
    
    def _log_metrics(self, predictions, targets, relative_error, loss):
        """Enhanced logging for the custom loss"""
        
        # Volume prediction metrics
        pred_std = predictions.std().item()
        pred_range = f"[{predictions.min().item():.3f}, {predictions.max().item():.3f}]"
        
        target_std = targets.std().item()
        target_range = f"[{targets.min().item():.3f}, {targets.max().item():.3f}]"
        
        # Relative error statistics
        rel_err_mean = relative_error.mean().item()
        rel_err_std = relative_error.std().item()
        rel_err_range = f"[{relative_error.min().item():.4f}, {relative_error.max().item():.4f}]"
        
        print(f"📊 [Sample {self.sample_count:,}] LogCosh Relative Loss:")
        print(f"   📉 Loss: {loss.item():.6f}")
        print(f"   📈 Vol Pred - Range: {pred_range}, Std: {pred_std:.6f}")
        print(f"   📋 Vol Target - Range: {target_range}, Std: {target_std:.6f}")
        print(f"   📏 Relative Error - Mean: {rel_err_mean:.4f}, Std: {rel_err_std:.4f}, Range: {rel_err_range}")


class LogCoshRelativeCombinedLoss(nn.Module):
    def __init__(self, regression_weight=0.5, classification_weight=0.5, 
                 invariance_weight=0.1, epsilon=1e-6, log_interval=100000):
        super().__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.invariance_weight = invariance_weight
        self.epsilon = epsilon
        self.log_interval = log_interval
        self.sample_count = 0
    
    def forward(self, predictions, targets, tetrahedron_features=None, model=None):
        """
        Combined loss using LogCosh Relative for regression
        """
        # Extract components
        cls_logits = predictions[:, 0]
        vol_pred = predictions[:, 1]
        has_intersection = targets[:, 0]
        vol_target = targets[:, 1]
        
        # Compute base losses
        classification_loss = nn.functional.binary_cross_entropy_with_logits(cls_logits, has_intersection)
        regression_loss = self._compute_regression_loss(vol_pred, vol_target)
        
        # Compute invariance loss if inputs provided
        invariance_loss = torch.tensor(0.0, device=predictions.device)
        if tetrahedron_features is not None and model is not None and self.invariance_weight > 0:
            invariance_loss = self._compute_invariance_loss(model, tetrahedron_features)
        
        # Combine all losses
        total_loss = (self.regression_weight * regression_loss + 
                     self.classification_weight * classification_loss +
                     self.invariance_weight * invariance_loss)
        
        # Logging
        self.sample_count += len(predictions)
        if self.sample_count % self.log_interval < len(predictions):
            self._log_metrics(cls_logits, vol_pred, has_intersection, vol_target, 
                            classification_loss, regression_loss, invariance_loss, total_loss)
        
        return total_loss
    
    def _compute_invariance_loss(self, model, tetrahedron_features):
        """Compute invariance loss using LogCosh Relative"""
        import GeometryUtils as gu
        
        # Get predictions for original input
        original_pred = model(tetrahedron_features)
        
        # Swap tetrahedrons
        swapped_features = gu.swap_tetrahedrons(tetrahedron_features)
        
        # Get predictions for swapped input
        swapped_pred = model(swapped_features)
        
        # Use LogCosh Relative for consistency (same as volume regression)
        return self._compute_regression_loss(swapped_pred, original_pred)
    
    def _compute_regression_loss(self, predictions, targets):
        """LogCosh Relative loss for volume regression - no clamping for natural gradients"""
        
        # Calculate relative error denominator: max(|y_i|, |ŷ_i|, ε)
        abs_pred = torch.abs(predictions)
        abs_target = torch.abs(targets)
        max_magnitude = torch.maximum(
            torch.maximum(abs_pred, abs_target),
            torch.tensor(self.epsilon, device=predictions.device)
        )
        
        # Calculate relative error - let gradients flow naturally
        relative_error = (targets - predictions) / max_magnitude
        
        # Apply log(cosh(x)) - smooth and differentiable
        log_cosh_loss = torch.log(torch.cosh(relative_error))
        
        return torch.mean(log_cosh_loss)
    
    def _log_metrics(self, cls_logits, vol_pred, has_intersection, vol_target, 
                    cls_loss, reg_loss, inv_loss, total_loss):
        """Enhanced logging with LogCosh Relative loss"""
        
        # Classification metrics
        pos_ratio = has_intersection.mean().item()
        
        # Volume prediction metrics (no clamping in display either)
        vol_pred_std = vol_pred.std().item()
        vol_pred_range = f"[{vol_pred.min().item():.3f}, {vol_pred.max().item():.3f}]"
        
        vol_target_std = vol_target.std().item()
        vol_target_range = f"[{vol_target.min().item():.3f}, {vol_target.max().item():.3f}]"
        
        # Loss breakdown
        weighted_cls = self.classification_weight * cls_loss.item()
        weighted_reg = self.regression_weight * reg_loss.item()
        weighted_inv = self.invariance_weight * inv_loss.item()
        
        print(f"📊 [Sample {self.sample_count:,}] LogCosh Combined Loss:")
        print(f"   📉 Total: {total_loss.item():.4f} (Cls: {weighted_cls:.4f} + Reg: {weighted_reg:.4f} + Inv: {weighted_inv:.4f})")
        print(f"   🎯 Pos ratio: {pos_ratio:.3f}")
        print(f"   📈 Vol Pred - Range: {vol_pred_range}, Std: {vol_pred_std:.4f}")
        print(f"   📋 Vol Target - Range: {vol_target_range}, Std: {vol_target_std:.4f}")
        print(f"   🔄 Invariance Loss: {inv_loss.item():.6f}")

class QuantileLogCoshRelativeLoss(nn.Module):
    def __init__(self, tau=0.7, epsilon=1e-6, log_interval=100000):
        """
        Quantile-inspired LogCosh with relative scaling
        
        Args:
            tau: Quantile parameter (0.5 = balanced, >0.5 penalizes underpredictions more)
                 tau=0.7 means 70% penalty weight on underpredictions, 30% on overpredictions
            epsilon: Small value for numerical stability
            log_interval: Logging frequency
        """
        super().__init__()
        self.tau = tau
        self.epsilon = epsilon
        self.log_interval = log_interval
        self.sample_count = 0
    
    def forward(self, predictions, targets):
        """
        Quantile LogCosh loss: asymmetric penalty based on prediction direction
        """
        # Calculate relative error denominator: max(|y_i|, |ŷ_i|, ε)
        abs_pred = torch.abs(predictions)
        abs_target = torch.abs(targets)
        max_magnitude = torch.maximum(
            torch.maximum(abs_pred, abs_target),
            torch.tensor(self.epsilon, device=predictions.device)
        )
        
        # Calculate relative error: (y_i - ŷ_i) / max(|y_i|, |ŷ_i|, ε)
        relative_error = (targets - predictions) / max_magnitude
        
        # Apply quantile weighting:
        # - When relative_error > 0 (underprediction): weight = tau
        # - When relative_error < 0 (overprediction): weight = (1 - tau)
        quantile_weights = torch.where(
            relative_error >= 0,
            torch.tensor(self.tau, device=predictions.device),      # Underprediction penalty
            torch.tensor(1.0 - self.tau, device=predictions.device) # Overprediction penalty
        )
        
        # Apply log(cosh(x)) with quantile weighting
        log_cosh_values = torch.log(torch.cosh(relative_error))
        weighted_loss = quantile_weights * log_cosh_values
        
        # Mean over batch
        loss = torch.mean(weighted_loss)
        
        # Logging
        self.sample_count += len(predictions)
        if self.sample_count % self.log_interval < len(predictions):
            self._log_metrics(predictions, targets, relative_error, quantile_weights, loss)
        
        return loss
    
    def _log_metrics(self, predictions, targets, relative_error, weights, loss):
        """Enhanced logging for quantile LogCosh loss"""
        
        # Volume prediction metrics
        pred_std = predictions.std().item()
        pred_range = f"[{predictions.min().item():.3f}, {predictions.max().item():.3f}]"
        
        target_std = targets.std().item()
        target_range = f"[{targets.min().item():.3f}, {targets.max().item():.3f}]"
        
        # Relative error and quantile statistics
        rel_err_mean = relative_error.mean().item()
        rel_err_std = relative_error.std().item()
        
        # Count underpredictions vs overpredictions
        underpred_count = (relative_error >= 0).sum().item()
        overpred_count = (relative_error < 0).sum().item()
        total_samples = len(relative_error)
        
        # Average weights applied
        avg_weight = weights.mean().item()
        
        print(f"📊 [Sample {self.sample_count:,}] Quantile LogCosh Loss (τ={self.tau}):")
        print(f"   📉 Loss: {loss.item():.6f}")
        print(f"   📈 Vol Pred - Range: {pred_range}, Std: {pred_std:.6f}")
        print(f"   📋 Vol Target - Range: {target_range}, Std: {target_std:.6f}")
        print(f"   📏 Relative Error - Mean: {rel_err_mean:.4f}, Std: {rel_err_std:.4f}")
        print(f"   ⚖️  Predictions: Under={underpred_count}/{total_samples} ({underpred_count/total_samples*100:.1f}%), Over={overpred_count}/{total_samples} ({overpred_count/total_samples*100:.1f}%)")
        print(f"   🎯 Avg Weight: {avg_weight:.3f} (Under penalty: {self.tau:.1f}, Over penalty: {1-self.tau:.1f})")


class QuantileLogCoshRelativeCombinedLoss(nn.Module):
    def __init__(self, regression_weight=0.5, classification_weight=0.4, 
                 invariance_weight=0.1, tau=0.5, epsilon=1e-6, log_interval=100000):
        """
        Combined loss using Quantile LogCosh Relative for regression
        
        Args:
            tau: Quantile parameter for underprediction penalty (>0.5 penalizes underpredictions more)
        """
        super().__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.invariance_weight = invariance_weight
        self.tau = tau
        self.epsilon = epsilon
        self.log_interval = log_interval
        self.sample_count = 0
    
    def forward(self, predictions, targets, tetrahedron_features=None, model=None):
        """
        Combined loss using Quantile LogCosh Relative for regression
        """
        # Extract components
        cls_logits = predictions[:, 0]
        vol_pred = predictions[:, 1]
        has_intersection = targets[:, 0]
        vol_target = targets[:, 1]
        
        # Compute base losses
        classification_loss = nn.functional.binary_cross_entropy_with_logits(cls_logits, has_intersection)
        regression_loss = self._compute_regression_loss(vol_pred, vol_target)
        
        # Compute invariance loss if inputs provided
        invariance_loss = torch.tensor(0.0, device=predictions.device)
        if tetrahedron_features is not None and model is not None and self.invariance_weight > 0:
            invariance_loss = self._compute_invariance_loss(model, tetrahedron_features)
        
        # Combine all losses
        total_loss = (self.regression_weight * regression_loss + 
                     self.classification_weight * classification_loss +
                     self.invariance_weight * invariance_loss)
        
        # Logging
        self.sample_count += len(predictions)
        if self.sample_count % self.log_interval < len(predictions):
            self._log_metrics(cls_logits, vol_pred, has_intersection, vol_target, 
                            classification_loss, regression_loss, invariance_loss, total_loss)
        
        return total_loss
    
    def _compute_invariance_loss(self, model, tetrahedron_features):
        """Compute invariance loss using Quantile LogCosh Relative"""
        import GeometryUtils as gu
        
        # Get predictions for original input
        original_pred = model(tetrahedron_features)
        
        # Swap tetrahedrons
        swapped_features = gu.swap_tetrahedrons(tetrahedron_features)
        
        # Get predictions for swapped input
        swapped_pred = model(swapped_features)
        
        # Use Quantile LogCosh Relative for consistency (same as volume regression)
        return self._compute_regression_loss(swapped_pred, original_pred)
    
    def _compute_regression_loss(self, predictions, targets):
        """Quantile LogCosh Relative loss for volume regression"""
        
        # Calculate relative error denominator: max(|y_i|, |ŷ_i|, ε)
        abs_pred = torch.abs(predictions)
        abs_target = torch.abs(targets)
        max_magnitude = torch.maximum(
            torch.maximum(abs_pred, abs_target),
            torch.tensor(self.epsilon, device=predictions.device)
        )
        
        # Calculate relative error
        relative_error = (targets - predictions) / max_magnitude
        
        # Apply quantile weighting for asymmetric penalty
        quantile_weights = torch.where(
            relative_error >= 0,
            torch.tensor(self.tau, device=predictions.device),      # Underprediction penalty
            torch.tensor(1.0 - self.tau, device=predictions.device) # Overprediction penalty
        )
        
        # Apply log(cosh(x)) with quantile weighting
        log_cosh_values = torch.log(torch.cosh(relative_error))
        weighted_loss = quantile_weights * log_cosh_values
        
        return torch.mean(weighted_loss)
    
    def _log_metrics(self, cls_logits, vol_pred, has_intersection, vol_target, 
                    cls_loss, reg_loss, inv_loss, total_loss):
        """Enhanced logging with Quantile LogCosh loss"""
        
        # Classification metrics
        pos_ratio = has_intersection.mean().item()
        
        # Volume prediction metrics
        vol_pred_std = vol_pred.std().item()
        vol_pred_range = f"[{vol_pred.min().item():.3f}, {vol_pred.max().item():.3f}]"
        
        vol_target_std = vol_target.std().item()
        vol_target_range = f"[{vol_target.min().item():.3f}, {vol_target.max().item():.3f}]"
        
        # Prediction direction analysis
        relative_error = (vol_target - vol_pred) / torch.maximum(
            torch.maximum(torch.abs(vol_pred), torch.abs(vol_target)),
            torch.tensor(self.epsilon, device=vol_pred.device)
        )
        underpred_pct = (relative_error >= 0).float().mean().item() * 100
        
        # Loss breakdown
        weighted_cls = self.classification_weight * cls_loss.item()
        weighted_reg = self.regression_weight * reg_loss.item()
        weighted_inv = self.invariance_weight * inv_loss.item()
        
        print(f"📊 [Sample {self.sample_count:,}] Quantile LogCosh Combined Loss (τ={self.tau}):")
        print(f"   📉 Total: {total_loss.item():.4f} (Cls: {weighted_cls:.4f} + Reg: {weighted_reg:.4f} + Inv: {weighted_inv:.4f})")
        print(f"   🎯 Pos ratio: {pos_ratio:.3f}")
        print(f"   📈 Vol Pred - Range: {vol_pred_range}, Std: {vol_pred_std:.4f}")
        print(f"   📋 Vol Target - Range: {vol_target_range}, Std: {vol_target_std:.4f}")
        print(f"   ⚖️  Underpredictions: {underpred_pct:.1f}% (penalty weight: {self.tau:.1f})")
        print(f"   🔄 Invariance Loss: {inv_loss.item():.6f}")

class QuantileRMSLELoss(nn.Module):
    def __init__(self, tau=0.7, epsilon=1e-8, log_interval=100000):
        """
        Quantile-weighted RMSLE loss that penalizes underpredictions more
        
        Args:
            tau: Quantile parameter (>0.5 penalizes underpredictions more)
            epsilon: Small value for numerical stability
            log_interval: Logging frequency
        """
        super().__init__()
        self.tau = tau
        self.epsilon = epsilon
        self.log_interval = log_interval
        self.sample_count = 0
    
    def forward(self, predictions, targets):
        """
        Quantile RMSLE: RMSLE with asymmetric weighting based on prediction direction
        """
        # Ensure positive values for log operation
        pred_safe = torch.clamp(predictions, min=self.epsilon)
        target_safe = torch.clamp(targets, min=self.epsilon)
        
        # Calculate log differences
        log_pred = torch.log(pred_safe + 1.0)
        log_target = torch.log(target_safe + 1.0)
        log_diff = log_target - log_pred
        
        # Apply quantile weighting based on prediction direction
        # Positive log_diff = underprediction (pred < target)
        # Negative log_diff = overprediction (pred > target)
        quantile_weights = torch.where(
            log_diff >= 0,
            torch.tensor(self.tau, device=predictions.device),      # Underprediction penalty
            torch.tensor(1.0 - self.tau, device=predictions.device) # Overprediction penalty
        )
        
        # Apply quantile weighting to squared log differences
        squared_log_diff = log_diff ** 2
        weighted_squared_log_diff = quantile_weights * squared_log_diff
        
        # Compute weighted MSLE and take square root
        weighted_msle = torch.mean(weighted_squared_log_diff)
        quantile_rmsle = torch.sqrt(weighted_msle + self.epsilon)
        
        # Logging
        self.sample_count += len(predictions)
        if self.sample_count % self.log_interval < len(predictions):
            self._log_metrics(predictions, targets, log_diff, quantile_weights, quantile_rmsle)
        
        return quantile_rmsle
    
    def _log_metrics(self, predictions, targets, log_diff, weights, loss):
        """Enhanced logging for quantile RMSLE loss"""
        
        # Volume prediction metrics
        pred_std = predictions.std().item()
        pred_range = f"[{predictions.min().item():.3f}, {predictions.max().item():.3f}]"
        
        target_std = targets.std().item()
        target_range = f"[{targets.min().item():.3f}, {targets.max().item():.3f}]"
        
        # Log difference and quantile statistics
        log_diff_mean = log_diff.mean().item()
        log_diff_std = log_diff.std().item()
        
        # Count underpredictions vs overpredictions
        underpred_count = (log_diff >= 0).sum().item()
        overpred_count = (log_diff < 0).sum().item()
        total_samples = len(log_diff)
        
        # Average weights applied
        avg_weight = weights.mean().item()
        
        print(f"📊 [Sample {self.sample_count:,}] Quantile RMSLE Loss (τ={self.tau}):")
        print(f"   📉 Loss: {loss.item():.6f}")
        print(f"   📈 Vol Pred - Range: {pred_range}, Std: {pred_std:.6f}")
        print(f"   📋 Vol Target - Range: {target_range}, Std: {target_std:.6f}")
        print(f"   📏 Log Diff - Mean: {log_diff_mean:.4f}, Std: {log_diff_std:.4f}")
        print(f"   ⚖️  Predictions: Under={underpred_count}/{total_samples} ({underpred_count/total_samples*100:.1f}%), Over={overpred_count}/{total_samples} ({overpred_count/total_samples*100:.1f}%)")
        print(f"   🎯 Avg Weight: {avg_weight:.3f} (Under penalty: {self.tau:.1f}, Over penalty: {1-self.tau:.1f})")


class QuantileRMSLECombinedLoss(nn.Module):
    def __init__(self, regression_weight=0.5, classification_weight=0.4, 
                 invariance_weight=0.1, tau=0.7, epsilon=1e-8, log_interval=100000):
        """
        Combined loss using Quantile RMSLE for regression
        
        Args:
            tau: Quantile parameter for underprediction penalty (>0.5 penalizes underpredictions more)
        """
        super().__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.invariance_weight = invariance_weight
        self.tau = tau
        self.epsilon = epsilon
        self.log_interval = log_interval
        self.sample_count = 0
    
    def forward(self, predictions, targets, tetrahedron_features=None, model=None):
        """
        Combined loss using Quantile RMSLE for regression
        """
        # Extract components
        cls_logits = predictions[:, 0]
        vol_pred = predictions[:, 1]
        has_intersection = targets[:, 0]
        vol_target = targets[:, 1]
        
        # Compute base losses
        classification_loss = nn.functional.binary_cross_entropy_with_logits(cls_logits, has_intersection)
        regression_loss = self._compute_regression_loss(vol_pred, vol_target)
        
        # Compute invariance loss if inputs provided
        invariance_loss = torch.tensor(0.0, device=predictions.device)
        if tetrahedron_features is not None and model is not None and self.invariance_weight > 0:
            invariance_loss = self._compute_invariance_loss(model, tetrahedron_features)
        
        # Combine all losses
        total_loss = (self.regression_weight * regression_loss + 
                     self.classification_weight * classification_loss +
                     self.invariance_weight * invariance_loss)
        
        # Logging
        self.sample_count += len(predictions)
        if self.sample_count % self.log_interval < len(predictions):
            self._log_metrics(cls_logits, vol_pred, has_intersection, vol_target, 
                            classification_loss, regression_loss, invariance_loss, total_loss)
        
        return total_loss
    
    def _compute_invariance_loss(self, model, tetrahedron_features):
        """Compute invariance loss using Quantile RMSLE"""
        import GeometryUtils as gu
        
        # Get predictions for original input
        original_pred = model(tetrahedron_features)
        
        # Swap tetrahedrons
        swapped_features = gu.swap_tetrahedrons(tetrahedron_features)
        
        # Get predictions for swapped input
        swapped_pred = model(swapped_features)
        
        # Use Quantile RMSLE for consistency (same as volume regression)
        return self._compute_regression_loss(swapped_pred, original_pred)
    
    def _compute_regression_loss(self, predictions, targets):
        """Quantile RMSLE loss for volume regression"""
        
        # Ensure positive values for log operation
        pred_safe = torch.clamp(predictions, min=self.epsilon)
        target_safe = torch.clamp(targets, min=self.epsilon)
        
        # Calculate log differences
        log_pred = torch.log(pred_safe + 1.0)
        log_target = torch.log(target_safe + 1.0)
        log_diff = log_target - log_pred
        
        # Apply quantile weighting for asymmetric penalty
        quantile_weights = torch.where(
            log_diff >= 0,
            torch.tensor(self.tau, device=predictions.device),      # Underprediction penalty
            torch.tensor(1.0 - self.tau, device=predictions.device) # Overprediction penalty
        )
        
        # Apply quantile weighting to squared log differences
        squared_log_diff = log_diff ** 2
        weighted_squared_log_diff = quantile_weights * squared_log_diff
        
        # Compute weighted MSLE and take square root
        weighted_msle = torch.mean(weighted_squared_log_diff)
        return torch.sqrt(weighted_msle + self.epsilon)
    
    def _log_metrics(self, cls_logits, vol_pred, has_intersection, vol_target, 
                    cls_loss, reg_loss, inv_loss, total_loss):
        """Enhanced logging with Quantile RMSLE loss"""
        
        # Classification metrics
        pos_ratio = has_intersection.mean().item()
        
        # Volume prediction metrics
        vol_pred_std = vol_pred.std().item()
        vol_pred_range = f"[{vol_pred.min().item():.3f}, {vol_pred.max().item():.3f}]"
        
        vol_target_std = vol_target.std().item()
        vol_target_range = f"[{vol_target.min().item():.3f}, {vol_target.max().item():.3f}]"
        
        # Prediction direction analysis
        pred_safe = torch.clamp(vol_pred, min=self.epsilon)
        target_safe = torch.clamp(vol_target, min=self.epsilon)
        log_diff = torch.log(target_safe + 1.0) - torch.log(pred_safe + 1.0)
        underpred_pct = (log_diff >= 0).float().mean().item() * 100
        
        # Loss breakdown
        weighted_cls = self.classification_weight * cls_loss.item()
        weighted_reg = self.regression_weight * reg_loss.item()
        weighted_inv = self.invariance_weight * inv_loss.item()
        
        print(f"📊 [Sample {self.sample_count:,}] Quantile RMSLE Combined Loss (τ={self.tau}):")
        print(f"   📉 Total: {total_loss.item():.4f} (Cls: {weighted_cls:.4f} + Reg: {weighted_reg:.4f} + Inv: {weighted_inv:.4f})")
        print(f"   🎯 Pos ratio: {pos_ratio:.3f}")
        print(f"   📈 Vol Pred - Range: {vol_pred_range}, Std: {vol_pred_std:.4f}")
        print(f"   📋 Vol Target - Range: {vol_target_range}, Std: {vol_target_std:.4f}")
        print(f"   ⚖️  Underpredictions: {underpred_pct:.1f}% (penalty weight: {self.tau:.1f})")
        print(f"   🔄 Invariance Loss: {inv_loss.item():.6f}")

# ============================================================================
# MAIN TRAINER
# ============================================================================

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
        train_data_path = os.path.join(self.processed_data_path, "train", "train_data.csv")
        val_data_path = os.path.join(self.processed_data_path, "val", "val_data.csv")
        
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
        import GeometryUtils as gu
        
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

# ============================================================================
# TETRAHEDRON DATASET CLASS
# ============================================================================

class TetrahedronDataset(torch.utils.data.Dataset):
    def __init__(self, data_source):
        """
        Load data from a CSV file path or pandas DataFrame.
        NO NORMALIZATION - keep original volume values!
        
        Args:
            data_source: Path to CSV file (str) or pandas DataFrame
        """
        if isinstance(data_source, str):
            df = pd.read_csv(data_source)
            print(f"   Loaded {len(df):,} samples from {os.path.basename(data_source)}")
        elif isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
            print(f"   Using DataFrame with {len(df):,} samples")
        else:
            raise ValueError("data_source must be either a file path (str) or pandas DataFrame")
        
        # Extract features and targets - NO NORMALIZATION
        self.tetrahedron_features = df.iloc[:, :-2].values.astype(np.float64)
        self.intersection_volume = df.iloc[:, -2].values.astype(np.float64)
        self.intersection_status = df.iloc[:, -1].values.astype(np.float64)
        
        # Just store volume stats for info, but don't normalize
        positive_volumes = self.intersection_volume[self.intersection_volume > 0]
        if len(positive_volumes) > 0:
            print(f"   Volume range: [{positive_volumes.min():.2e}, {positive_volumes.max():.2e}]")
        
        # Convert to tensors - keep original values
        self.tetrahedron_features = torch.tensor(self.tetrahedron_features, dtype=torch.float64)
        self.intersection_status = torch.tensor(self.intersection_status, dtype=torch.float64).reshape(-1, 1)
        self.intersection_volume = torch.tensor(self.intersection_volume, dtype=torch.float64).reshape(-1, 1)
        
        # Analyze data distribution
        intersecting_count = (self.intersection_status > 0.5).sum().item()
        non_intersecting_count = len(self.intersection_status) - intersecting_count
        
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
        """Return features, status, volume"""
        return (
            self.tetrahedron_features[idx], 
            self.intersection_status[idx], 
            self.intersection_volume[idx]
        )