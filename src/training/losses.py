import torch
import torch.nn as nn
import src.utils.geometry as gu

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
