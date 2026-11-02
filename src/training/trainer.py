import base64
import copy
import math
import time
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.training.losses import LOSS_BY_TASK
from src.utils.dataset_files import processed_split_path


def _load_split(path, task, batch_size, shuffle):
    """CSV columns are features..., IntersectionVolume, HasIntersection."""
    data = torch.tensor(pd.read_csv(path).values, dtype=torch.float64)
    features, volume, status = data[:, :-2], data[:, -2:-1], data[:, -1:]
    targets = {
        "IntersectionStatus": status,
        "IntersectionVolume": volume,
        "IntersectionStatus_IntersectionVolume": torch.cat([status, volume], dim=1),
    }[task]
    print(f"  {path}: {len(data):,} samples")
    return DataLoader(TensorDataset(features, targets), batch_size=batch_size, shuffle=shuffle)


def _run_epoch(loader, model, loss_fn, device, optimizer=None):
    """Mean batch loss over `loader`; trains when an optimizer is given, otherwise only evaluates."""
    total = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for features, targets in loader:
            loss = loss_fn(model(features.to(device)), targets.to(device))
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / len(loader)


def _plot_losses(history):
    """Base64 PNG of the train/val loss curves."""
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [h["train_loss"] for h in history], "o-", label="Training Loss", alpha=0.8)
    plt.plot(epochs, [h["val_loss"] for h in history], "s-", label="Validation Loss", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def train(model, config):
    """Train `model` with AdamW and early stopping; returns the model with its best-validation weights.

    Needs `epochs >= 1`. Raises FloatingPointError as soon as a loss is NaN or infinite.
    """
    task = config["task"]
    device = torch.device(config["device"])
    epochs = config.get("epochs", 100)
    patience = config.get("early_stopping_patience", 10)
    batch_size = config.get("batch_size", 1024)
    data_path = config.get("processed_data_path", "data/processed")

    model = model.to(device).double()
    loss_fn = LOSS_BY_TASK[task]
    optimizer = optim.AdamW(model.parameters(), lr=config.get("learning_rate", 1e-3), weight_decay=1e-5, foreach=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"-- Training {task}: {epochs} epochs, batch size {batch_size}, device {device} --")
    train_loader = _load_split(processed_split_path(data_path, "train"), task, batch_size, shuffle=True)
    # val is shuffled too: the CSVs are grouped by intersection type and losses are averaged per batch,
    # so unshuffled batches would be single-type and give a different (less representative) val loss
    val_loader = _load_split(processed_split_path(data_path, "val"), task, batch_size, shuffle=True)

    history, best_val_loss, best_state, stale_epochs = [], float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        train_loss = _run_epoch(train_loader, model, loss_fn, device, optimizer)
        model.eval()
        val_loss = _run_epoch(val_loader, model, loss_fn, device)
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise FloatingPointError(
                f"Loss is not finite in epoch {epoch} (train {train_loss}, val {val_loss}); "
                "check the learning rate and the processed data")

        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "epoch_time": time.time() - start})
        print(f"Epoch {epoch}/{epochs}: train {train_loss:.6f}, val {val_loss:.6f}, {history[-1]['epoch_time']:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss, best_state, stale_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

    model.load_state_dict(best_state)
    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    return {
        "trained_model": model,
        "training_metrics": {"final_val_loss": best_val_loss},
        "loss_plot": _plot_losses(history),
    }
