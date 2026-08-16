"""
Training loop for LSTM forecaster.

Features:
  - Early stopping with patience
  - Gradient clipping (critical for LSTMs)
  - Learning rate scheduling (ReduceLROnPlateau)
  - Optional W&B logging
  - Huber loss option (robust to PM2.5 outlier spikes)
  - Best model checkpointing

Usage:
    from src.training.train_lstm import train_model
    model, history = train_model(model, train_loader, val_loader, config)
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path

# Try importing wandb, but make it optional
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def train_model(model, train_loader, val_loader, config,
                model_name="lstm", use_wandb=False, save_dir="models"):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Model parameters: {model.count_parameters():,}")
    
    model = model.to(device)
    
    # Save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Loss function 
    loss_name = config.get("loss", "huber")
    if loss_name == "huber":
        criterion = nn.HuberLoss(delta=1.0)  
    elif loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name == "mae":
        criterion = nn.L1Loss()
    else:
        criterion = nn.HuberLoss(delta=1.0)
    
    print(f"Loss function: {loss_name}")
    
    # Optimizer 
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 1e-5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    
    # LR Scheduler 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Training config 
    max_epochs = config.get("max_epochs", 100)
    patience = config.get("early_stopping_patience", 10)
    grad_clip = config.get("grad_clip", 1.0)
    
    # W&B init 
    if use_wandb and HAS_WANDB:
        wandb.init(
            project=config.get("wandb_project", "aqi-forecasting"),
            name=model_name,
            config=config,
        )
        wandb.watch(model, log_freq=100)
    
    # Training loop
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    
    print(f"\n{'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | "
          f"{'LR':>10} | {'Time':>6} | Status")
    print("-" * 75)
    
    for epoch in range(max_epochs):
        epoch_start = time.time()
        
        # Training phase 
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping for LSTM stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation phase 
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start
        
        # Record history 
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)
        
        # LR scheduling 
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch
            status = "saved"
            
            # Save best model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
            }
            torch.save(checkpoint, save_path / f"best_{model_name}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                status = "early stop"
            elif patience_counter >= patience - 3:
                status = f"patience {patience_counter}/{patience}"
        
        # Logging 
        print(f"{epoch+1:5d} | {avg_train_loss:12.6f} | {avg_val_loss:12.6f} | "
              f"{current_lr:10.2e} | {elapsed:5.1f}s | {status}")
        
        if use_wandb and HAS_WANDB:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": current_lr,
            })
        
        # Stop if patience exhausted 
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # Load best model 
    best_ckpt = torch.load(save_path / f"best_{model_name}.pt",
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"\nLoaded best model from epoch {best_epoch + 1} "
          f"(val_loss={best_val_loss:.6f})")
    
    if use_wandb and HAS_WANDB:
        wandb.finish()
    
    return model, history


def evaluate_model(model, test_loader, device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            predictions = model(batch_x)
            
            all_preds.append(predictions.cpu().numpy())
            all_actuals.append(batch_y.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_actuals = np.concatenate(all_actuals, axis=0)
    
    print(f"Test predictions: {all_preds.shape}")
    print(f"  Pred range: [{all_preds.min():.2f}, {all_preds.max():.2f}]")
    print(f"  Actual range: [{all_actuals.min():.2f}, {all_actuals.max():.2f}]")
    
    return all_preds, all_actuals


def plot_training_history(history, save_path=None):
    """Plot training and validation loss curves."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss curves
    axes[0].plot(history["train_loss"], label="Train", linewidth=1.5)
    axes[0].plot(history["val_loss"], label="Validation", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1].plot(history["lr"], color="#DC2626", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()
    return fig
