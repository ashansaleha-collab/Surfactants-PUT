from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _move_Xy_to_device(batch: Dict[str, Dict[str, torch.Tensor]], device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
    X = {k: v.to(device, non_blocking=True) for k, v in batch["X"].items()}
    y = batch["y"].to(device, non_blocking=True)
    return {"X": X, "y": y}


def _forward_batch(model, batch: Dict[str, Dict[str, torch.Tensor]]):
    # Pass the full feature dict X to the model in one go
    return model(batch["X"])


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    n_epochs: int = 100,
    patience: int = 25,
):
    history_train_loss: List[float] = []
    history_val_mse: List[float] = []
    history_val_loss: List[float] = []
    history_val_mae: List[float] = []
    history_val_r2: List[float] = []

    best_val_mse = float("inf")
    best_epoch = -1
    best_state = None
    patience_counter = 0

    epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = _move_Xy_to_device(batch, device)
            y = batch["y"]
            preds = _forward_batch(model, batch)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            epoch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss_epoch = float(np.mean(train_losses)) if train_losses else float("nan")
        history_train_loss.append(train_loss_epoch)

        # Validation
        model.eval()
        val_preds_list = []
        val_y_list = []
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_Xy_to_device(batch, device)
                y = batch["y"]
                preds = _forward_batch(model, batch)
                loss = criterion(preds, y)
                val_losses.append(loss.item())
                val_preds_list.append(preds.detach().cpu().numpy())
                val_y_list.append(y.detach().cpu().numpy())

        val_preds = np.concatenate(val_preds_list, axis=0) if val_preds_list else np.array([])
        val_y = np.concatenate(val_y_list, axis=0) if val_y_list else np.array([])
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        val_mse = mean_squared_error(val_y, val_preds) if val_y.size else float("nan")
        val_mae = mean_absolute_error(val_y, val_preds) if val_y.size else float("nan")
        val_r2 = r2_score(val_y, val_preds) if val_y.size else float("nan")

        history_val_mse.append(float(val_mse))
        history_val_loss.append(float(val_loss))
        history_val_mae.append(float(val_mae))
        history_val_r2.append(float(val_r2))

        print(
            f"Epoch {epoch+1}: train_loss={train_loss_epoch:.4f}, val_loss={val_loss:.4f}, val_mse={val_mse:.4f}, val_mae={val_mae:.4f}, val_r2={val_r2:.4f}"
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_pbar.set_description(f"Early stopping at epoch {epoch + 1}")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final val predictions with best model
    model.eval()
    final_preds_list = []
    final_y_list = []
    with torch.no_grad():
        for batch in val_loader:
            batch = _move_Xy_to_device(batch, device)
            y = batch["y"]
            preds = _forward_batch(model, batch)
            final_preds_list.append(preds.detach().cpu().numpy())
            final_y_list.append(y.detach().cpu().numpy())

    final_preds = np.concatenate(final_preds_list, axis=0) if final_preds_list else np.array([])
    final_y = np.concatenate(final_y_list, axis=0) if final_y_list else np.array([])

    results = {
        "val_mae": float(mean_absolute_error(final_y, final_preds)) if final_y.size else float("nan"),
        "val_r2": float(r2_score(final_y, final_preds)) if final_y.size else float("nan"),
        "val_rmse": float(np.sqrt(mean_squared_error(final_y, final_preds))) if final_y.size else float("nan"),
        "best_epoch": int(best_epoch),
        "train_loss_curve": history_train_loss,
        "val_mse_curve": history_val_mse,
        "val_loss_curve": history_val_loss,
    }

    return results, final_preds


def train_only(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    n_epochs: int = 100,
):
    """Simple training loop without any validation.

    Returns a results dict containing the training loss curve.
    """
    history_train_loss: List[float] = []

    epoch_pbar = tqdm(range(n_epochs), desc="Training (no val)", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = _move_Xy_to_device(batch, device)
            y = batch["y"]
            preds = _forward_batch(model, batch)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            epoch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss_epoch = float(np.mean(train_losses)) if train_losses else float("nan")
        history_train_loss.append(train_loss_epoch)
        epoch_pbar.set_postfix({"train_loss": f"{train_loss_epoch:.4f}"})

    results = {
        "train_loss_curve": history_train_loss,
    }

    return results
