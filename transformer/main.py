"""
Train surfactant pCMC prediction models using MoLFormer transformer embeddings
instead of chemical descriptors.

This script follows the same workflow as paper1.ipynb but replaces
the 19 Chen et al. descriptors with embeddings from the MoLFormer-XL model.
"""

import pathlib

import numpy as np
import torch
import torch.nn as nn
# Import data loading functions
from normalize_df import load_concat
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from utils_data import StandardScaler, CMCDataset, collate_Xy_to_device
from trainer import train_model
from plotting import plot_curves, plot_val_scatter
from utils_runtime import seed_everything

from model_base import CMCRegressor


project_root = pathlib.Path("transformer")

# Output directory
output_dir = project_root / "processed_data"
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("SURFACTANT pCMC PREDICTION WITH TRANSFORMER EMBEDDINGS")
print("=" * 70)


# ============================================================
# REPRODUCIBILITY
# ============================================================


seed_everything(42)


# ============================================================
# LOAD TRANSFORMER MODEL
# ============================================================
print("\nLoading MoLFormer-XL transformer model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Using device: {device}")

transformer_model = AutoModel.from_pretrained(
    "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
).to(device)

tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

print("   Model loaded successfully!")


# ============================================================
# MERGE DATASETS
# ============================================================
print("\nMerging datasets...")


merged_df = load_concat(expert_data=True, paper_1=True, paper_4=True)

print(f"   Total samples: {len(merged_df)}")
print(f"   Data preview:\n{merged_df.head()}")
merged_df.info()

# ============================================================
# PREPARE DATA FOR TRAINING
# ============================================================
print("\nPreparing data for training...")

# Extract features
smiles_list = merged_df["smiles"].tolist()
temperatures = merged_df["temp"].values
y = merged_df["pCMC"].values

print(f"   Total samples: {len(smiles_list)}")
print(f"   pCMC range: [{y.min():.3f}, {y.max():.3f}]")
print(f"   pCMC mean: {y.mean():.3f} ± {y.std():.3f}")


# ============================================================
# CREATE AND TRAIN MODEL (single train/validation split)
# ============================================================
print("\nCreating and training CMCRegressor model (no cross-validation)...")

# Initialize model
# Avoid getattr per project rules; fail loudly if missing
embedding_dim = transformer_model.config.hidden_size
hidden_dim = 256
dropout = 0.2

model = CMCRegressor(
    transformer_model=transformer_model,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    dropout=dropout,
).to(device)

print(f"   Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

# Training setup
head_lr = 1e-3
transformer_lr = 1e-5
# weight_decay = 0.01
criterion = nn.HuberLoss(delta=0.5, reduction="mean")

# Use different learning rates for head vs transformer for stable finetuning
optimizer = torch.optim.Adam(
    [
        {"params": model.transformer.parameters(), "lr": transformer_lr},
        {"params": model.regression_head.parameters(), "lr": head_lr},
    ],
)

# Reproducibility and split
np.random.seed(42)
torch.manual_seed(42)

train_smiles, val_smiles, train_temps, val_temps, train_y, val_y = train_test_split(
    smiles_list,
    temperatures,
    y,
    test_size=0.1,
    random_state=42,
    shuffle=True,
)

print(f"   Train size: {len(train_smiles)} | Val size: {len(val_smiles)}")

# --------------------------------------------
# Pre-tokenize SMILES for train/val once
# --------------------------------------------
print("\nTokenizing train/val SMILES once...")

# Tokenize to CPU tensors; move to device in batches inside the loop
train_inputs = tokenizer(
    train_smiles,
    padding=True,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)
val_inputs = tokenizer(
    val_smiles,
    padding=True,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)

# Convert targets to tensors (CPU)
train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
val_y_tensor = torch.tensor(val_y, dtype=torch.float32)

# --------------------------------------------
# Temperature standardization (fit on train only)
# --------------------------------------------
temp_scaler = StandardScaler()
train_temps_scaled = temp_scaler.fit_transform(train_temps)
val_temps_scaled = temp_scaler.transform(val_temps)

# Build datasets and loaders
batch_size = 32
train_inputs = tokenizer(
    train_smiles,
    padding=True,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)
val_inputs = tokenizer(
    val_smiles,
    padding=True,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)

train_ds = CMCDataset(
    input_ids=train_inputs["input_ids"],
    attention_mask=train_inputs["attention_mask"],
    temps_scaled=train_temps_scaled,
    targets=train_y,
)
val_ds = CMCDataset(
    input_ids=val_inputs["input_ids"],
    attention_mask=val_inputs["attention_mask"],
    temps_scaled=val_temps_scaled,
    targets=val_y,
)

def _collate(batch_list):
    return collate_Xy_to_device(batch_list, device)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=_collate)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=_collate)

# Recreate model (unchanged) and optimizer
model = CMCRegressor(
    transformer_model=transformer_model,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    dropout=dropout,
).to(device)

head_lr = 1e-3
transformer_lr = 1e-5
criterion = nn.HuberLoss(delta=0.5, reduction="mean")
optimizer = torch.optim.Adam(
    [
        {"params": model.transformer.parameters(), "lr": transformer_lr},
        {"params": model.regression_head.parameters(), "lr": head_lr},
    ]
)

results, val_predictions = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    n_epochs=100,
    patience=25,
)


# ============================================================
# OVERALL RESULTS
# ============================================================
print("\nOverall Performance Summary...")

n_total = len(y)

print("\n" + "=" * 70)
print("FINAL RESULTS (Transformer Embeddings + Temperature)")
print("=" * 70)
print(f"   Total samples: {n_total}")
print(f"   Split: Train={len(train_smiles)} | Val={len(val_smiles)} (best epoch: {results['best_epoch']})")
print(f"   Val R²:   {results['val_r2']:.4f}")
print(f"   Val MAE:  {results['val_mae']:.4f}")
print(f"   Val RMSE: {results['val_rmse']:.4f}")
print("\n" + "=" * 70)


# ============================================================
# SAVE MODEL
# ============================================================
print("\nSaving model...")

models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)

# Save model state dict
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "transformer_model_name": "ibm/MoLFormer-XL-both-10pct",
        "embedding_dim": int(embedding_dim),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "head_lr": float(head_lr),
        "transformer_lr": float(transformer_lr),
        # "weight_decay": float(weight_decay),
        "results": results,
    },
    models_dir / "pCMC_model_transformer.pt",
)

print(f"   Model saved to: {models_dir}")
print("      - pCMC_model_transformer.pt")


# ============================================================
# VISUALIZATION
# ============================================================
print("\nCreating visualizations...")

train_curve_path = output_dir / "training_curve.png"
plot_curves(results, results.get("best_epoch", -1), train_curve_path, title_suffix="")
print(f"   Saved: {train_curve_path}")

val_plot_path = output_dir / "val_predictions_transformer.png"
plot_val_scatter(
    y_true=val_y,
    y_pred=val_predictions,
    path=val_plot_path,
    title=f"Val Predictions (R²={results['val_r2']:.3f}, MAE={results['val_mae']:.3f})",
)
print(f"   Saved: {val_plot_path}")

print("\n" + "=" * 70)
print("SCRIPT COMPLETED SUCCESSFULLY!")
print("=" * 70)
