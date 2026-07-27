"""
Train surfactant pCMC prediction model using MoLFormer embeddings with optional
additive SMILES and additive concentration as auxiliary inputs.

This mirrors transformer/main.py but adds additive-aware features.
"""

import pathlib
import re

import numpy as np
import torch
import torch.nn as nn
from model_additives import CMCRegressorWithAdditives
from normalize_df import load_concat_additives
from plotting import plot_curves, plot_val_scatter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from trainer import train_model
from transformers import AutoModel, AutoTokenizer
from utils_data import (
    CMCDatasetAdditives,
    StandardScaler,
    collate_Xy_to_device,
    scale_concentrations,
)
from utils_runtime import seed_everything

project_root = pathlib.Path("transformer")

# Output directory
output_dir = project_root / "processed_data"
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("SURFACTANT pCMC WITH ADDITIVE FEATURES (TRANSFORMER EMBEDDINGS)")
print("=" * 70)


# ============================================================
# REPRODUCIBILITY
# ============================================================

seed_everything(42)


# ============================================================
# LOAD TRANSFORMER MODEL
# ============================================================
print("\nLoading MoLFormer-XL transformer model (shared for main/additive)...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Using device: {device}")

model_name_main = "ibm/MoLFormer-XL-both-10pct"
model_name_add = "ibm/MoLFormer-XL-both-10pct"

# Load a single shared transformer model instance and reuse for both paths
# Enable deterministic eval path exposed by MoLFormer to avoid nondeterministic attention backends
_shared_transformer = AutoModel.from_pretrained(
    model_name_main,
    deterministic_eval=True,
    trust_remote_code=True,
).to(device)
transformer_model_main = _shared_transformer
transformer_model_add = _shared_transformer

tokenizer = AutoTokenizer.from_pretrained(model_name_main, trust_remote_code=True)

print("   Model loaded successfully (shared instance).")


# ============================================================
# MERGE DATASETS
# ============================================================
print("\nMerging datasets (with additives)...")

merged_df = load_concat_additives()

print(f"   Total samples: {len(merged_df)}")
print(f"   Columns: {list(merged_df.columns)}")


# ============================================================
# PREPARE DATA FOR TRAINING
# ============================================================
print("\nPreparing data for training...")

smiles_list = merged_df["smiles"].astype(str).tolist()
temperatures = merged_df["temp"].values
y = merged_df["pCMC"].values

# Additive features (strings can be NaN)
add_smiles_col = merged_df["additive"].astype("string")
additive_smiles = [s if (s is not None and str(s) != "<NA>") else "" for s in add_smiles_col]

conc_series = merged_df["conc_additives"]
conc_values = conc_series.values

print(f"   Total samples: {len(smiles_list)}")
print(f"   pCMC range: [{y.min():.3f}, {y.max():.3f}] | mean={y.mean():.3f} ± {y.std():.3f}")


# ============================================================
# MODEL INIT
# ============================================================
print("\nCreating model...")

embedding_dim = transformer_model_main.config.hidden_size
assert transformer_model_add.config.hidden_size == embedding_dim, (
    "Main and additive transformer hidden sizes must match"
)
hidden_dim = 256

# Reproducibility and split
np.random.seed(42)
torch.manual_seed(42)

(
    train_smiles,
    val_smiles,
    train_temps,
    val_temps,
    train_y,
    val_y,
    train_add_smiles,
    val_add_smiles,
    train_conc,
    val_conc,
) = train_test_split(
    smiles_list,
    temperatures,
    y,
    additive_smiles,
    conc_values,
    test_size=0.1,
    random_state=42,
    shuffle=True,
)

print(f"   Train size: {len(train_smiles)} | Val size: {len(val_smiles)}")


# --------------------------------------------
# Tokenize SMILES once (train/val)
# --------------------------------------------
print("\nTokenizing SMILES (main + additive) once...")

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

# Additives (empty string for missing)
train_add_inputs = tokenizer(
    [s if (s is not None and str(s).strip() != "") else "" for s in train_add_smiles],
    padding=True,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)
val_add_inputs = tokenizer(
    [s if (s is not None and str(s).strip() != "") else "" for s in val_add_smiles],
    padding=True,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)


# --------------------------------------------
# Build masks and scale numeric covariates (fit on train)
# --------------------------------------------
def _mask_from_smiles(s_list):
    return np.array(
        [1.0 if (s is not None and str(s).strip() != "") else 0.0 for s in s_list], dtype=np.float32
    ).reshape(-1, 1)


def _mask_from_conc(c_array):
    return np.array(
        [0.0 if (c is None or (isinstance(c, float) and np.isnan(c))) else 1.0 for c in c_array], dtype=np.float32
    ).reshape(-1, 1)


train_add_embed_mask = _mask_from_smiles(train_add_smiles)
val_add_embed_mask = _mask_from_smiles(val_add_smiles)

train_add_conc_mask = _mask_from_conc(train_conc)
val_add_conc_mask = _mask_from_conc(val_conc)

# Temperature scaling (standardize on train)
temp_scaler = StandardScaler()
train_temps_scaled = temp_scaler.fit_transform(train_temps)
val_temps_scaled = temp_scaler.transform(val_temps)

# Concentration scaling (fit on present only)
present_train_conc = np.array(
    [c for c in train_conc if not (c is None or (isinstance(c, float) and np.isnan(c)))], dtype=float
)
if present_train_conc.size == 0:
    conc_mean, conc_std = 0.0, 1.0
else:
    conc_mean = float(np.mean(present_train_conc))
    conc_std = float(np.std(present_train_conc))
    if conc_std == 0.0:
        conc_std = 1.0

train_conc_scaled = scale_concentrations(train_conc, conc_mean, conc_std)
val_conc_scaled = scale_concentrations(val_conc, conc_mean, conc_std)

train_add_embed_mask_t = torch.tensor(train_add_embed_mask, dtype=torch.float32)
val_add_embed_mask_t = torch.tensor(val_add_embed_mask, dtype=torch.float32)
train_add_conc_mask_t = torch.tensor(train_add_conc_mask, dtype=torch.float32)
val_add_conc_mask_t = torch.tensor(val_add_conc_mask, dtype=torch.float32)


batch_size = 32


# Datasets and loaders
def _collate(batch_list):
    return collate_Xy_to_device(batch_list, device)


train_ds = CMCDatasetAdditives(
    input_ids=train_inputs["input_ids"],
    attention_mask=train_inputs["attention_mask"],
    temps_scaled=train_temps_scaled,
    targets=train_y,
    add_input_ids=train_add_inputs["input_ids"],
    add_attention_mask=train_add_inputs["attention_mask"],
    add_conc_scaled=train_conc_scaled,
    add_embed_mask=train_add_embed_mask,
    add_conc_mask=train_add_conc_mask,
)
val_ds = CMCDatasetAdditives(
    input_ids=val_inputs["input_ids"],
    attention_mask=val_inputs["attention_mask"],
    temps_scaled=val_temps_scaled,
    targets=val_y,
    add_input_ids=val_add_inputs["input_ids"],
    add_attention_mask=val_add_inputs["attention_mask"],
    add_conc_scaled=val_conc_scaled,
    add_embed_mask=val_add_embed_mask,
    add_conc_mask=val_add_conc_mask,
)

_dl_generator = torch.Generator().manual_seed(42)
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    collate_fn=_collate,
    generator=_dl_generator,
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    collate_fn=_collate,
    generator=_dl_generator,
)

# Create model and optimizer
model = CMCRegressorWithAdditives(
    transformer_model_main=transformer_model_main,
    transformer_model_add=transformer_model_add,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
).to(device)

print(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Optimizer and loss (single definition here)
head_lr = 1e-3
transformer_lr = 1e-5
criterion = nn.HuberLoss(delta=0.5, reduction="mean")

if model.transformer_main is model.transformer_add:
    optimizer = torch.optim.Adam(
        [
            {"params": model.transformer_main.parameters(), "lr": transformer_lr},
            {"params": model.regression_head.parameters(), "lr": head_lr},
        ]
    )
else:
    optimizer = torch.optim.Adam(
        [
            {"params": model.transformer_main.parameters(), "lr": transformer_lr},
            {"params": model.transformer_add.parameters(), "lr": transformer_lr},
            {"params": model.regression_head.parameters(), "lr": head_lr},
        ]
    )

def infer_pCMC_vs_temperature(
    smiles: str,
    additive_smiles: str | None = None,
    additive_conc: float | None = None,
    temp_range: np.ndarray | None = None,
    model=model,
    tokenizer=tokenizer,
    device=device,
    batch_size: int = 32,
    show_progress: bool = True,
    plot: bool = True,
    output_dir=output_dir,
):
    """
    Run a temperature sweep inference for a single surfactant (optionally with an additive)
    and optionally plot/save the results.

    Returns: dict with keys: "temperatures" (np.ndarray), "predictions" (np.ndarray), "plot_path" (path or None)
    """
    import matplotlib.pyplot as plt

    if temp_range is None:
        temp_range = np.linspace(20, 60, 21)

    main_smiles_list = [smiles] * len(temp_range)
    if additive_smiles is None:
        additive_smiles_list = [None] * len(temp_range)
    else:
        additive_smiles_list = [additive_smiles] * len(temp_range)

    # Apply the same scaling used during training
    temps_scaled = (temp_range - float(temp_scaler.mean_)) / float(temp_scaler.std_)
    if additive_conc is not None:
        conc_scaled_for_infer = (float(additive_conc) - float(conc_mean)) / float(conc_std)
    else:
        conc_scaled_for_infer = None

    predictions = model.predict(
        smiles_list=main_smiles_list,
        temperatures=temps_scaled,
        additive_smiles=additive_smiles_list,
        additive_conc=conc_scaled_for_infer,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    preds = np.asarray(predictions)

    plot_path = None
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        ax.plot(
            temp_range,
            preds,
            marker="o",
            linewidth=2.5,
            markersize=6,
            color="#1f77b4",
            label="Predicted pCMC",
        )
        ax.fill_between(temp_range, preds - 0.1, preds + 0.1, alpha=0.2, color="#1f77b4")

        if additive_smiles is None:
            dataset_mask = (merged_df["smiles"] == smiles) & (merged_df["additive"].isna())
        else:
            dataset_mask = (merged_df["smiles"] == smiles) & (merged_df["additive"] == additive_smiles)
        dataset_points = merged_df[dataset_mask]
        if len(dataset_points) > 0:
            ax.scatter(
                dataset_points["temp"],
                dataset_points["pCMC"],
                color="red",
                s=100,
                marker="o",
                zorder=5,
                label="Dataset points",
                edgecolors="darkred",
                linewidth=1.5,
            )

        ax.set_xlabel("Temperature (°C)", fontsize=12, fontweight="bold")
        ax.set_ylabel("pCMC", fontsize=12, fontweight="bold")
        title_add = "No Additive" if additive_smiles is None else f"Additive: {additive_smiles} (conc={additive_conc})"
        ax.set_title(f"pCMC vs Temperature for {smiles} ({title_add})", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11)
        plt.tight_layout()

        safe_smiles = re.sub(r"[^0-9A-Za-z._-]", "_", smiles)
        file_name = f"inference_temp_sweep_{safe_smiles}"
        if additive_smiles is not None:
            safe_add = re.sub(r"[^0-9A-Za-z._-]", "_", additive_smiles)
            file_name += f"_add_{safe_add}_{additive_conc}"
        file_name += ".png"

        plot_path = output_dir / file_name
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    # Print concise summary and table
    print(f"\nTarget SMILES: {smiles}")
    print(f"   Additive: {additive_smiles} | Concentration: {additive_conc}")
    print(f"   Temperature range: {temp_range[0]:.1f} to {temp_range[-1]:.1f} °C ({len(temp_range)} points)")
    print(f"   Predictions shape: {preds.shape}")
    print(f"   Min pCMC: {preds.min():.4f} | Max pCMC: {preds.max():.4f} | Mean pCMC: {preds.mean():.4f}")

    print("\n" + "=" * 60)
    print("Temperature Sweep Results:")
    print("=" * 60)
    print(f"{'Temperature (°C)':<20} {'Predicted pCMC':<20}")
    print("-" * 60)
    for temp, pred in zip(temp_range, preds):
        print(f"{temp:<20.1f} {pred:<20.4f}")
    print("=" * 60)

    return {"temperatures": temp_range, "predictions": preds, "plot_path": plot_path}


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


# results and val_predictions already computed by trainer


# ============================================================
# SUMMARY
# ============================================================
print("\nOverall Performance Summary...")
n_total = len(y)

print("\n" + "=" * 70)
print("FINAL RESULTS (Mol + Temp + Additive Embed + Additive Conc)")
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

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "transformer_model_names": {
            "main": model_name_main,
            "additive": model_name_add,
        },
        "embedding_dim": int(embedding_dim),
        "hidden_dim": int(hidden_dim),
        "head_lr": float(head_lr),
        "transformer_lr": float(transformer_lr),
        "temp_mean": float(temp_scaler.mean_),
        "temp_std": float(temp_scaler.std_),
        "conc_mean": float(conc_mean),
        "conc_std": float(conc_std),
        "uses_additives": True,
        "results": results,
    },
    models_dir / "pCMC_model_transformer_additives.pt",
)

print(f"   Model saved to: {models_dir}")
print("      - pCMC_model_transformer_additives.pt")


# ============================================================
# VISUALIZATION
# ============================================================
print("\nCreating visualizations...")
train_curve_path = output_dir / "training_curve_additives.png"
plot_curves(results, results.get("best_epoch", -1), train_curve_path, title_suffix="(Additives)")
print(f"   Saved: {train_curve_path}")

val_plot_path = output_dir / "val_predictions_transformer_additives.png"
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
