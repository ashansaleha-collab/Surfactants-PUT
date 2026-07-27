# Transformer-Based pCMC Prediction

This script trains surfactant pCMC prediction models using MoLFormer-XL transformer embeddings instead of chemical descriptors. The training now uses a single train/validation split (no cross-validation) and produces clear training and evaluation plots.

## Overview

- Input: SMILES strings from Paper 1 (Chen et al. 2024) and Paper 4
- Features: MoLFormer-XL pooled embedding (768-dim) + temperature (1-dim)
- Model: Frozen transformer backbone + small regression head
- Output: Trained model state, training curve, and validation scatter plot

## Requirements

Dependencies are listed in the project `requirements.txt`. Make sure `torch` and `transformers` are installed.

## Usage

Run the training script from the project root or the `transformer/` folder:

```bash
python transformer/main.py
```

The script will automatically:
1. Load MoLFormer-XL and tokenize SMILES
2. Load and merge Paper 1 and Paper 4 datasets
3. Split into train/validation (80/20, random_state=42)
4. Train only the regression head with early stopping on validation MSE
5. Save the best model head and plots

## Outputs

Saved under `transformer/processed_data/`:
- `merged_data.csv` — merged dataset used for training
- `training_curve.png` — Train vs Validation MSE over epochs, best epoch marked
- `val_predictions_transformer.png` — Validation Predicted vs Actual pCMC scatter plot

Saved under `transformer/models/`:
- `pCMC_model_transformer.pt` — PyTorch state dict plus metrics and hyperparameters

## Console Summary

The end of the run prints a summary like:

```
FINAL RESULTS (Transformer Embeddings + Temperature)
======================================================================
   Total samples: <n>
   Split: Train=<n_train> | Val=<n_val> (best epoch: <epoch>)
   Val R²:   <r2>
   Val MAE:  <mae>
   Val RMSE: <rmse>
======================================================================
```

## Notes

- GPU is used if available; otherwise CPU is used.
- You can tweak hidden size, dropout, learning rate, batch size, epochs, and patience in `main.py`.

## References

- MoLFormer-XL (Hugging Face): ibm/MoLFormer-XL-both-10pct
- Chen et al. (2024) for the original dataset

### Additives Variant

- New script: `transformer/main_additives.py` trains a model that also consumes optional additive information from `sources/CMC_surfactants_database_v2.csv`.
- Expected columns after normalization: `smiles`, `temp`, `pCMC`, `additive` (SMILES or NaN), `conc_additives` (float or NaN).
- Missing handling is explicit:
   - `additive` missing → additive embedding is zeroed and an `add_embed_mask` feature is set to 0 (1 if present).
   - `conc_additives` missing → concentration is set to 0 after standardization and an `add_conc_mask` feature is set to 0 (1 if present).
- Architecture: uses two separate MoLFormer backbones — one for main SMILES and one for additive SMILES — both fine-tuned jointly alongside the regression head. The two encoders do not share weights.
- Train:

```
uv run python transformer/main_additives.py
```

- Artifacts: `processed_data/training_curve_additives.png`, `processed_data/val_predictions_transformer_additives.png`, and model file `transformer/models/pCMC_model_transformer_additives.pt`.
   The saved checkpoint stores `transformer_model_names.main` and `transformer_model_names.additive` for reproducibility.

## Refactor Notes

- Shared modules introduced to reduce duplication:
   - `utils_runtime.py`: seeding, device selection, transformer loader.
   - `utils_data.py`: simple `StandardScaler`, tokenization helper, `CMCDataset` and `CMCDatasetAdditives`, and device collate.
   - `trainer.py`: training and validation loop with early stopping on validation MSE; returns metrics and final predictions.
   - `plotting.py`: training curve and validation scatter plotting helpers.
- The scripts `main.py` and `main_additives.py` now build datasets and use DataLoaders, preserving the original hyperparameters, metrics, and saved artifact paths.

### Reproducibility

- Both `transformer/main.py` and `transformer/main_additives.py` set deterministic seeds for Python, NumPy, and PyTorch, and configure cuDNN to deterministic mode.
- Train/test splits already use `random_state=42`.
- Note: some transformer operations can still be nondeterministic on certain GPUs or driver/toolchain versions; the scripts request deterministic algorithms when available.
