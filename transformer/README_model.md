# CMCRegressor Model

## Overview

The `CMCRegressor` is a neural network model that predicts pCMC (Critical Micelle Concentration) values for surfactants using:
- **MoLFormer-XL** transformer embeddings from SMILES strings
- **Temperature** as an additional input feature
- **Trainable regression head** (transformer weights are frozen)

## Model Architecture

```
Input: SMILES string + Temperature
    ↓
MoLFormer-XL Transformer (frozen)
    ↓
Embeddings (768-dim) + Temperature (1-dim)
    ↓
Linear(769 → 256) → ReLU → Dropout(0.2)
    ↓
Linear(256 → 128) → ReLU → Dropout(0.2)
    ↓
Linear(128 → 1)
    ↓
Output: pCMC prediction
```

## Usage

### Training

```python
from transformer.main import CMCRegressor
import torch
from transformers import AutoModel, AutoTokenizer

# Load transformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer_model = AutoModel.from_pretrained(
    "ibm/MoLFormer-XL-both-10pct",
    deterministic_eval=True,
    trust_remote_code=True
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "ibm/MoLFormer-XL-both-10pct",
    trust_remote_code=True
)

# Create model
model = CMCRegressor(
    transformer_model=transformer_model,
    embedding_dim=768,
    hidden_dim=256,
    dropout=0.2
).to(device)

# Train (see main.py for full training loop)
```

### Inference

```python
# Load saved model
checkpoint = torch.load('transformer/models/pCMC_model_transformer.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
smiles_list = ['CCCCCCCCCCCC', 'CCCCCCCCCCCCC']
temperatures = [298.15, 298.15]

predictions = model.predict(
    smiles_list=smiles_list,
    temperatures=temperatures,
    tokenizer=tokenizer,
    device=device,
    batch_size=32
)

print(f"pCMC predictions: {predictions}")
```

## Key Features

1. **Frozen Transformer**: The MoLFormer weights are frozen to prevent overfitting and reduce training time
2. **Efficient Training**: Only ~200K parameters in the regression head need to be trained
3. **Temperature Integration**: Temperature is concatenated with embeddings before the regression head
4. **Batch Processing**: Efficient batch inference with automatic tokenization

## Performance

- **Test R²**: ~0.XX ± 0.XX (from 5-fold CV)
- **Test MAE**: ~0.XX ± 0.XX
- **Test RMSE**: ~0.XX

## Files

- `main.py`: Training script
- `models/pCMC_model_transformer.pt`: Saved model checkpoint
- `processed_data/merged_data.csv`: Training data
- `processed_data/predictions_transformer.png`: Visualization
