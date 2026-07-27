import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


class CMCRegressor(nn.Module):
    """
    pCMC prediction model using MoLFormer embeddings + regression head
    """

    def __init__(self, transformer_model, embedding_dim=768, hidden_dim=256, dropout=0.0):
        super().__init__()
        self.transformer = transformer_model

        # Freeze transformer weights (only train regression head)
        # for param in self.transformer.parameters():
        #     param.requires_grad = False

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),  # +1 for temperature
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # Output: pCMC prediction
        )

    def forward(self, X):
        """
        Forward pass

        Args:
            X: dict with keys 'input_ids', 'attention_mask', 'temperature'

        Returns:
            pCMC predictions [batch_size, 1]
        """
        # Get transformer embeddings (enable gradients for finetuning)
        input_ids = X["input_ids"]
        attention_mask = X["attention_mask"]
        temperature = X["temperature"]
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Prefer pooled output; fallback to CLS token if pooler is unavailable
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output  # [batch_size, embedding_dim]
        else:
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Concatenate embeddings with temperature
        combined = torch.cat([embeddings, temperature], dim=1)  # [batch_size, embedding_dim + 1]

        # Pass through regression head
        prediction = self.regression_head(combined)

        return prediction.squeeze(-1)  # [batch_size]

    def predict(self, smiles_list, temperatures, tokenizer, device, batch_size=32, show_progress=False):
        """
        Predict pCMC for a list of SMILES and temperatures

        Args:
            smiles_list: List of SMILES strings
            temperatures: List or array of temperatures
            tokenizer: Tokenizer for SMILES
            device: torch device
            batch_size: Batch size for inference
            show_progress: Whether to show progress bar

        Returns:
            predictions: numpy array of pCMC predictions
        """
        self.eval()
        predictions = []

        n_batches = (len(smiles_list) + batch_size - 1) // batch_size
        batch_iterator = range(0, len(smiles_list), batch_size)

        if show_progress:
            batch_iterator = tqdm(batch_iterator, total=n_batches, desc="Predicting", unit="batch")

        with torch.no_grad():
            for i in batch_iterator:
                batch_smiles = smiles_list[i : i + batch_size]
                batch_temps = temperatures[i : i + batch_size]

                # Tokenize
                inputs = tokenizer(batch_smiles, padding=True, return_tensors="pt", truncation=True, max_length=512).to(
                    device
                )

                # Prepare temperature tensor
                temp_tensor = torch.tensor(batch_temps, dtype=torch.float32).unsqueeze(1).to(device)

                # Forward pass
                preds = self.forward({
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "temperature": temp_tensor,
                })

                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)
