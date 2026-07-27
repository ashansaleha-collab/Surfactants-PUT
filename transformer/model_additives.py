import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional
from tqdm.auto import tqdm


class CMCRegressorWithAdditives(nn.Module):
    """
    pCMC prediction model using separate MoLFormer embeddings for the main molecule
    and the additive SMILES, plus additive concentration and presence masks.

    Inputs to the regression head:
      [mol_embed, temp, (add_embed * add_embed_mask), (add_conc * add_conc_mask), add_embed_mask, add_conc_mask]
    """

    def __init__(
        self,
        transformer_model_main,
        transformer_model_add,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.transformer_main = transformer_model_main
        self.transformer_add = transformer_model_add

        in_features = (
            embedding_dim  # mol embed
            + 1  # temperature
            + embedding_dim  # additive embed
            + 1  # additive concentration (scaled)
            + 1  # add_embed_mask
            + 1  # add_conc_mask
        )

        self.regression_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _embed_with(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output
        # return outputs.last_hidden_state[:, 0, :]

    def forward(self, X: dict) -> torch.Tensor:
        # Main molecule embedding
        input_ids = X["input_ids"]
        attention_mask = X["attention_mask"]
        temperature = X["temperature"]
        add_input_ids = X["add_input_ids"]
        add_attention_mask = X["add_attention_mask"]
        add_conc = X["add_conc"]
        add_embed_mask = X["add_embed_mask"]
        add_conc_mask = X["add_conc_mask"]

        mol_embed = self._embed_with(self.transformer_main, input_ids, attention_mask)

        # Additive embedding (always computed for consistent shapes)
        add_embed = self._embed_with(self.transformer_add, add_input_ids, add_attention_mask)

        # Gate additive features
        add_embed = add_embed * add_embed_mask
        gated_add_conc = add_conc * add_conc_mask

        # Concatenate features
        feat = torch.cat(
            [
                mol_embed,
                temperature,
                add_embed,
                gated_add_conc,
                add_embed_mask,
                add_conc_mask,
            ],
            dim=1,
        )

        pred = self.regression_head(feat)
        return pred.squeeze(-1)

    def predict(
        self,
        smiles_list: List[str],
        temperatures: np.ndarray,
        additive_smiles: Optional[List[Optional[str]]],
        additive_conc: Optional[np.ndarray],
        tokenizer,
        device,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        self.eval()
        preds_out = []

        n = len(smiles_list)
        n_batches = (n + batch_size - 1) // batch_size
        iterator = range(0, n, batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="Predicting", unit="batch")

        # Normalize additive_conc input shape: support scalar or array-like
        full_add_conc: Optional[np.ndarray]
        if additive_conc is None:
            full_add_conc = None
        else:
            if np.isscalar(additive_conc):
                full_add_conc = np.full(len(smiles_list), float(additive_conc), dtype=np.float32)
            else:
                arr = np.asarray(additive_conc, dtype=np.float32)
                if arr.ndim > 1:
                    arr = arr.reshape(-1)
                full_add_conc = arr

        with torch.no_grad():
            for i in iterator:
                sl = smiles_list[i : i + batch_size]
                tl = temperatures[i : i + batch_size]

                if additive_smiles is None:
                    add_sl = [""] * len(sl)
                else:
                    add_sl = [s if (s is not None and str(s).strip() != "") else "" for s in additive_smiles[i : i + batch_size]]

                if full_add_conc is None:
                    add_cl = [np.nan] * len(sl)
                else:
                    add_cl = full_add_conc[i : i + batch_size]

                # Masks
                add_embed_mask_np = np.array([1.0 if s != "" else 0.0 for s in add_sl], dtype=np.float32).reshape(-1, 1)
                add_conc_mask_np = np.array([0.0 if (c is None or (isinstance(c, float) and np.isnan(c))) else 1.0 for c in add_cl], dtype=np.float32).reshape(-1, 1)

                # Replace NaNs in concentration with 0 (post-scaling or raw) before tensor
                add_cl_np = np.array([0.0 if (c is None or (isinstance(c, float) and np.isnan(c))) else float(c) for c in add_cl], dtype=np.float32).reshape(-1, 1)

                # Tokenize
                main_inputs = tokenizer(sl, padding=True, return_tensors="pt", truncation=True, max_length=512).to(device)
                add_inputs = tokenizer(add_sl, padding=True, return_tensors="pt", truncation=True, max_length=512).to(device)

                temp_t = torch.tensor(tl, dtype=torch.float32, device=device).unsqueeze(1)
                add_conc_t = torch.tensor(add_cl_np, dtype=torch.float32, device=device)
                add_embed_mask_t = torch.tensor(add_embed_mask_np, dtype=torch.float32, device=device)
                add_conc_mask_t = torch.tensor(add_conc_mask_np, dtype=torch.float32, device=device)

                batch_preds = self.forward({
                    "input_ids": main_inputs["input_ids"],
                    "attention_mask": main_inputs["attention_mask"],
                    "temperature": temp_t,
                    "add_input_ids": add_inputs["input_ids"],
                    "add_attention_mask": add_inputs["attention_mask"],
                    "add_conc": add_conc_t,
                    "add_embed_mask": add_embed_mask_t,
                    "add_conc_mask": add_conc_mask_t,
                })

                preds_out.extend(batch_preds.detach().cpu().numpy())

        return np.array(preds_out)
