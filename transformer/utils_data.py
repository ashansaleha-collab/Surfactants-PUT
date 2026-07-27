from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class StandardScaler:
    def __init__(self):
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        m = float(np.mean(x))
        s = float(np.std(x))
        if s == 0.0:
            s = 1.0
        self.mean_ = m
        self.std_ = s
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def tokenize_smiles_list(tokenizer, smiles: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
    enc = tokenizer(smiles, padding=True, return_tensors="pt", truncation=True, max_length=max_length)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


class CMCDataset(Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temps_scaled: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.temp = torch.tensor(temps_scaled, dtype=torch.float32).reshape(-1, 1)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "X": {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "temperature": self.temp[idx],
            },
            "y": self.y[idx],
        }


class CMCDatasetAdditives(Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temps_scaled: np.ndarray,
        targets: np.ndarray,
        add_input_ids: torch.Tensor,
        add_attention_mask: torch.Tensor,
        add_conc_scaled: np.ndarray,
        add_embed_mask: np.ndarray,
        add_conc_mask: np.ndarray,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.temp = torch.tensor(temps_scaled, dtype=torch.float32).reshape(-1, 1)
        self.y = torch.tensor(targets, dtype=torch.float32)

        self.add_input_ids = add_input_ids
        self.add_attention_mask = add_attention_mask
        self.add_conc = torch.tensor(add_conc_scaled, dtype=torch.float32).reshape(-1, 1)
        self.add_embed_mask = torch.tensor(add_embed_mask, dtype=torch.float32).reshape(-1, 1)
        self.add_conc_mask = torch.tensor(add_conc_mask, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "X": {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "temperature": self.temp[idx],
                "add_input_ids": self.add_input_ids[idx],
                "add_attention_mask": self.add_attention_mask[idx],
                "add_conc": self.add_conc[idx],
                "add_embed_mask": self.add_embed_mask[idx],
                "add_conc_mask": self.add_conc_mask[idx],
            },
            "y": self.y[idx],
        }


def scale_concentrations(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    out = []
    for c in values:
        if c is None or (isinstance(c, float) and np.isnan(c)):
            out.append(0.0)
        else:
            out.append((float(c) - mean) / std)
    return np.array(out, dtype=np.float32).reshape(-1, 1)


def collate_Xy_to_device(batch_list: List[Dict[str, torch.Tensor]], device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Collate a list of samples shaped as {"X": {...}, "y": tensor} into a batch.
    Note: returns CPU tensors; trainer moves them to the appropriate device.
    """
    x_keys = batch_list[0]["X"].keys()
    X = {k: torch.stack([b["X"][k] for b in batch_list], dim=0) for k in x_keys}
    y = torch.stack([b["y"] for b in batch_list], dim=0)
    return {"X": X, "y": y}
