from __future__ import annotations

import math
import shutil
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from cli.data import AnnotatedSurfactantsDf, SurfactantsDf

# Lazy import model class from training package
from transformer.model_additives import CMCRegressorWithAdditives
from transformer.trainer import train_model as _train_loop, train_only as _train_only
from transformer.utils_data import (
    CMCDatasetAdditives,
    StandardScaler,
    collate_Xy_to_device,
    scale_concentrations,
)

from . import Model

HF_URL = "https://huggingface.co/Val-2/surfactants/resolve/main/pCMC_model_transformer_additives.pt?download=true"
DEFAULT_CKPT_NAME = "pCMC_model_transformer_additives.pt"


class TransformerModel(Model):
    """Transformer-based pCMC predictor.

    Inference is supported via `predict`. A minimal `train` implementation is
    added to allow K-Fold early-stopping cross validation without rewriting
    the original standalone training script. The model architecture itself is
    reused from `transformer.model_additives`.
    """

    def __init__(self, params: dict[str, str]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size: int = 64
        self.model_name_main = "ibm/MoLFormer-XL-both-10pct"
        self.head_lr: float = 1e-3
        self.transformer_lr: float = 1e-5
        self.epochs: int = int(params.get("epochs", 50))
        self.patience: int = int(params.get("patience", 25))

        transformer_shared = AutoModel.from_pretrained(
            self.model_name_main, deterministic_eval=True, trust_remote_code=True
        ).to(self.device)
        transformer_main = transformer_shared
        transformer_add = transformer_shared
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_main, trust_remote_code=True)

        embedding_dim = transformer_shared.config.hidden_size
        hidden_dim = 2048

        net = CMCRegressorWithAdditives(
            transformer_model_main=transformer_main,
            transformer_model_add=transformer_add,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.tokenizer = tokenizer
        self.model = net
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

    def get_param_grid(self):
        return {"patience": [20]}

    def load(self, _data_dir: Path | None = None) -> None:
        # Resolve checkpoint path. Priority:
        # 1) explicit local path passed via self.local_model_path
        # 2) _data_dir if provided: if it's a URL -> download; if it's a local file -> use it
        # 3) default HF_URL download into cache
        _data_dir = None
        models_dir = Path("cache") / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path: Path | None = None
        if _data_dir is not None:
            # Allow _data_dir to be a URL string or a Path to a local file
            _s = str(_data_dir)
            if _s.startswith("http://") or _s.startswith("https://"):
                ckpt_path = models_dir / Path(_s).name
                if not ckpt_path.exists():
                    print(f"Downloading model from URL: {_s}")
                    with urlopen(_s) as r, open(ckpt_path, "wb") as f:
                        shutil.copyfileobj(r, f)
                else:
                    print("Model already present in cache.")
            else:
                p = Path(_data_dir)
                if not p.exists():
                    raise FileNotFoundError(f"Provided _data_dir does not exist: {p}")
                ckpt_path = p
                print(f"Loading model from provided local path: {ckpt_path}")
        # 3) fallback to default HF_URL
        else:
            print("Loading model from Hugging Face cache...")
            ckpt_path = models_dir / "pCMC_model_transformer_additives.pt"
            if not ckpt_path.exists():
                print("Downloading model...")
                with urlopen(HF_URL) as r, open(ckpt_path, "wb") as f:
                    shutil.copyfileobj(r, f)
            print("Model found in cache.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model_name_main = ckpt["transformer_model_names"]["main"]
        # model_name_add retained for completeness in checkpoint; shared backbone is used
        _ = ckpt["transformer_model_names"]["additive"]
        embedding_dim = int(ckpt["embedding_dim"])
        hidden_dim = int(ckpt["hidden_dim"])
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Load shared MoLFormer backbone and tokenizer
        transformer_main = AutoModel.from_pretrained(
            self.model_name_main, deterministic_eval=True, trust_remote_code=True
        ).to(self.device)
        transformer_add = transformer_main  # shared instance as in training
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_main, trust_remote_code=True)

        # Build and load model
        model = CMCRegressorWithAdditives(
            transformer_model_main=transformer_main,
            transformer_model_add=transformer_add,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self.model = model

        # Load scalers
        self.temp_mean: float = ckpt["temp_mean"]
        self.temp_std: float = ckpt["temp_std"]
        self.conc_mean: float = ckpt["conc_mean"]
        self.conc_std: float = ckpt["conc_std"]

    def save(self, _data_dir: Path) -> None:
        """Save the current model and metadata to a checkpoint file.

        If `_data_dir` is a directory, the checkpoint is written into it with
        the fixed filename `pCMC_model_transformer_additives.pt`. If it's a
        filepath, that path is used directly.
        """
        import torch

        if self.model is None:
            raise RuntimeError("No model available to save. Instantiate or load a model first.")

        # Resolve target path
        target = Path(_data_dir)
        if target.exists() and target.is_dir():
            models_dir = target
            models_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = models_dir / "pCMC_model_transformer_additives.pt"
        else:
            # Treat provided path as a file path (parent dir may need creation)
            ckpt_path = target
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state_dict": self.model.state_dict(),
            "transformer_model_names": {"main": self.model_name_main, "additive": self.model_name_main},
            "embedding_dim": getattr(self, "embedding_dim", self.model.transformer_main.config.hidden_size),
            "hidden_dim": getattr(self, "hidden_dim", self.model.regression_head[0].out_features),
            "temp_mean": self.temp_mean,
            "temp_std": self.temp_std,
            "conc_mean": self.conc_mean,
            "conc_std": self.conc_std,
            "uses_additives": True,
            "results": {},
        }

        torch.save(payload, ckpt_path)

        models_dir_msg = ckpt_path.parent
        print(f"   Model saved to: {models_dir_msg}")
        print("      - pCMC_model_transformer_additives.pt")

    def _default_checkpoint_path(self) -> Path:
        models_dir = Path("cache") / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir / DEFAULT_CKPT_NAME

    def train(
        self,
        samples: AnnotatedSurfactantsDf,
        val_samples: AnnotatedSurfactantsDf | None = None,
    ):
        """Train model on `samples` with optional validation set for early stopping.

        This is a trimmed adaptation of the logic in `transformer/main_additives.py`.
        Only essential steps kept; no plotting or file output.
        """

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        required_cols = [
            "surfactant_smiles",
            "temperature",
            "pcmc",
            "additive_smiles",
            "additive_concentration",
        ]
        for c in required_cols:
            if c not in samples.df.columns:
                raise ValueError(f"Missing required column '{c}' for training.")

        df_train = samples.df.copy()
        df_val = val_samples.df.copy() if val_samples is not None else None

        # Extract columns
        def _prep(df):
            smiles = df["surfactant_smiles"].astype(str).tolist()
            temps = df["temperature"].to_numpy(dtype=float)
            y = df["pcmc"].to_numpy(dtype=float)
            add_smiles_raw = df["additive_smiles"].astype("string")
            add_smiles = [s if (s is not None and str(s) != "<NA>") else "" for s in add_smiles_raw]
            conc_vals = df["additive_concentration"].to_numpy()
            return smiles, temps, y, add_smiles, conc_vals

        (train_smiles, train_temps, train_y, train_add_smiles, train_conc) = _prep(df_train)
        if df_val is not None:
            (val_smiles, val_temps, val_y, val_add_smiles, val_conc) = _prep(df_val)
        else:
            val_smiles = []
            val_temps = np.array([], dtype=float)
            val_y = np.array([], dtype=float)
            val_add_smiles = []
            val_conc = np.array([], dtype=float)

        head_lr = self.head_lr
        transformer_lr = self.transformer_lr
        n_epochs = self.epochs
        batch_size = self.batch_size

        # Tokenization
        tok_train_main = self.tokenizer(
            train_smiles, padding=True, return_tensors="pt", truncation=True, max_length=512
        )
        if df_val is not None:
            tok_val_main = self.tokenizer(val_smiles, padding=True, return_tensors="pt", truncation=True, max_length=512)
        else:
            tok_val_main = None
        tok_train_add = self.tokenizer(
            [s if (s is not None and str(s).strip() != "") else "" for s in train_add_smiles],
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        if df_val is not None:
            tok_val_add = self.tokenizer(
                [s if (s is not None and str(s).strip() != "") else "" for s in val_add_smiles],
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
        else:
            tok_val_add = None

        def _mask_smiles(lst):
            return np.array(
                [1.0 if (s is not None and str(s).strip() != "") else 0.0 for s in lst], dtype=np.float32
            ).reshape(-1, 1)

        def _mask_conc(arr):
            return np.array(
                [0.0 if (c is None or (isinstance(c, float) and np.isnan(c))) else 1.0 for c in arr], dtype=np.float32
            ).reshape(-1, 1)

        train_add_embed_mask = _mask_smiles(train_add_smiles)
        train_add_conc_mask = _mask_conc(train_conc)
        if df_val is not None:
            val_add_embed_mask = _mask_smiles(val_add_smiles)
            val_add_conc_mask = _mask_conc(val_conc)
        else:
            val_add_embed_mask = None
            val_add_conc_mask = None

        # Scaling
        temp_scaler = StandardScaler()
        train_temps_scaled = temp_scaler.fit_transform(train_temps)
        if df_val is not None:
            val_temps_scaled = temp_scaler.transform(val_temps)
        else:
            val_temps_scaled = None
        if temp_scaler.std_ is None or temp_scaler.mean_ is None:
            raise RuntimeError("Temperature scaler not fitted properly.")
        self.temp_mean = float(temp_scaler.mean_)
        self.temp_std = float(temp_scaler.std_)

        present_train_conc = np.array(
            [c for c in train_conc if not (c is None or (isinstance(c, float) and np.isnan(c)))], dtype=float
        )
        if present_train_conc.size == 0:
            self.conc_mean, self.conc_std = 0.0, 1.0
        else:
            self.conc_mean = float(np.mean(present_train_conc))
            self.conc_std = float(np.std(present_train_conc))
            if self.conc_std == 0.0:
                self.conc_std = 1.0

        train_conc_scaled = scale_concentrations(train_conc, self.conc_mean, self.conc_std)
        val_conc_scaled = scale_concentrations(val_conc, self.conc_mean, self.conc_std) if df_val is not None else None

        # Datasets
        def _collate(bl):
            return collate_Xy_to_device(bl, self.device)

        ds_train = CMCDatasetAdditives(
            input_ids=tok_train_main["input_ids"],
            attention_mask=tok_train_main["attention_mask"],
            temps_scaled=train_temps_scaled,
            targets=train_y,
            add_input_ids=tok_train_add["input_ids"],
            add_attention_mask=tok_train_add["attention_mask"],
            add_conc_scaled=train_conc_scaled,
            add_embed_mask=train_add_embed_mask,
            add_conc_mask=train_add_conc_mask,
        )
        ds_val = None
        if df_val is not None:
            ds_val = CMCDatasetAdditives(
                input_ids=tok_val_main["input_ids"],
                attention_mask=tok_val_main["attention_mask"],
                temps_scaled=val_temps_scaled,
                targets=val_y,
                add_input_ids=tok_val_add["input_ids"],
                add_attention_mask=tok_val_add["attention_mask"],
                add_conc_scaled=val_conc_scaled,
                add_embed_mask=val_add_embed_mask,
                add_conc_mask=val_add_conc_mask,
            )

        g = torch.Generator().manual_seed(42)
        dl_train = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=_collate,
            generator=g,
        )
        dl_val = None
        if ds_val is not None:
            dl_val = DataLoader(
                ds_val,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                collate_fn=_collate,
                generator=g,
            )

        crit = nn.HuberLoss(delta=0.5, reduction="mean")
        if self.model.transformer_main is self.model.transformer_add:
            opt = torch.optim.Adam(
                [
                    {"params": self.model.transformer_main.parameters(), "lr": transformer_lr},
                    {"params": self.model.regression_head.parameters(), "lr": head_lr},
                ]
            )
        else:
            opt = torch.optim.Adam(
                [
                    {"params": self.model.transformer_main.parameters(), "lr": transformer_lr},
                    {"params": self.model.transformer_add.parameters(), "lr": transformer_lr},
                    {"params": self.model.regression_head.parameters(), "lr": head_lr},
                ]
            )

        if dl_val is not None:
            results, _ = _train_loop(
                model=self.model,
                train_loader=dl_train,
                val_loader=dl_val,
                criterion=crit,
                optimizer=opt,
                device=self.device,
                n_epochs=n_epochs,
                patience=self.patience,
            )
            print(
                f"Finished training: best_epoch={results.get('best_epoch')}, val_r2={results.get('val_r2'):.4f}, val_mae={results.get('val_mae'):.4f}"
            )
            ckpt_path = self._default_checkpoint_path()
            self.save(ckpt_path)
            return {"epochs": results["best_epoch"], "checkpoint": ckpt_path}
        else:
            results = _train_only(
                model=self.model,
                train_loader=dl_train,
                criterion=crit,
                optimizer=opt,
                device=self.device,
                n_epochs=n_epochs,
            )
            print(f"Finished training (no validation). Ran {n_epochs} epochs.")
            ckpt_path = self._default_checkpoint_path()
            self.save(ckpt_path)
            return {"epochs": n_epochs, "checkpoint": ckpt_path}

    def predict(self, samples: SurfactantsDf) -> list[float]:
        import numpy as np
        import torch

        if (
            self.model is None
            or self.tokenizer is None
            or self.temp_mean is None
            or self.temp_std is None
            or self.conc_mean is None
            or self.conc_std is None
            or self.device is None
        ):
            raise RuntimeError("Model is not loaded. Call load() first.")

        df = samples.df
        smiles_list = df["surfactant_smiles"].astype(str).tolist()
        temperatures = df["temperature"].to_numpy(dtype=float)

        # Additives
        add_smiles_series = df["additive_smiles"] if "additive_smiles" in df.columns else None
        add_conc_series = df["additive_concentration"] if "additive_concentration" in df.columns else None

        if add_smiles_series is not None:
            add_smiles_raw = add_smiles_series.astype("string").tolist()
            additive_smiles = [s if (s is not None and str(s) != "<NA>") else "" for s in add_smiles_raw]
        else:
            additive_smiles = [""] * len(smiles_list)

        if add_conc_series is not None:
            add_conc_vals = add_conc_series.to_numpy()
        else:
            add_conc_vals = np.array([np.nan] * len(smiles_list), dtype=float)

        # Scale temperature
        temps_scaled = (temperatures - self.temp_mean) / self.temp_std

        # Scale concentration with masking
        conc_scaled = []
        for c in add_conc_vals:
            if c is None or (isinstance(c, float) and math.isnan(c)):
                conc_scaled.append(0.0)
            else:
                conc_scaled.append((float(c) - self.conc_mean) / self.conc_std)
        conc_scaled = np.array(conc_scaled, dtype=np.float32)

        # Masks
        add_embed_mask_np = np.array(
            [1.0 if (s is not None and str(s).strip() != "") else 0.0 for s in additive_smiles], dtype=np.float32
        ).reshape(-1, 1)
        add_conc_mask_np = np.array(
            [0.0 if (c is None or (isinstance(c, float) and math.isnan(c))) else 1.0 for c in add_conc_vals],
            dtype=np.float32,
        ).reshape(-1, 1)

        # Batched inference matching training shapes
        preds_out: list[float] = []
        n = len(smiles_list)
        for i in range(0, n, self.batch_size):
            sl = smiles_list[i : i + self.batch_size]
            tl = temps_scaled[i : i + self.batch_size]
            add_sl = additive_smiles[i : i + self.batch_size]
            add_cl = conc_scaled[i : i + self.batch_size]
            add_embed_mask_b = add_embed_mask_np[i : i + self.batch_size]
            add_conc_mask_b = add_conc_mask_np[i : i + self.batch_size]

            main_inputs = self.tokenizer(sl, padding=True, return_tensors="pt", truncation=True, max_length=512)
            add_inputs = self.tokenizer(add_sl, padding=True, return_tensors="pt", truncation=True, max_length=512)

            # Move to device
            main_inputs = {k: v.to(self.device) for k, v in main_inputs.items()}
            add_inputs = {k: v.to(self.device) for k, v in add_inputs.items()}
            temp_t = torch.tensor(tl, dtype=torch.float32, device=self.device).reshape(-1, 1)
            add_conc_t = torch.tensor(add_cl, dtype=torch.float32, device=self.device).reshape(-1, 1)
            add_embed_mask_t = torch.tensor(add_embed_mask_b, dtype=torch.float32, device=self.device)
            add_conc_mask_t = torch.tensor(add_conc_mask_b, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                batch_preds = self.model(
                    {
                        "input_ids": main_inputs["input_ids"],
                        "attention_mask": main_inputs["attention_mask"],
                        "temperature": temp_t,
                        "add_input_ids": add_inputs["input_ids"],
                        "add_attention_mask": add_inputs["attention_mask"],
                        "add_conc": add_conc_t,
                        "add_embed_mask": add_embed_mask_t,
                        "add_conc_mask": add_conc_mask_t,
                    }
                )
            preds_out.extend(batch_preds.detach().cpu().numpy().tolist())

        return [float(x) for x in preds_out]
