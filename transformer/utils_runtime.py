import os
import random
import torch
from transformers import AutoModel, AutoTokenizer


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, Torch and enforce deterministic CUDA behavior.

    This function now also sets flags/environments to reduce nondeterminism:
      - CUBLAS workspace config for deterministic GEMMs
      - Disable TF32 matmul convolutions for consistency
      - Enforce deterministic algorithms via torch.use_deterministic_algorithms
      - cuDNN deterministic + disable benchmark autotuning
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    random.seed(seed)
    try:
        import numpy as np  # local import to avoid hard dep here
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # cuDNN deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enforce deterministic algorithms (raise if unsupported op appears)
    torch.use_deterministic_algorithms(True)

    # Force deterministic attention backend: prefer math SDPA, disable flash/mem-efficient
#     if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_flash_sdp"):
#         torch.backends.cuda.enable_flash_sdp(False)
#         torch.backends.cuda.enable_mem_efficient_sdp(False)
#         torch.backends.cuda.enable_math_sdp(True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_transformer(model_name: str, device: torch.device):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer
