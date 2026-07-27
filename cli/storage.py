from pathlib import Path

STORAGE_PATH = Path("data")


def model_path(name: str) -> Path:
    return STORAGE_PATH / "models" / name


def ensure_model_path(name: str) -> Path:
    path = model_path(name)
    path.mkdir(parents=True, exist_ok=True)
    return path
