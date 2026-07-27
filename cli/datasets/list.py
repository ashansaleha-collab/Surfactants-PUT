from . import Dataset
from .example import ExampleDataset
from .expert import ExpertDataset
from .lab import LabDataset
from .merged import MergedDataset
from .paper1_dedup import Paper1DedupDataset
from .paper4_dedup import Paper4DedupDataset


def datasets() -> dict[str, type[Dataset]]:
    """
    Returns a dictionary mapping dataset names (str) to the
    dataset class (not an instance).
    """
    return {
        "example": ExampleDataset,
        "expert": ExpertDataset,
        "lab": LabDataset,
        "paper1": lambda: Paper1DedupDataset(dedup=False),
        "paper1_dedup": lambda: Paper1DedupDataset(dedup=True),
        "paper4": lambda: Paper4DedupDataset(dedup=False),
        "paper4_dedup": lambda: Paper4DedupDataset(dedup=True),
        "everything": lambda: MergedDataset(
            [
                create_dataset("expert"),
                create_dataset("paper1"),
                create_dataset("paper4"),
            ],
        ),
    }


def create_dataset(dataset_name: str) -> Dataset:
    """Create dataset instance by name."""
    dataset_cls = datasets().get(dataset_name)
    if dataset_cls is None:
        msg = f"Unknown dataset: {dataset_name}"
        raise ValueError(msg)

    # The create_dataset function is responsible for calling
    # the constructor (with parentheses)
    return dataset_cls()
