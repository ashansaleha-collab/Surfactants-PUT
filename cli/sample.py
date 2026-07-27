from dataclasses import dataclass


@dataclass
class Additive:
    smiles: str
    concentration: float  # TODO: Unit?


@dataclass
class Sample:
    surfactant_smiles: str

    temperature: float
    """temperature, in Celsius"""

    additive: Additive | None
    """additive, if any"""

    pcmc: float | None = None
    """pCMC (annotation)"""
