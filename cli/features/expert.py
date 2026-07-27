import re

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from cli.data import SurfactantsDf

from . import FeatureExtractor


class ExpertDataExtractor(FeatureExtractor):
    """Feature extractor that adds features present in expert data.

    - Tail (C Number)
    - Surfactant_Type
    - Molecular_Weight
    """

    def _extract_tail_c_number(self, smiles: str):
        # For now just count n of leading C's
        n_c = re.match(r"^(C+)", smiles)
        return len(n_c.group(1)) if n_c else 0

    def _extract_surfactant_type(self, smiles_lower: str):
        # https://problem-classes.slack.com/archives/C09K8PCTX9A/p1761996617912169?thread_ts=1761825245.206479&cid=C09K8PCTX9A

        # 1. ANIONIC SURFACTANTS
        # Looks for: sulfate (SO4), sulfonate (SO3), carboxylate (COO-), phosphate
        anionic_patterns = [
            "s(=o)(=o)([o-])",  # Sulfate
            "s(=o)(=o)[o-]",  # Sulfonate
            "c(=o)[o-]",  # Carboxylate
            "p(=o)([o-])",  # Phosphate
            "[o-]",  # Generic anion
            "os(=o)(=o)[o-]",  # Sulfate ester
        ]

        has_anionic = any(pattern in smiles_lower for pattern in anionic_patterns)

        # 2. CATIONIC SURFACTANTS
        # Look for: quaternary ammonium [N+], protonated amine [NH3+]
        cationic_patterns = [
            "[n+]",  # Quaternary ammonium
            "[nh3+]",  # Protonated primary amine
            "[nh2+]",  # Protonated secondary amine
            "[nh+]",  # Protonated tertiary amine
        ]

        has_cationic = any(pattern in smiles_lower for pattern in cationic_patterns)

        # 3. ZWITTERIONIC SURFACTANTS
        # Has both positive and negative charges
        if has_anionic and has_cationic:
            return "zwitterionic"

        # 4. ANIONIC
        if has_anionic:
            return "anionic"

        # 5. CATIONIC
        if has_cationic:
            return "cationic"

        # 6. NONIONIC (default if no charges found)
        # Common nonionic patterns: polyethylene oxide, sugars, alcohols
        return "nonionic"

    def _extract_molecular_weight(self, smiles: str):
        # RDKIT

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        return ExactMolWt(mol)

    def extract(self, samples: SurfactantsDf) -> pd.DataFrame:
        expert_rows = []

        for idx, row in samples.df.iterrows():
            smiles = row["surfactant_smiles"]
            tail_c_number = self._extract_tail_c_number(smiles)
            surfactant_type = self._extract_surfactant_type(smiles.lower())
            molecular_weight = self._extract_molecular_weight(smiles)

            expert_rows.append(
                {
                    "tail_c_number": tail_c_number,
                    "surfactant_type": surfactant_type,
                    "molecular_weight": molecular_weight,
                },
            )

        df = pd.DataFrame(expert_rows).set_index(samples.df.index)

        # Surfactant type is categorical
        surfactant_types = [
            "zwitterionic",
            "anionic",
            "cationic",
            "nonionic",
        ]
        df["surfactant_type"] = pd.Categorical(
            df["surfactant_type"],
            categories=surfactant_types,
        )

        return df
