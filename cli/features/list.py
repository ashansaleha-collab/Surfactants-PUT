from . import FeatureExtractor
from .chen import ChenExtractor
from .expert import ExpertDataExtractor
from .fingerprints import FingerprintsExtractor


def feature_extractors():
    # fmt: off
    return {
        "chen": ChenExtractor,
        "estate": lambda: FingerprintsExtractor(fp_type="estate"),
        "estate0": lambda: FingerprintsExtractor(fp_type="estate0"),
        "expert": ExpertDataExtractor,
        "morgan": lambda: FingerprintsExtractor(fp_type="morgan"),
        "atompair": lambda: FingerprintsExtractor(fp_type="atompair"),
        "autocorr": lambda: FingerprintsExtractor(fp_type="autocorr"),
        "avalon": lambda: FingerprintsExtractor(fp_type="avalon"),
        "bcut2d": lambda: FingerprintsExtractor(fp_type="bcut2d"),
        "e3fp": lambda: FingerprintsExtractor(fp_type="e3fp"),
        "ecfp": lambda: FingerprintsExtractor(fp_type="ecfp"),
        "electroshape": lambda: FingerprintsExtractor(fp_type="electroshape"),
        "erg": lambda: FingerprintsExtractor(fp_type="erg"),
        "functionalgroups": lambda: FingerprintsExtractor(fp_type="functionalgroups"),
        "getaway": lambda: FingerprintsExtractor(fp_type="getaway"),
        "ghosecrippen": lambda: FingerprintsExtractor(fp_type="ghosecrippen"),
        # "klekotaroth": lambda: FingerprintsExtractor(fp_type="klekotaroth"), # TOO SLOW
        "laggner": lambda: FingerprintsExtractor(fp_type="laggner"),
        "layered": lambda: FingerprintsExtractor(fp_type="layered"),
        "lingo": lambda: FingerprintsExtractor(fp_type="lingo"),
        "maccs": lambda: FingerprintsExtractor(fp_type="maccs"),
        "map": lambda: FingerprintsExtractor(fp_type="map"),
        "mhfp": lambda: FingerprintsExtractor(fp_type="mhfp"),
        # "mordred": lambda: FingerprintsExtractor(fp_type="mordred"), # TOO SLOW
        "morse": lambda: FingerprintsExtractor(fp_type="morse"),
        "mqns": lambda: FingerprintsExtractor(fp_type="mqns"),
        "pattern": lambda: FingerprintsExtractor(fp_type="pattern"),
        "pharmacophore": lambda: FingerprintsExtractor(fp_type="pharmacophore"),
        "physiochemicalproperties": lambda: FingerprintsExtractor(fp_type="physiochemicalproperties"),
        "pubchem": lambda: FingerprintsExtractor(fp_type="pubchem"),
        "rdf": lambda: FingerprintsExtractor(fp_type="rdf"),
        "rdkit": lambda: FingerprintsExtractor(fp_type="rdkit"),
        "rdkit2ddescriptors": lambda: FingerprintsExtractor(fp_type="rdkit2ddescriptors"),
        "secfp": lambda: FingerprintsExtractor(fp_type="secfp"),
        "topologicaltorsion": lambda: FingerprintsExtractor(fp_type="topologicaltorsion"),
        "usr": lambda: FingerprintsExtractor(fp_type="usr"),
        "usrcat": lambda: FingerprintsExtractor(fp_type="usrcat"),
        "vsa": lambda: FingerprintsExtractor(fp_type="vsa"),
        "whim": lambda: FingerprintsExtractor(fp_type="whim"),
    }
    # fmt: on


def create_feature_extractor(name: str) -> FeatureExtractor:
    """Create feature extractor instance by name."""
    model_cls = feature_extractors().get(name)
    if model_cls is None:
        msg = f"Unknown feature extractor: {name}"
        raise ValueError(msg)
    return model_cls()
