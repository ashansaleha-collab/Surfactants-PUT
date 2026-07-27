from io import StringIO

import pandas as pd

from cli.data import SurfactantsDf

from . import Dataset


class ExampleDataset(Dataset):
    """Dataset with some randomized samples (taken from expert data)."""

    def samples(self) -> SurfactantsDf:
        # And yes, I just put a part of CMC_surfactants_v2_4.csv here.
        csv = r"""SMILES,Tail (C number),Tail (SMILES),Head (SMILES),Head 2 (SMILES),Head 3 (SMILES),Counterion,Additives,Conc. A.,pH,IUPAC_Name,Preferred_Name,Surfactant_Type,Molecular_Formula,Molecular_Weight,CMC,pCMC,Temp_Celsius
CCC(CC)(CC)[Si]O[Si](C)(CCSCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO)O[Si]C(CC)(CC)CC,,CCC(CC)(CC)[Si]O[Si](C)()O[Si]C(CC)(CC)CC,CCSCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO,,,,,,,,,non-ionic,,1094.61,0.0009,3.04575749056068,25.0
CCC(CC)COCCOCCOCCOCCOCCOCCO,4.0,CCC(CC)C,OCCOCCOCCOCCOCCOCCO,,,,,,,,,non-ionic,,366.26,,,
CCC=COOCCOCCOCCOCCOCCOCC,4.0,CCC=C,OOCCOCCOCCOCCOCCOCC,,,,,,,,,non-ionic,,336.21,0.8933,0.049002666011195,25.0
CCCC(CC(F)C(F)(F)F)OS(=O)(=O)[O-].[Na+],6.0,CCCC(CC(F)C(F)(F)F),OS(=O)(=O)[O-],,,[Na+],,,,"sodium [4-(1,1,1-trifluoro-2-fluoroethyl)butyl] sulfate",sodium C6 secondary sulfate (fluoro-branched),anionic,C7H11F4NaO4S,290.02,0.025,1.60205999132796,25.0
CCCC(CCC)(CCC)[Si]O[Si](C)(CCSCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO)O[Si]C(CCC)(CCC)CCC,,CCCC(CCC)(CCC)[Si]O[Si](C)()O[Si]C(CCC)(CCC)CCC,CCSCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO,,,,,,,,,non-ionic,,1178.7,0.0002,3.69897000433602,25.0
CCCC(CCC)C(O)OCCOCCOCCOCCOCCOCC,7.0,CCCC(CCC)C,(O)OCCOCCOCCOCCOCCOCC,,,,,,,,,non-ionic,,394.29,0.023,1.63827216398241,25.0
CCCC(CCC)COCCOCCOCCOCCOCCOCCO,7.0,CCCC(CCC)C,OCCOCCOCCOCCOCCOCCO,,,,,,,,,non-ionic,,394.29,,,
CCCC(F)(OS(=O)(=O)[O-])C(F)(F)C(F)(F)C(F)(F)F.[Na+],7.0,CCCC(F)()C(F)(F)C(F)(F)C(F)(F)F,OS(=O)(=O)[O-],,,[Na+],,,,"sodium 4-fluoro-4-(1,1,2,2,3,3,3-heptafluoropropyl)butyl sulfate",sodium C4 secondary sulfate (fluoro-branched),anionic,C7H7F8NaO4S,361.98,0.0004699,3.32799455497705,25.0
CCCC(O)OCCOCCOCCOCCOCCOCC,4.0,CCCC,(O)OCCOCCOCCOCCOCCOCC,,,,,,,,,non-ionic,,338.23,0.8,0.0969100130080564,25.0
CCCC(OS(=O)(=O)[O-])C(F)C(F)(F)C(F)(F)F.[Na+],7.0,CCCC()C(F)C(F)(F)C(F)(F)F,OS(=O)(=O)[O-],,,[Na+],,,,"sodium 4-(1,1,2,2,3,3-hexafluoropropyl)butyl sulfate",,anionic,C7H9F6NaO4S,326.0,0.003999,2.39804859586648,25.0
CCCC=NCCC[N+](C)(C)CCCCCCCCCC.[Br-],10.0,CCCC=NCCC()CCCCCCCCCC,[N+](C)(C),,,[Br-],,,,"N-(3-(butylidenamino)propyl)-N,N-dimethyldecan-1-ammonium","N-(3-(butylidenamino)propyl)-N,N-dimethyldecan-1-ammonium",cationic,,376.25,0.001589,2.79887610279262,25.0
CCCC=NCCC[N+](C)(C)CCCCCCCCCCCC.[Br-],12.0,CCCC=NCCC()CCCCCCCCCCCC,[N+](C)(C),,,[Br-],,,,"N-(3-(butylidenamino)propyl)-N,N-dimethyldodecan-1-ammonium","N-(3-(butylidenamino)propyl)-N,N-dimethyldodecan-1-ammonium",cationic,,404.28,0.000771,3.11294562194904,25.0
CCCC=NCCC[N+](C)(C)CCCCCCCCCCCCCCCC.[Br-],16.0,CCCC=NCCC()CCCCCCCCCCCCCCCC,[N+](C)(C),,,[Br-],,,,"N-(3-(butylidenamino)propyl)-N,N-dimethylhexadecan-1-ammonium","N-(3-(butylidenamino)propyl)-N,N-dimethylhexadecan-1-ammonium",cationic,,460.34,8.41e-05,4.07520400420209,25.0
CCCCC(=O)OCC(COC(=O)CCCC)S(=O)(=O)[O-].[Na+],5.0,CCCCC(=O)OCC(COC(=O)CCCC),S(=O)(=O)[O-],,,[Na+],,,,sodium bis(pentanoyl)–2-sulfonatosuccinate,,anionic,C13H23NaO7S,346.11,0.05768,1.23897474828863,25.0
CCCCC(CC)CC(=O)OCC(COC(=O)CC(CC)CCCC)S(=O)(=O)[O-].[Na+],8.0,CCCCC(CC)CC(=O)OCC(COC(=O)CC(CC)CCCC),S(=O)(=O)[O-],,,[Na+],,,,sodium bis(2-ethylhexyl)-2-sulfonatosuccinate,,anionic,C20H37NaO7S,458.23,0.2173,0.662940273679475,25.0
CCCCC(CC)COC(=O)CC(C(=O)OCC(CC)CCCC)S(=O)(=O)[O-].[Na+],8.0,CCCCC(CC)COC(=O)CC(C(=O)OCC(CC)CCCC),S(=O)(=O)[O-],,,[Na+],,,,"sodium-1,4-bis(2-ethylhexoxy)-1,4-dioxobutane-2-sulfonate",Docusate sodium,anionic,C20H37NaO7S,444.22,0.0025,2.60205999132796,25.0
CCCCC(CCCC)C(O)OCCOCCOCCOCCOCCOCC,9.0,CCCCC(CCCC)C,(O)OCCOCCOCCOCCOCCOCC,,,,,,,,,non-ionic,,422.32,0.0031,2.50863830616573,25.0
CCCCCCCCC(=O)[O-].[K+],9.0,CCCCCCCC,C(=O)[O-],,,[K+],[Cl-].[K+],0.677,,potassium nonanoate,Potassium nonanoate,anionic,C9H17KO2,196.09,0.097,1.01322826573376,25.0
CCCCCCCCC(=O)[O-].[K+],9.0,CCCCCCCC,C(=O)[O-],,,[K+],[Cl-].[K+],0.855,,potassium nonanoate,Potassium nonanoate,anionic,C9H17KO2,196.09,0.096,1.01772876696043,25.0
CCCCCCCCC(=O)[O-].[K+],9.0,CCCCCCCC,C(=O)[O-],,,[K+],[Cl-].[K+],1.29,,potassium nonanoate,Potassium nonanoate,anionic,C9H17KO2,196.09,0.064,1.19382002601611,25.0
CCCCCCCCC(=O)[O-].[K+],9.0,CCCCCCCC,C(=O)[O-],,,[K+],[Cl-].[K+],1.63,,potassium nonanoate,Potassium nonanoate,anionic,C9H17KO2,196.09,0.054,1.26760624017703,25.0
CCCCCCCCC(=O)[O-].[K+],9.0,CCCCCCCC,C(=O)[O-],,,[K+],[Cl-].[K+],1.92,,potassium nonanoate,Potassium nonanoate,anionic,C9H17KO2,196.09,0.048,1.31875876262441,25.0
CCCCCCCCC(=O)[O-].[K+],9.0,CCCCCCCC,C(=O)[O-],,,[K+],[Cl-].[K+],2.11,,potassium nonanoate,Potassium nonanoate,anionic,C9H17KO2,196.09,0.042,1.3767507096021,25.0
CCCCCCCCC(=O)[O-].[K+],9.0,CCCCCCCC,C(=O)[O-],,,[K+],[Cl-].[K+],2.39,,potassium nonanoate,Potassium nonanoate,anionic,C9H17KO2,196.09,0.04,1.39794000867204,25.0
CCCCCCCCC(=O)[O-].[K+],9.0,CCCCCCCC,C(=O)[O-],,,[K+],[Cl-].[K+],2.64,,potassium nonanoate,Potassium nonanoate,anionic,C9H17KO2,196.09,0.038,1.42021640338319,25.0
"""

        df = pd.read_csv(StringIO(csv))
        # rename columns
        df = df.rename(
            columns={
                "SMILES": "surfactant_smiles",
                "Additives": "additive_smiles",
                "Conc. A.": "additive_concentration",
                "Temp_Celsius": "temperature",
                "pCMC": "pcmc",
            },
        )
        # temperature 25 by default
        df["temperature"] = df["temperature"].fillna(25.0)
        return SurfactantsDf(df)
