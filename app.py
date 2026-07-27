import streamlit as st
import pandas as pd
import joblib
import os
import sys
from pathlib import Path
from datetime import datetime
import uuid

# 1. SETUP: Add repo root to path so the pickle loader can find 'cli' module classes
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# File Constants
MODEL_FILE = APP_DIR / "lgbm-2026-01-08.pkl"
HISTORY_FILE = APP_DIR / "prediction_history.csv"


def _patched_dataset_setup(df: pd.DataFrame) -> pd.DataFrame:
    """Patched version of _initial_dataset_setup that works across pandas versions."""
    feature_cols = df.columns.tolist()
    if "surfactant_smiles" in feature_cols:
        feature_cols.remove("surfactant_smiles")
    if "additive_smiles" in feature_cols:
        feature_cols.remove("additive_smiles")

    if "surfactant_type" in feature_cols:
        # Convert Categorical to plain strings to avoid pandas compatibility issues
        if isinstance(df["surfactant_type"].dtype, pd.CategoricalDtype):
            categories = df["surfactant_type"].cat.categories.tolist()
            df["surfactant_type"] = df["surfactant_type"].astype(str)
        else:
            categories = None
        one_hot = pd.get_dummies(df["surfactant_type"], prefix="surfactant_type", dummy_na=True)
        if categories is not None:
            for cat in categories:
                col_name = f"surfactant_type_{cat}"
                if col_name not in one_hot.columns:
                    one_hot[col_name] = 0
        df.drop("surfactant_type", axis=1, inplace=True)
        df[one_hot.columns] = one_hot
        feature_cols += one_hot.columns.tolist()
        if "surfactant_type" in feature_cols:
            feature_cols.remove("surfactant_type")

    return df[feature_cols]


# 2. HELPER FUNCTIONS
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        model = joblib.load(MODEL_FILE)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

    # Patch the preprocess FunctionTransformer to work across pandas versions
    try:
        from sklearn.preprocessing import FunctionTransformer
        # model.model.model = ModelWrapper -> SklearnModel -> Pipeline
        pipeline = model.model.model
        if hasattr(pipeline, "named_steps") and "preprocess" in pipeline.named_steps:
            pipeline.named_steps["preprocess"] = FunctionTransformer(
                _patched_dataset_setup
            )
    except Exception:
        pass

    return model


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except:
            return pd.DataFrame()
    # Return empty structure if file doesn't exist
    return pd.DataFrame(columns=[
        "ID", "Timestamp", "Surfactant", "Temperature",
        "Additive", "Conc", "Predicted_pCMC", "Predicted_CMC_M",
        "Feedback", "Actual_pCMC"
    ])

def save_history(df):
    df.to_csv(HISTORY_FILE, index=False)

def add_to_history(surfactant, temp, additive, conc, pred_pcmc):
    new_row = {
        "ID": str(uuid.uuid4())[:8],
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Surfactant": surfactant,
        "Temperature": temp,  # Saved as raw float (no rounding)
        "Additive": additive if additive else "None",
        "Conc": conc,
        "Predicted_pCMC": round(pred_pcmc, 4),
        "Predicted_CMC_M": f"{10**(-pred_pcmc):.6e}",
        "Feedback": "",
        "Actual_pCMC": ""
    }
    df = load_history()
    # Use pandas concat instead of append
    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
    save_history(df)

# 3. PAGE CONFIG
st.set_page_config(page_title="Surfactant Predictor & History", page_icon="🧪", layout="wide")

st.title("Surfactant pCMC Predictor")

# Load the model
model = load_model()

if not model:
    st.error(f"Model file '{MODEL_FILE}' not found in {APP_DIR}")
    st.stop()

# 4. TABS LAYOUT
tab1, tab2 = st.tabs(["New Prediction", "History & Feedback"])

# --- TAB 1: PREDICTION ---
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Parameters")
        surfactant_input = st.text_area(
            "Surfactant SMILES",
            value="CCCCCCC/C=C\\CCCC(O)CCCC(=O)[O-].[K+]",
            height=100
        )
        temperature_input = st.number_input(
            "Temperature (°C)",
            value=55.0,
            format="%.8f",
            step=0.000001
        )

    with col2:
        st.subheader("Additive (Optional)")
        has_additive = st.checkbox("Include Additive")
        additive_input = None
        additive_conc_input = 0.0

        if has_additive:
            additive_input = st.text_input("Additive SMILES", value="O")
            additive_conc_input = st.number_input("Additive Concentration", value=0.1)

    if st.button("Run Prediction", type="primary"):
        if not surfactant_input:
            st.warning("Please enter a Surfactant SMILES string.")
        else:
            with st.spinner("Predicting..."):
                try:
                    input_data = pd.DataFrame(
                        [
                            [
                                surfactant_input,
                                temperature_input,
                                additive_input,
                                additive_conc_input
                            ],
                        ],
                        columns=[
                            "surfactant_smiles",
                            "temperature",
                            "additive_smiles",
                            "additive_concentration",
                        ],
                    )
                    # Force object dtype for string columns to satisfy typedframe schema
                    for col in ["surfactant_smiles", "additive_smiles"]:
                        input_data[col] = input_data[col].astype("object")

                    prediction = model.predict(input_data)[0]

                    st.success("Prediction Complete!")
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric("Predicted pCMC", f"{prediction:.4f}")
                    with res_col2:
                        cmc_molar = 10**(-prediction)
                        st.metric("Estimated CMC (M)", f"{cmc_molar:.6e}")

                    add_to_history(
                        surfactant_input,
                        temperature_input,
                        additive_input,
                        additive_conc_input,
                        prediction
                    )
                    st.toast("Result saved to History!")

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.info("Ensure 'cli' and dependencies (rdkit, typedframe) are installed.")

# --- TAB 2: HISTORY & FEEDBACK ---
with tab2:
    st.header("Prediction Log")
    st.markdown("View past queries and provide feedback on accuracy.")

    history_df = load_history()

    if history_df.empty:
        st.info("No history yet. Make a prediction in the first tab!")
    else:
        edited_df = st.data_editor(
            history_df,
            column_config={
                "ID": st.column_config.TextColumn(disabled=True),
                "Timestamp": st.column_config.TextColumn(disabled=True),
                "Surfactant": st.column_config.TextColumn(disabled=True, width="medium"),
                "Temperature": st.column_config.NumberColumn(
                    disabled=True,
                    format="%.6f"
                ),
                "Additive": st.column_config.TextColumn(disabled=True),
                "Conc": st.column_config.NumberColumn(disabled=True),
                "Predicted_pCMC": st.column_config.NumberColumn(disabled=True),
                "Predicted_CMC_M": st.column_config.TextColumn(disabled=True),
                "Feedback": st.column_config.SelectboxColumn(
                    "Feedback",
                    options=["", "Good", "Overestimated", "Underestimated", "Bad"],
                    required=False
                ),
                "Actual_pCMC": st.column_config.NumberColumn(
                    "Actual pCMC (if known)",
                    min_value=0.0,
                    max_value=10.0,
                    step=0.0001
                )
            },
            hide_index=True,
            use_container_width=True,
            num_rows="fixed"
        )

        if st.button("Save Feedback"):
            save_history(edited_df)
            st.success("Feedback updated successfully!")
