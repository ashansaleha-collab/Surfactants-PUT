import streamlit as st
import pandas as pd
import joblib
import os
import sys
from datetime import datetime
import uuid

# 1. SETUP: Add current directory to path
# This is crucial so the pickle loader can find the 'cli' module classes
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

# File Constants
MODEL_FILE = "lgbm-2026-01-08.pkl"
HISTORY_FILE = "prediction_history.csv"

# 2. HELPER FUNCTIONS
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None
    return None

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
    st.error(f"Model file '{MODEL_FILE}' not found in {current_dir}")
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
        # UPDATED: Added format="%.6f" to prevent rounding in the input box
        temperature_input = st.number_input(
            "Temperature (°C)",
            value=55.0,
            format="%.8f",  # Allows up to 6 decimal places
            step=0.000001   # Allows fine-grained increments
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
                    # Prepare Dataframe matching your snippet exactly
                    input_data = pd.DataFrame(
                        [
                            [
                                surfactant_input,
                                temperature_input,
                                additive_input, # Passed as None if not selected
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

                    # Predict
                    prediction = model.predict(input_data)[0]

                    # Display Results
                    st.success("Prediction Complete!")
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric("Predicted pCMC", f"{prediction:.4f}")
                    with res_col2:
                        cmc_molar = 10**(-prediction)
                        st.metric("Estimated CMC (M)", f"{cmc_molar:.6e}")

                    # Log to History
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
        # Use Data Editor to allow changes to 'Feedback' and 'Actual_pCMC'
        # We disable editing for the input/result columns to preserve integrity
        edited_df = st.data_editor(
            history_df,
            column_config={
                "ID": st.column_config.TextColumn(disabled=True),
                "Timestamp": st.column_config.TextColumn(disabled=True),
                "Surfactant": st.column_config.TextColumn(disabled=True, width="medium"),

                # UPDATED: Added format="%.6f" to display full temperature precision in history
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

        # Save button (Streamlit reruns on edit, but explicit save is clearer)
        if st.button("Save Feedback"):
            save_history(edited_df)
            st.success("Feedback updated successfully!")
