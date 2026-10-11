import streamlit as st
import pandas as pd
import numpy as np
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
    mode = st.radio(
        "Prediction mode",
        ["Single point", "Range sweep (find best conditions)"],
        horizontal=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Parameters")
        surfactant_input = st.text_area(
            "Surfactant SMILES",
            value="CCCCCCC/C=C\\CCCC(O)CCCC(=O)[O-].[K+]",
            height=100
        )
        if mode == "Single point":
            temperature_input = st.number_input(
                "Temperature (°C)",
                value=55.0,
                format="%.8f",
                step=0.000001
            )
        else:
            st.markdown("**Temperature range (°C)**")
            t_lo, t_hi = st.slider(
                "Select min and max",
                min_value=0.0, max_value=100.0,
                value=(20.0, 60.0), step=0.5,
            )
            temperature_input = (t_lo, t_hi)
            n_steps = st.number_input(
                "Points per range (fixed step count)",
                min_value=2, max_value=50, value=20, step=1,
            )

    with col2:
        st.subheader("Additive (Optional)")
        has_additive = st.checkbox("Include Additive")
        additive_input = None
        additive_conc_input = 0.0

        if has_additive:
            additive_input = st.text_input("Additive SMILES", value="O")
            if mode == "Single point":
                additive_conc_input = st.number_input("Additive Concentration", value=0.1)
            else:
                st.markdown("**Additive concentration range**")
                c_lo, c_hi = st.slider(
                    "Select min and max",
                    min_value=0.0, max_value=1.0,
                    value=(0.0, 0.5), step=0.01,
                )
                additive_conc_input = (c_lo, c_hi)

    if mode == "Single point":
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
    else:
        if st.button("Run Sweep", type="primary"):
            if not surfactant_input:
                st.warning("Please enter a Surfactant SMILES string.")
            else:
                t_lo, t_hi = temperature_input
                if t_lo > t_hi:
                    t_lo, t_hi = t_hi, t_lo
                if has_additive:
                    c_lo, c_hi = additive_conc_input
                    if c_lo > c_hi:
                        c_lo, c_hi = c_hi, c_lo
                    if c_lo < 0:
                        st.warning("Negative concentration clamped to 0.")
                        c_lo = 0.0
                    concs = np.linspace(c_lo, c_hi, num=int(n_steps))
                else:
                    concs = np.array([0.0])
                temps = np.linspace(t_lo, t_hi, num=int(n_steps))

                if len(temps) * len(concs) > 2500:
                    st.error("Sweep grid too large (>2500 cells). Reduce points per range.")
                else:
                    with st.spinner(f"Predicting {len(temps) * len(concs)} grid points..."):
                        try:
                            rows = [
                                [surfactant_input, float(t), additive_input, float(c)]
                                for t in temps for c in concs
                            ]
                            sweep_df = pd.DataFrame(
                                rows,
                                columns=[
                                    "surfactant_smiles",
                                    "temperature",
                                    "additive_smiles",
                                    "additive_concentration",
                                ],
                            )
                            for col in ["surfactant_smiles", "additive_smiles"]:
                                sweep_df[col] = sweep_df[col].astype("object")

                            preds = model.predict(sweep_df)
                            sweep_df["predicted_pcmc"] = preds
                            sweep_df["predicted_cmc_M"] = [10 ** (-p) for p in preds]

                            best_idx = int(np.argmax(preds))
                            best = sweep_df.iloc[best_idx]
                            st.success("Sweep Complete — best = max pCMC (lowest CMC).")
                            b1, b2, b3, b4 = st.columns(4)
                            with b1:
                                st.metric("Best Temp (°C)", f"{best['temperature']:.4f}")
                            with b2:
                                st.metric("Best Conc", f"{best['additive_concentration']:.6f}")
                            with b3:
                                st.metric("Best pCMC", f"{best['predicted_pcmc']:.4f}")
                            with b4:
                                st.metric("Best CMC (M)", f"{best['predicted_cmc_M']:.6e}")

                            ranked = sweep_df.sort_values(
                                "predicted_pcmc", ascending=False
                            ).reset_index(drop=True)
                            st.subheader("Full sweep (ranked by pCMC)")
                            st.dataframe(ranked, use_container_width=True)

                            # Response curves through the optimum
                            st.subheader("Response curves through optimum")
                            at_best_conc = sweep_df[
                                sweep_df["additive_concentration"] == best["additive_concentration"]
                            ].sort_values("temperature")
                            st.line_chart(
                                at_best_conc.set_index("temperature")["predicted_pcmc"]
                            )
                            if has_additive:
                                at_best_temp = sweep_df[
                                    sweep_df["temperature"] == best["temperature"]
                                ].sort_values("additive_concentration")
                                st.line_chart(
                                    at_best_temp.set_index("additive_concentration")["predicted_pcmc"]
                                )

                            add_to_history(
                                surfactant_input,
                                float(best["temperature"]),
                                additive_input,
                                float(best["additive_concentration"]),
                                float(best["predicted_pcmc"]),
                            )
                            st.toast("Best row saved to History!")
                        except Exception as e:
                            st.error(f"Sweep failed: {str(e)}")
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
