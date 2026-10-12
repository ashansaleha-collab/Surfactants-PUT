import streamlit as st
import pandas as pd
import numpy as np
import itertools
import joblib
import os
import sys
import traceback
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

# Sweep grid resolution (physical precisions agreed with the lab):
# every 0.5 °C, every 0.01 concentration units. There is deliberately no
# cell-count cap: the slider bounds (temp 0-100 °C, conc 0-1) cap the
# ranges instead (worst case ~201 x 101 ≈ 20k predictions).
TEMP_STEP = 0.5
CONC_STEP = 0.01
N_TOP_RESULTS = 10


def sweep_axis(lo: float, hi: float, step: float) -> list[float]:
    """Every value in [lo, hi] at the given precision, endpoint included.

    Rounds to the step's decimals so arange float dust (0.060000000004)
    never leaks into the grid.
    """
    decimals = max(0, -int(np.floor(np.log10(step))))
    return [round(float(v), decimals) for v in np.arange(lo, hi + step / 2, step)]



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


def show_error(prefix: str, e: Exception):
    """User-friendly error + full traceback for troubleshooting.

    The short message is what users report; the expander holds the exact
    traceback to paste into a bug report.
    """
    st.error(f"{prefix}: {e}")
    with st.expander("Show error details (copy this when reporting)"):
        st.code(traceback.format_exc(), language="python")
    st.info("Ensure 'cli' and dependencies (rdkit, typedframe) are installed.")

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
                "Temperature min and max (°C)",
                min_value=0.0, max_value=100.0,
                value=(20.0, 60.0), step=0.5,
                key="temp_range",
            )
            temperature_input = (t_lo, t_hi)
            st.caption(f"Sweep precision: every {TEMP_STEP} °C.")

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
                    "Concentration min and max",
                    min_value=0.0, max_value=1.0,
                    value=(0.0, 0.5), step=0.01,
                    key="conc_range",
                )
                additive_conc_input = (c_lo, c_hi)
                st.caption(f"Sweep precision: every {CONC_STEP}.")

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
                        show_error("Prediction failed", e)
    else:
        # Full enumeration at physical precision (sliders guarantee lo <= hi
        # and conc >= 0, so no clamping needed). No cell cap: slider bounds
        # are the caps.
        t_lo, t_hi = temperature_input
        temps = sweep_axis(t_lo, t_hi, TEMP_STEP)
        if has_additive:
            c_lo, c_hi = additive_conc_input
            concs = sweep_axis(c_lo, c_hi, CONC_STEP)
        else:
            concs = [0.0]
        st.info(
            f"Sweep grid: {len(temps)} temperatures × {len(concs)} concentrations "
            f"= {len(temps) * len(concs):,} predictions at one click."
        )

        if st.button("Run Sweep", type="primary"):
            if not surfactant_input.strip():
                st.warning("Please enter a Surfactant SMILES string.")
            else:
                with st.spinner(f"Predicting {len(temps) * len(concs):,} grid points..."):
                    try:
                        rows = [
                            [surfactant_input.strip(), t, additive_input, c]
                            for t, c in itertools.product(temps, concs)
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
                        st.subheader(f"Top {N_TOP_RESULTS} conditions (ranked by pCMC)")
                        st.dataframe(
                            ranked.head(N_TOP_RESULTS), use_container_width=True
                        )

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
                            surfactant_input.strip(),
                            float(best["temperature"]),
                            additive_input,
                            float(best["additive_concentration"]),
                            float(best["predicted_pcmc"]),
                        )
                        st.toast("Best row saved to History!")
                    except Exception as e:
                        show_error("Sweep failed", e)

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
