from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

CRITICAL_MODEL_DIR = BASE_DIR / "trained_critical_temperature_models"
CRITICAL_FEATURE_COLS = ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "K2O", "Na2O", "Si/Al"]
CRITICAL_SUMMARY_FILE = CRITICAL_MODEL_DIR / "critical_temperature_best_model_summary.csv"

DATASETS = {
    "newton": {
        "label": "Newtonian System",
        "file": BASE_DIR / "newton_raw.xlsx",
        "columns": ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "K2O", "Na2O", "Si_AI", "T", "viscosity"],
        "feature_cols": ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "K2O", "Na2O", "Si_AI", "T"],
        "fixed_model": "xgb",
        "model_dir": BASE_DIR / "trained_newton_models_drop3",
        "summary_file": BASE_DIR / "trained_newton_models_drop3" / "newton_drop3_model_summary.csv",
        "model_files": {
            "xgb": "newton_drop3_xgb_model.joblib",
        },
        "scope_note": "This model was trained on Newtonian-system data and is recommended for interpolation or near-neighbor prediction within the training range.",
    },
    "nonnewton": {
        "label": "Non-Newtonian System",
        "file": BASE_DIR / "nonnewton_raw.xlsx",
        "columns": ["SiO2", "Al2O3", "CaO", "Fe2O3", "MgO", "K2O", "Na2O", "Si_AI", "shearrate", "T", "viscosity"],
        "feature_cols": ["SiO2", "Al2O3", "CaO", "Fe2O3", "MgO", "K2O", "Na2O", "Si_AI", "shearrate", "T"],
        "fixed_model": "bp",
        "model_dir": BASE_DIR / "trained_nonnewton_models",
        "summary_file": BASE_DIR / "trained_nonnewton_models" / "nonnewton_model_summary.csv",
        "model_files": {
            "bp": "nonnewton_bp_model.joblib",
        },
        "scope_note": "This model was trained on non-Newtonian-system data.",
    },
}

MODEL_LABELS = {"bp": "BP", "xgb": "XGBoost"}

DISPLAY_LABELS = {
    "SiO2": "SiO2",
    "Al2O3": "Al2O3",
    "Fe2O3": "Fe2O3",
    "CaO": "CaO",
    "MgO": "MgO",
    "K2O": "K2O",
    "Na2O": "Na2O",
    "Si_AI": "Si/Al",
    "T": "Temperature T",
    "shearrate": "Shear Rate (used only for non-Newtonian prediction)",
}


@st.cache_data(show_spinner=False)
def load_dataset(dataset_name: str):
    cfg = DATASETS[dataset_name]
    expected_cols = cfg["columns"]

    # First try reading with the first row treated as header.
    df = pd.read_excel(cfg["file"])
    df = df.rename(columns={"Si/AI": "Si_AI", "V": "viscosity"})
    if set(expected_cols).issubset(set(df.columns)):
        return df[expected_cols].copy()

    # Fallback for datasets stored without header rows.
    df = pd.read_excel(cfg["file"], header=None)
    df.columns = expected_cols
    return df


@st.cache_resource(show_spinner=True)
def get_saved_model(dataset_name: str, model_name: str):
    cfg = DATASETS[dataset_name]
    model_path = cfg["model_dir"] / cfg["model_files"][model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


@st.cache_resource(show_spinner=True)
def get_critical_temperature_model():
    model_path = CRITICAL_MODEL_DIR / "critical_temperature_best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Critical temperature model file not found: {model_path}")
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def get_feature_ranges(dataset_name: str):
    df = load_dataset(dataset_name)
    cfg = DATASETS[dataset_name]
    stats = {}
    for col in cfg["feature_cols"]:
        series = df[col].astype(float)
        stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
        }
    return stats


def format_metrics_table(df):
    df = df.drop(columns=["split_seed", "candidate_id"], errors="ignore")
    df = df.rename(columns={"model": "Model", "best_model": "Model"})

    preferred_cols = ["Model", "R2", "RMSE", "MAE", "MAPE"]
    visible_cols = [col for col in preferred_cols if col in df.columns]

    return df[visible_cols]


def get_metrics_table(dataset_name: str):
    path = DATASETS[dataset_name]["summary_file"]
    if path.exists():
        df = pd.read_csv(path)
        fixed_label = MODEL_LABELS[DATASETS[dataset_name]["fixed_model"]]
        if "model" in df.columns:
            df = df[df["model"] == fixed_label].reset_index(drop=True)
        return format_metrics_table(df)
    return pd.DataFrame()


def get_critical_metrics_table():
    if CRITICAL_SUMMARY_FILE.exists():
        df = pd.read_csv(CRITICAL_SUMMARY_FILE)
        return format_metrics_table(df)
    return pd.DataFrame()


def format_warning(value, min_v, max_v):
    if value < min_v or value > max_v:
        return "Outside training range"
    return "Within training range"


st.set_page_config(page_title="Viscosity Prediction App", page_icon="📈", layout="wide")

st.title("Viscosity Prediction App")
st.caption(
    "This app uses pre-trained saved models to predict the critical temperature first, "
    "and then automatically selects the Newtonian or non-Newtonian viscosity model."
)

with st.sidebar:
    st.header("Run Mode")
    mode = st.radio("Prediction Mode", ["Automatic System Identification", "Manual System Selection"], index=0)

    if mode == "Manual System Selection":
        dataset_name = st.selectbox(
            "Select System",
            options=list(DATASETS.keys()),
            format_func=lambda x: DATASETS[x]["label"],
        )
    else:
        dataset_name = None

    st.info("Current app configuration: Newtonian System - XGBoost / Non-Newtonian System - BP")
    st.caption(
        "Default rule: if the actual temperature T is greater than or equal to the predicted critical temperature Tcv, "
        "the sample is classified as Newtonian; otherwise, it is classified as non-Newtonian."
    )


common_ranges = get_feature_ranges("newton")
nonnewton_ranges = get_feature_ranges("nonnewton")

st.subheader("Input Parameters")

inputs = {}
cols = st.columns(3)
common_features = ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "K2O", "Na2O", "Si_AI"]

for idx, feature in enumerate(common_features + ["T", "shearrate"]):
    if feature == "shearrate":
        meta = nonnewton_ranges[feature]
    else:
        meta = common_ranges[feature]

    label = DISPLAY_LABELS[feature]

    with cols[idx % 3]:
        inputs[feature] = st.number_input(
            label,
            min_value=float(meta["min"]),
            max_value=float(meta["max"]),
            value=float(meta["mean"]),
            step=float(max((meta["max"] - meta["min"]) / 200, 0.01)),
            format="%.6f",
        )

st.divider()

if st.button("Run Prediction", type="primary", use_container_width=True):
    critical_input = pd.DataFrame(
        [[
            inputs["SiO2"],
            inputs["Al2O3"],
            inputs["Fe2O3"],
            inputs["CaO"],
            inputs["MgO"],
            inputs["K2O"],
            inputs["Na2O"],
            inputs["Si_AI"],
        ]],
        columns=CRITICAL_FEATURE_COLS,
    )

    critical_model = get_critical_temperature_model()
    tcv_pred = float(critical_model.predict(critical_input)[0])

    current_dataset = dataset_name
    if mode == "Automatic System Identification":
        current_dataset = "newton" if inputs["T"] >= tcv_pred else "nonnewton"

    cfg = DATASETS[current_dataset]
    selected_model = cfg["fixed_model"]

    X_input = pd.DataFrame([inputs])[cfg["feature_cols"]]
    model = get_saved_model(current_dataset, selected_model)
    pred = float(model.predict(X_input)[0])

    left, right = st.columns([1.2, 1.0])

    with left:
        st.metric("Predicted Critical Temperature Tcv", f"{tcv_pred:.2f}")
        st.metric("Predicted Viscosity", f"{pred:.4f}")
        st.write(f"Selected System: `{cfg['label']}`")
        st.write(f"Model: `{MODEL_LABELS[selected_model]}`")
        st.write("Status: `Using saved models; no retraining is performed after page launch`")

    with right:
        st.markdown("**Input Range Check**")

        for feature in common_features + ["T"]:
            meta = common_ranges[feature]
            state = format_warning(inputs[feature], meta["min"], meta["max"])
            st.write(f"- `{feature}`: {state}")

        if current_dataset == "nonnewton":
            meta = nonnewton_ranges["shearrate"]
            state = format_warning(inputs["shearrate"], meta["min"], meta["max"])
            st.write(f"- `shearrate`: {state}")

st.divider()

left_ref, right_ref = st.columns(2)

with left_ref:
    st.subheader("Critical Temperature Model Performance")
    critical_metrics = get_critical_metrics_table()

    if not critical_metrics.empty:
        st.dataframe(critical_metrics, use_container_width=True, hide_index=True)
    else:
        st.info("Critical temperature model performance table was not found.")

with right_ref:
    st.subheader("Viscosity Model Performance")
    current_dataset_for_table = dataset_name if dataset_name is not None else "newton"
    metrics_df = get_metrics_table(current_dataset_for_table)

    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info("The performance table for the selected system was not found.")

with st.expander("App Notes"):
    st.write(
        "1. The app first predicts the critical temperature Tcv based on chemical composition. "
        "2. In automatic mode, if the actual temperature T is greater than or equal to Tcv, "
        "the Newtonian XGBoost model is used; otherwise, the non-Newtonian BP model is used. "
        "3. The app calls pre-trained model files saved in the current repository."
    )
