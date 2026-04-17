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
        "label": "牛顿体系",
        "file": BASE_DIR / "newton_raw.xlsx",
        "columns": ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "K2O", "Na2O", "Si_AI", "T", "viscosity"],
        "feature_cols": ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "K2O", "Na2O", "Si_AI", "T"],
        "fixed_model": "xgb",
        "model_dir": BASE_DIR / "trained_newton_models_drop3",
        "summary_file": BASE_DIR / "trained_newton_models_drop3" / "newton_drop3_model_summary.csv",
        "model_files": {
            "xgb": "newton_drop3_xgb_model.joblib",
        },
        "scope_note": "该模型基于牛顿体系数据训练，适用于当前训练范围内的插值或近邻预测。",
    },
    "nonnewton": {
        "label": "非牛顿体系",
        "file": BASE_DIR / "nonnewton_raw.xlsx",
        "columns": ["SiO2", "Al2O3", "CaO", "Fe2O3", "MgO", "K2O", "Na2O", "Si_AI", "shearrate", "T", "viscosity"],
        "feature_cols": ["SiO2", "Al2O3", "CaO", "Fe2O3", "MgO", "K2O", "Na2O", "Si_AI", "shearrate", "T"],
        "fixed_model": "bp",
        "model_dir": BASE_DIR / "trained_nonnewton_models",
        "summary_file": BASE_DIR / "trained_nonnewton_models" / "nonnewton_model_summary.csv",
        "model_files": {
            "bp": "nonnewton_bp_model.joblib",
        },
        "scope_note": "该模型基于非牛顿煤灰渣数据训练，结论适用范围应限定于煤体系。",
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
    "T": "T",
    "shearrate": "shearrate（仅非牛顿预测时使用）",
}


@st.cache_data(show_spinner=False)
def load_dataset(dataset_name: str):
    cfg = DATASETS[dataset_name]
    df = pd.read_excel(cfg["file"], header=None)
    df.columns = cfg["columns"]
    return df


@st.cache_resource(show_spinner=True)
def get_saved_model(dataset_name: str, model_name: str):
    cfg = DATASETS[dataset_name]
    model_path = cfg["model_dir"] / cfg["model_files"][model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    return joblib.load(model_path)


@st.cache_resource(show_spinner=True)
def get_critical_temperature_model():
    model_path = CRITICAL_MODEL_DIR / "critical_temperature_best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"未找到临界温度模型文件: {model_path}")
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


def get_metrics_table(dataset_name: str):
    path = DATASETS[dataset_name]["summary_file"]
    if path.exists():
        df = pd.read_csv(path)
        fixed_label = MODEL_LABELS[DATASETS[dataset_name]["fixed_model"]]
        if "model" in df.columns:
            return df[df["model"] == fixed_label].reset_index(drop=True)
        return df
    return pd.DataFrame()


def get_critical_metrics_table():
    if CRITICAL_SUMMARY_FILE.exists():
        return pd.read_csv(CRITICAL_SUMMARY_FILE)
    return pd.DataFrame()


def format_warning(value, min_v, max_v):
    if value < min_v or value > max_v:
        return "超出训练范围"
    return "训练范围内"


st.set_page_config(page_title="粘度预测交互页面", page_icon="📈", layout="wide")

st.title("粘度预测交互页面")
st.caption("基于已经训练并保存好的模型，先预测临界温度，再自动选择牛顿或非牛顿模型进行粘度预测。")

with st.sidebar:
    st.header("运行模式")
    mode = st.radio("选择预测方式", ["自动判别体系", "手动选择体系"], index=0)
    if mode == "手动选择体系":
        dataset_name = st.selectbox("选择体系", options=list(DATASETS.keys()), format_func=lambda x: DATASETS[x]["label"])
    else:
        dataset_name = None
    st.info("当前网页只调用：牛顿体系 XGBoost / 非牛顿体系 BP")
    st.caption("默认判别规则：当实际温度 T >= 预测临界温度 Tcv 时，归为牛顿区；否则归为非牛顿区。")


common_ranges = get_feature_ranges("newton")
nonnewton_ranges = get_feature_ranges("nonnewton")

st.subheader("输入参数")

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

if st.button("开始预测", type="primary", use_container_width=True):
    critical_input = pd.DataFrame(
        [[inputs["SiO2"], inputs["Al2O3"], inputs["Fe2O3"], inputs["CaO"], inputs["MgO"], inputs["K2O"], inputs["Na2O"], inputs["Si_AI"]]],
        columns=CRITICAL_FEATURE_COLS,
    )
    critical_model = get_critical_temperature_model()
    tcv_pred = float(critical_model.predict(critical_input)[0])

    current_dataset = dataset_name
    if mode == "自动判别体系":
        current_dataset = "newton" if inputs["T"] >= tcv_pred else "nonnewton"
    cfg = DATASETS[current_dataset]
    selected_model = cfg["fixed_model"]
    X_input = pd.DataFrame([inputs])[cfg["feature_cols"]]
    model = get_saved_model(current_dataset, selected_model)
    pred = float(model.predict(X_input)[0])

    left, right = st.columns([1.2, 1.0])
    with left:
        st.metric("预测临界温度 Tcv", f"{tcv_pred:.2f}")
        st.metric("预测粘度", f"{pred:.4f}")
        st.write(f"体系判别结果：`{cfg['label']}`")
        st.write(f"模型：`{MODEL_LABELS[selected_model]}`")
        st.write("状态：`调用已保存模型，不在页面启动后重复训练`")
    with right:
        st.markdown("**输入范围检查**")
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
    st.subheader("临界温度模型参考性能")
    critical_metrics = get_critical_metrics_table()
    if not critical_metrics.empty:
        st.dataframe(critical_metrics, use_container_width=True, hide_index=True)
    else:
        st.info("未找到临界温度模型参考性能表。")
with right_ref:
    st.subheader("粘度模型参考性能")
    current_dataset_for_table = dataset_name if dataset_name is not None else "newton"
    metrics_df = get_metrics_table(current_dataset_for_table)
    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info("未找到该体系的参考性能表。")

with st.expander("页面说明"):
    st.write(
        "1. 页面首先基于组成预测临界温度 Tcv；2. 在自动模式下，若实际温度 T >= Tcv，则调用牛顿体系 XGBoost 模型，否则调用非牛顿体系 BP 模型；"
        "3. 页面调用的是当前路径下已经训练并保存的模型文件；4. 非牛顿体系模型基于煤灰渣数据训练，不建议外推到未覆盖的原料体系。"
    )
