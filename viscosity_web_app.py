from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

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


def format_warning(value, min_v, max_v):
    if value < min_v or value > max_v:
        return "超出训练范围"
    return "训练范围内"


st.set_page_config(page_title="粘度预测交互页面", page_icon="📈", layout="wide")

st.title("粘度预测交互页面")
st.caption("基于已经训练并保存好的最佳机器学习模型构建的交互工具，用于输入成分与工况后快速预测粘度。")

with st.sidebar:
    st.header("体系设置")
    dataset_name = st.selectbox("选择体系", options=list(DATASETS.keys()), format_func=lambda x: DATASETS[x]["label"])
    selected_model = DATASETS[dataset_name]["fixed_model"]
    st.info(f"当前调用模型：{MODEL_LABELS[selected_model]}")
    st.write(DATASETS[dataset_name]["scope_note"])


feature_ranges = get_feature_ranges(dataset_name)
cfg = DATASETS[dataset_name]

st.subheader(f"{cfg['label']} 输入参数")

inputs = {}
cols = st.columns(3)
for idx, feature in enumerate(cfg["feature_cols"]):
    meta = feature_ranges[feature]
    with cols[idx % 3]:
        inputs[feature] = st.number_input(
            feature,
            min_value=float(meta["min"]),
            max_value=float(meta["max"]),
            value=float(meta["mean"]),
            step=float(max((meta["max"] - meta["min"]) / 200, 0.01)),
            format="%.6f",
        )

st.divider()

if st.button("开始预测", type="primary", use_container_width=True):
    X_input = pd.DataFrame([inputs])[cfg["feature_cols"]]
    model = get_saved_model(dataset_name, selected_model)
    pred = float(model.predict(X_input)[0])

    left, right = st.columns([1.2, 1.0])
    with left:
        st.metric("预测粘度", f"{pred:.4f}")
        st.write(f"模型：`{MODEL_LABELS[selected_model]}`")
        st.write(f"体系：`{cfg['label']}`")
        st.write("状态：`调用已保存模型，不在页面启动后重复训练`")
    with right:
        st.markdown("**输入范围检查**")
        for feature in cfg["feature_cols"]:
            meta = feature_ranges[feature]
            state = format_warning(inputs[feature], meta["min"], meta["max"])
            st.write(f"- `{feature}`: {state}")

st.divider()

st.subheader("模型参考性能")
metrics_df = get_metrics_table(dataset_name)
if not metrics_df.empty:
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
else:
    st.info("未找到该体系的参考性能表。")

with st.expander("页面说明"):
    st.write(
        "1. 该页面调用的是当前路径下已经训练并保存的模型文件；2. 页面用于本地演示和快速估算，结果应限定在训练数据范围内理解；"
        "3. 非牛顿体系模型基于煤灰渣数据训练，不建议外推到未覆盖的原料体系。"
    )
