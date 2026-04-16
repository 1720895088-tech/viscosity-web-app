from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler



BASE_DIR = Path(__file__).resolve().parent

DATASETS = {
    "newton": {
        "label": "牛顿体系",
        "file": BASE_DIR / "newton_raw.xlsx",
        "columns": ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "K2O", "Na2O", "Si_AI", "T", "viscosity"],
        "feature_cols": ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "K2O", "Na2O", "Si_AI", "T"],
        "best_model": "xgb",
        "scope_note": "该模型基于牛顿体系数据训练，适用于当前训练范围内的插值或近邻预测。",
    },
    "nonnewton": {
        "label": "非牛顿体系",
        "file": BASE_DIR / "nonnewton_raw.xlsx",
        "columns": ["SiO2", "Al2O3", "CaO", "Fe2O3", "MgO", "K2O", "Na2O", "Si_AI", "shearrate", "T", "viscosity"],
        "feature_cols": ["SiO2", "Al2O3", "CaO", "Fe2O3", "MgO", "K2O", "Na2O", "Si_AI", "shearrate", "T"],
        "best_model": "bp",
        "scope_note": "该模型基于非牛顿煤灰渣数据训练，结论适用范围应限定于煤体系。",
    },
}

MODEL_LABELS = {"bp": "BP", "xgb": "XGBoost", "rf": "RF"}


@st.cache_data(show_spinner=False)
def load_dataset(dataset_name: str):
    cfg = DATASETS[dataset_name]
    df = pd.read_excel(cfg["file"], header=None)
    df.columns = cfg["columns"]
    return df


def build_model(dataset_name: str, model_name: str):
    log_transform = FunctionTransformer(np.log1p, np.expm1, validate=False)

    if model_name == "bp":
        bp_shape = (64, 32) if dataset_name == "newton" else (96, 48, 16)
        reg = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=bp_shape,
                        activation="relu",
                        solver="adam",
                        alpha=3e-4,
                        learning_rate="adaptive",
                        learning_rate_init=6e-4,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=120,
                        max_iter=15000,
                        random_state=2026,
                    ),
                ),
            ]
        )
        return TransformedTargetRegressor(regressor=reg, transformer=log_transform)

    if model_name == "xgb":
        from xgboost import XGBRegressor
        if dataset_name == "newton":
            reg = XGBRegressor(
                n_estimators=1200,
                learning_rate=0.02,
                max_depth=3,
                subsample=1.0,
                colsample_bytree=0.85,
                min_child_weight=1,
                gamma=0.1,
                reg_alpha=0.0,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=2026,
                n_jobs=4,
            )
        else:
            reg = XGBRegressor(
                n_estimators=800,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.85,
                colsample_bytree=0.9,
                min_child_weight=3,
                gamma=0.0,
                reg_alpha=0.0,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=2026,
                n_jobs=4,
            )
        return TransformedTargetRegressor(regressor=reg, transformer=log_transform)

    if dataset_name == "newton":
        reg = RandomForestRegressor(
            n_estimators=1200,
            min_samples_leaf=3,
            max_depth=8,
            max_features=0.8,
            bootstrap=True,
            random_state=2026,
            n_jobs=4,
        )
    else:
        reg = RandomForestRegressor(
            n_estimators=800,
            min_samples_leaf=2,
            random_state=2026,
            n_jobs=4,
        )
    return TransformedTargetRegressor(regressor=reg, transformer=log_transform)


@st.cache_resource(show_spinner=True)
def get_trained_model(dataset_name: str, model_name: str):
    df = load_dataset(dataset_name)
    cfg = DATASETS[dataset_name]
    X = df[cfg["feature_cols"]]
    y = df["viscosity"]
    model = build_model(dataset_name, model_name)
    model.fit(X, y)
    return model


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
    summary_dir = BASE_DIR / "best_grouped_run_package_raw" / "best_seed_2041" / dataset_name
    path = summary_dir / f"{dataset_name}_metrics_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def format_warning(value, min_v, max_v):
    if value < min_v or value > max_v:
        return "超出训练范围"
    return "训练范围内"


st.set_page_config(page_title="粘度预测交互页面", page_icon="📈", layout="wide")

st.title("粘度预测交互页面")
st.caption("基于当前最优机器学习模型构建的本地网页交互工具，用于输入成分与工况后快速预测粘度。")

with st.sidebar:
    st.header("模型设置")
    dataset_name = st.selectbox("选择体系", options=list(DATASETS.keys()), format_func=lambda x: DATASETS[x]["label"])
    model_options = ["best", *MODEL_LABELS.keys()]
    selected_model = st.selectbox(
        "选择模型",
        options=model_options,
        format_func=lambda x: "最优模型" if x == "best" else MODEL_LABELS[x],
    )
    if selected_model == "best":
        selected_model = DATASETS[dataset_name]["best_model"]
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
    model = get_trained_model(dataset_name, selected_model)
    pred = float(model.predict(X_input)[0])

    left, right = st.columns([1.2, 1.0])
    with left:
        st.metric("预测粘度", f"{pred:.4f}")
        st.write(f"模型：`{MODEL_LABELS[selected_model]}`")
        st.write(f"体系：`{cfg['label']}`")
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
        "1. 该页面调用的是当前路径下训练效果较好的模型配置；2. 页面用于本地演示和快速估算，结果应限定在训练数据范围内理解；"
        "3. 非牛顿体系模型基于煤灰渣数据训练，不建议外推到未覆盖的原料体系。"
    )
