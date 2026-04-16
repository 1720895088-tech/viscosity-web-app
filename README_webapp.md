# 粘度预测网页应用

这是一个基于 Streamlit 的本地/公网可部署网页，用于根据输入的组成和工况参数预测粘度。

## 本地启动

```bash
streamlit run viscosity_web_app.py
```

## 公网部署（推荐 Streamlit Community Cloud）

将以下文件上传到 GitHub 仓库：

- `viscosity_web_app.py`
- `requirements_streamlit.txt`
- `newton_raw.xlsx`
- `nonnewton_raw.xlsx`
- `.streamlit/config.toml`

然后在 Streamlit Community Cloud 中选择 `viscosity_web_app.py` 作为主入口文件部署。

## 页面功能

- 选择牛顿/非牛顿体系
- 选择最优模型或指定模型
- 输入各成分、温度与剪切速率（非牛顿体系）
- 输出预测粘度
- 显示模型参考性能表

## 注意事项

- 非牛顿体系模型基于煤灰渣数据训练，不建议外推到未覆盖的原料体系。
- 输入参数超出训练范围时，页面结果仅供参考。
