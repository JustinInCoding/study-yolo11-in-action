# study-yolo11-in-action
动手学 YOLO 

官方配套代码：https://gitee.com/jiabiao1602/yolo11_object_detection

## environment config

⚠️ pandasai 要求 python<=3.11

```bash

# 首次配置环境
conda create -n study-yolo11-in-action python=3.11

# 如果没有安装 yolo
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple/
# 用 pip show ultralytics 或者 yolo 来验证
pip show ultralytics
yolo
# 用 yolo checks 查看安装信息
yolo checks

pip install "numpy<2"
pip install jupyterlab
pip install ipywidgets
# 安装 jupyter 内核
conda install ipykernel
# 为当前环境注册内核
python -m ipykernel install --user --name study-yolo11-in-action --display-name "Python (study-yolo11-in-action)"
# 确认是否添加成功 (看看输出列表中是否含有刚注册的 study-yolo11-in-action )
jupyter kernelspec list
# 运行打开
jupyter lab

# 安装 Streamlit
pip install streamlit -i https://pypi.tuna.tsinghua.edu.cn/simple
# 查看已安装的 streamlit 的版本
streamlit --version
# 或者通过 streamlit hello 来启动一个本地服务查看官方的示例交互应用

# 安装 PandasAI
pip install pandasai==2.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
# 查看版本号
pip show pandasai
# 在 Jupyter Notebook 中，首先需要导入 PandasAI 以及 Pandas 库
# from pandasai import SmartDataframe, Agent
# import pandas as pd
# 更新 pandas 库
pip install --upgrade pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
# 如果缺少 seaborn
pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 Supervision 
pip install "supervision[desktop]" -i https://pypi.tuna.tsinghua.edu.cn/simple/
# 如果不需要 GUI
pip install supervision -i https://pypi.tuna.tsinghua.edu.cn/simple/
# 验证是否安装成功
python -c "import supervision as sv;  print(sv.__version__)"
# 查看版本号
pip show supervision

pip install trackers

```

## environment 2 config (pandas ai)

```bash

# 首次配置环境
conda create -n study-yolo11-in-action-2 python=3.11

# 如果没有安装 yolo
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple/
# 用 pip show ultralytics 或者 yolo 来验证
pip show ultralytics
yolo
# 用 yolo checks 查看安装信息
yolo checks

# pip install jupyterlab
# pip install ipywidgets
# # 安装 jupyter 内核
# conda install ipykernel
# # 为当前环境注册内核
# python -m ipykernel install --user --name study-yolo11-in-action --display-name "Python (study-yolo11-in-action)"
# # 确认是否添加成功 (看看输出列表中是否含有刚注册的 study-yolo11-in-action )
# jupyter kernelspec list
# # 运行打开
# jupyter lab

# 安装 Streamlit
pip install streamlit -i https://pypi.tuna.tsinghua.edu.cn/simple
# 查看已安装的 streamlit 的版本
streamlit --version
# 或者通过 streamlit hello 来启动一个本地服务查看官方的示例交互应用

# 安装 PandasAI
pip install pandasai==2.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
# 查看版本号
pip show pandasai
# 如果缺少 seaborn
pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install "pyarrow<26"
pip install pyyaml

```

## troubleshooting

### pandas 与 numpy 二进制不兼容（numpy.dtype size changed）

bee-detection.ipynb 导入 pandas 时报错：

```text
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```

原因：环境中 pandas 1.5.3（针对 numpy 1.x 编译的二进制包）与后装的 numpy 2.x 不兼容。安装 supervision/ultralytics 等库时依赖解析会把 numpy 升到 2.x，覆盖了 README 上面 `pip install "numpy<2"` 的结果。

修复：升级 pandas 到与 numpy 2 兼容的版本，并补装 seaborn（2026-11-12 实测 pandas 3.1.1 / numpy 2.4.6 / seaborn 0.13.2 可正常运行）：

```bash
pip install --upgrade "pandas>=2.2" seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证所有 notebook 依赖可正常导入：

```bash
python -c "import pandas as pd, numpy as np, seaborn as sns, cv2; from ultralytics import YOLO; print(pd.__version__, np.__version__)"
```

注意：升级 pandas 后，`pandasai==2.4.1`（要求 pandas==1.5.3）会出现依赖冲突提示。如果不用 pandasai 可忽略；需要用 pandasai 时建议单独建环境。

## environment 3 版本记录（适配所有 chapter 代码）

> 环境名：`study-yolo11-in-action-3`，记录日期：2026-11-27。
> 该环境实测可运行本书所有 chapter 代码，后续配置环境时可直接按以下版本安装。

### 核心依赖版本

| 包 | 版本 | 说明 |
|---|---|---|
| python | 3.11.18 | conda 创建，python=3.11 |
| ultralytics | 8.4.218 | YOLO11 主库 |
| torch | 2.15.0 | 深度学习框架 |
| torchvision | 0.29.2 | |
| numpy | 2.4.6 | |
| opencv-python | 5.1.0.95 | |
| pandas | 2.3.3 | |
| matplotlib | 3.11.2 | |
| scipy | 1.17.1 | |
| pillow | 10.4.0 | |
| supervision | 0.32.0 | 检测结果可视化/分析 |
| streamlit | 1.67.0 | Web 应用章节 |
| pandasai | 2.3.2 | PandasAI 章节 |
| pyarrow | 26.0.1 | pandasai 依赖 |
| PyYAML | 6.0.4 | |
| duckdb | 1.5.6 | pandasai 依赖 |
| openai | 1.109.1 | pandasai 后端调用 |
| polars | 2.1.0 | pandasai 依赖 |

### 环境差异说明

该环境**未安装**：jupyterlab / ipykernel（notebook 在其他环境运行）、seaborn、trackers。如需在本环境直接跑 notebook，需补装：

```bash
pip install jupyterlab ipywidgets seaborn trackers -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install ipykernel
python -m ipykernel install --user --name study-yolo11-in-action-3 --display-name "Python (study-yolo11-in-action-3)"
```

### 复现完整环境

一键按当前版本安装（完整冻结清单）：

```bash
# 先创建环境：conda create -n study-yolo11-in-action-3 python=3.11
pip install ultralytics==8.4.218 torch==2.15.0 torchvision==0.29.2 opencv-python==5.1.0.95 \
    numpy==2.4.6 pandas==2.3.3 matplotlib==3.11.2 scipy==1.17.1 pillow==10.4.0 \
    supervision==0.32.0 streamlit==1.67.0 pandasai==2.3.2 pyarrow==26.0.1 \
    PyYAML==6.0.4 duckdb==1.5.6 openai==1.109.1 polars==2.1.0
```

<details>
<summary>完整 pip freeze 清单（2026-11-27）</summary>

```text
altair==6.3.0
annotated-types==0.8.0
anyio==4.16.0
astor==0.8.1
attrs==26.1.0
av==18.1.1
certifi==2026.11.20
charset-normalizer==3.5.3
click==8.5.1
contourpy==1.3.3
cycler==0.12.1
defusedxml==0.7.1
distro==1.9.0
duckdb==1.5.6
Faker==19.13.0
filelock==4.3.1
fonttools==4.68.0
fsspec==2026.9.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.20
itsdangerous==2.2.0
Jinja2==3.1.6
jiter==0.19.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kiwisolver==1.5.1
MarkupSafe==3.0.4
matplotlib==3.11.2
mpmath==1.3.0
narwhals==2.29.0
networkx==3.6.1
numpy==2.4.6
nvidia-ml-py==13.615.71
openai==1.109.1
opencv-python==5.1.0.95
packaging==26.3
pandas==2.3.3
pandasai==2.3.2
pillow==10.4.0
pip==26.2.2
polars==2.1.0
polars-runtime-32==2.1.0
protobuf==7.36.2
psutil==7.2.2
pyarrow==26.0.1
pydantic==2.14.4
pydantic_core==2.50.5
pydeck==0.9.3
pyDeprecate==0.14.0
pyparsing==3.3.3
python-dateutil==4.0.0
python-dotenv==1.2.4
python-multipart==0.0.34
pytz==2026.6
PyYAML==6.0.4
referencing==0.37.0
requests==2.34.2
rpds-py==2026.9.1
scipy==1.17.1
setuptools==84.2.1
sniffio==1.3.1
SQLAlchemy==2.1.9
sqlglot==25.34.1
sqlglotrs==0.3.0
starlette==1.10.0
streamlit==1.67.0
supervision==0.32.0
sympy==1.14.0
torch==2.15.0
torchvision==0.29.2
tqdm==4.70.1
typing_extensions==4.16.0
typing-inspection==0.4.5
tzdata==2026.6
ultralytics==8.4.218
ultralytics-platform==0.1.121
ultralytics-thop==2.2.5
urllib3==2.8.0
uvicorn==0.54.0
websockets==17.4
wheel==0.48.0
```

</details>


