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
