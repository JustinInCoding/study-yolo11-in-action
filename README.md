# study-yolo11-in-action
动手学 YOLO 

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

```
