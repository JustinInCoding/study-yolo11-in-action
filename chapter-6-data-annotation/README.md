# Chapter 06 数据标注

## LabelIMG 的安装

```bash

pip install pyqt5 lxml labelImg
labelImg

# 有时候打开随便点击一下会崩
# Traceback (most recent call last):
#   File "<lib_path>/lib/python3.11/site-packages/labelImg/labelImg.py", line 965, in scroll_request
#     bar.setValue(bar.value() + bar.singleStep() * units)
# TypeError: setValue(self, a0: int): argument 1 has unexpected type 'float'
# [1]    22940 abort      labelImg

# 找到 labelImg.py 修改 
# 965 行
# bar.setValue(bar.value() + bar.singleStep() * units)
# 改为
# bar.setValue(int(bar.value() + bar.singleStep() * units))

# 导入图片后，点击 Create RectBox 会崩
# Traceback (most recent call last):
#   File "<lib_path>/lib/python3.11/site-packages/libs/canvas.py", line 530, in paintEvent
#     p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())
# TypeError: arguments did not match any overloaded call:
#   drawLine(self, l: QLineF): argument 1 has unexpected type 'float'
#   drawLine(self, line: QLine): argument 1 has unexpected type 'float'
#   drawLine(self, x1: int, y1: int, x2: int, y2: int): argument 1 has unexpected type 'float'
#   drawLine(self, p1: QPoint, p2: QPoint): argument 1 has unexpected type 'float'
#   drawLine(self, p1: Union[QPointF, QPoint], p2: Union[QPointF, QPoint]): argument 1 has unexpected type 'float'
# [1]    18658 abort      labelImg

# 找到该 canvas.py 修改
# 526 行
# p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)
# 改为
# p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))

# 530 行
# p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())
# 改为
# p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), int(self.pixmap.height()))

# 531 行
# p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())
# 改为
# p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()), int(self.prev_point.y()))

```

## Ybat 使用

下载：https://github.com/drainingsun/ybat

## bee-detect

Kaggle 上训练

```python

import os
import random
import pandas as pd
import seaborn as sns
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 初始化一个 YOLO11 模型
model = YOLO('yolo11n.pt') # 使用预训练的 yolo11n.pt 模型
# 训练模型
results = model.train(
    data = '/kaggle/input/datasets/lara311/bee-detection-dataset/data.yaml', # 数据集配置文件路径
    epochs = 30, # 训练轮数
    imgsz = 640, # 输入图片大小
    batch = 16, # 批处理大小
    device = 'cuda', # 设备选择，可以是 'cpu' 或 'cuda'
    save = True, # 保存训练结果
    plots = True # 保存训练和验证指标图以及预测实例
)

```

Kaggle 上训练完的 output: https://www.kaggle.com/code/justinwshj/bee-detect/output?scriptVersionId=368747556

数据集下载：https://www.kaggle.com/datasets/lara311/bee-detection-dataset
