import streamlit as st
from ultralytics import YOLO
import supervision as sv
from PIL import Image
import cv2
import numpy as np
from utils import detect_objects, set_background, set_centered_title

# 设置页面配置
st.set_page_config(
    page_title = " 道路车辆事故检测平台 ",
    page_icon = "public/icon.png", # 设置页面图标
    layout = 'wide'
)

# 设置 streamlit 应用的标题
# st.title(" 道路车辆事故检测平台 ")
set_centered_title(" 道路车辆事故检测平台 ")

# 在侧边栏添加配置选项
with st.sidebar:
    st.header(" 配置选项 ")
    # 添加置信度阈值滑块
    conf_threshold = st.slider(" 置信度阈值 ", min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.01)
    # 添加交并比阈值滑块
    iou_threshold = st.slider(" 交并比阈值 ", min_value = 0.0, max_value = 1.0, value = 0.7, step = 0.01)
    # 添加边界框粗细输入框
    line_width = st.number_input(" 边界框粗细 ", min_value = 1, max_value = 20, value = 2, step = 1)
    # 添加边界框形式选择框
    box_shape = st.selectbox(" 边界框形式 ", [" 矩形框 ", " 圆形框 ", " 四角框 "])
    # 添加文件上传器让用户上传图片
    uploaded_file = st.file_uploader(" 上传图片 ", type = ["jpg", "jpeg", "png"])

# 根据用户选择的边界框形式创建对应的边界框注释器
if box_shape == " 矩形框 ":
    box_annotator = sv.BoxAnnotator(thickness = line_width)
elif box_shape == " 圆形框 ":
    box_annotator = sv.RoundBoxAnnotator(thickness = line_width)
else:
    box_annotator = sv.BoxCornerAnnotator(thickness = line_width)

# 如果用户上传了图片
if uploaded_file is not None:
    # 打开上传的图片并转换为 numpy 数组
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # 调用 detect_objects 函数进行目标检测并将结果显示在页面上
    detected_image = detect_objects(image_np, conf_threshold, iou_threshold, box_annotator)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption = " 原始图片 ", use_column_width = True)
    with col2:
        st.image(detected_image, caption = " 检测后的图片 ", use_column_width = True)

with st.sidebar:
    # 修改背景
    st.header(" 界面背景设置 ")
    # 添加背景选项
    bg_image = st.selectbox(" 背景图片 ", ["bg", "bg1", "bg2"])
    # 添加背景图片透明度滑块
    bg_opacity = st.slider(" 背景图片透明度 ", min_value = 0.0, max_value = 1.0, value = 0.8, step = 0.01)

# 如果用户上传了背景图片
if bg_image is not None:
    set_background('public/' + bg_image + '.jpg', bg_opacity)