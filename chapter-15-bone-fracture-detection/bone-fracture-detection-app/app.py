import streamlit as st
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from utils import detect_objects, infer_uploaded_video

# 设置页面配置
st.set_page_config(
    page_title = " 骨折检测平台 ",
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)

# 设置页面标题
st.markdown(
    '<h1 style = "text-align: center; coloe: blue;">骨折检测平台</h1>',
    unsafe_allow_html = True
)

# 在侧边栏添加配置选项
with st.sidebar:
    st.header(" 配置选项 ")
    # 添加置信度阈值滑块
    conf_threshold = st.slider(
        ' 置信度阈值 ', min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.01
    )
    # 添加边框线粗细输入框
    line_width = st.number_input(
        ' 边框线粗细 ', min_value = 1, max_value = 20, value = 2, step = 1
    )
    # 添加边界框形式选择框
    box_shape = st.selectbox(
        ' 边界框形式 ', [' 矩形框 ', ' 圆角框 ', ' 四角框 ']
    )
    # 添加标签文字大小
    label_font_size = st.slider(
        ' 标签文字大小 ', min_value = 0.1, max_value = 1.5, value = 0.8, step = 0.1
    )
    # 用户可以选择输入类型：图片或视频
    input_type = st.radio(
        ' 选择输入类型 ', [' 图片 ', ' 视频 ']
    )

if 'history_image' not in st.session_state:
    st.session_state.history_image = []
if 'history_results' not in st.session_state:
    st.session_state.history_results = []
if 'history_frame' not in st.session_state:
    st.session_state.history_frame = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = -1

# 主界面布局
# 如果上传了文件并且输入类型是“图片”
if input_type == " 图片 ":
    # 上传图片文件
    uploaded_file = st.sidebar.file_uploader(' 上传图片 ', type = ["jpg", "jpeg", "png"])

    # 如果用户上传了图片
    if uploaded_file is not None:
        # 打开上传的图片并转换为 numpy 数组
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        # 侧边栏显示图片
        st.sidebar.image(image, caption = " 原始图片 ", use_container_width = True)
        # 调用 detect_objects 函数进行目标检测
        if st.sidebar.button(" 检测 "):
            results, frame = detect_objects(image_np, conf_threshold, box_shape, line_width, label_font_size)
            # 将原始图片和检测结果保存到历史记录中，并更新当前索引到最新检测结果
            st.session_state.history_image.append(image)
            st.session_state.history_results.append(results)
            st.session_state.history_frame.append(frame)
            st.session_state.current_index = len(st.session_state.history_frame) - 1

        # 提供导航按钮以浏览检测历史记录
        if st.sidebar.button(' 上一个 ') and st.session_state.current_index > 0:
            st.session_state.current_index -= 1

        if st.sidebar.button(' 下一个 ') and st.session_state.current_index < len(st.session_state.history_frame) - 1:
            st.session_state.current_index += 1

        # 创建三列布局，用于显示原始图片，检测结果图片和检测概要
        col1, col2, col3 = st.columns([2, 2, 1])
        # 在第一列显示原始图片
        with col1:
            if st.session_state.current_index >= 0 and st.session_state.current_index < len(st.session_state.history_image):
                st.image(st.session_state.history_image[st.session_state.current_index], caption = '原始图片', use_container_width = True)

        # 在第二列根据当前索引显示检测结果图片
        with col2:
            if st.session_state.current_index >= 0 and st.session_state.current_index < len(st.session_state.history_frame):
                st.image(st.session_state.history_frame[st.session_state.current_index], caption = f'检测结果 {st.session_state.current_index + 1} / {len(st.session_state.history_frame)} ', use_container_width = True)

        # 在第三列显示指标卡
        with col3:
            if st.session_state.current_index >= 0 and st.session_state.current_index < len(st.session_state.history_results):
                # 获取当前索引所对应的检测结果
                result = st.session_state.history_results[st.session_state.current_index][0]
                # 计算检测总目标数
                total_targets = len(result.boxes)
                # 计算骨折总面积占比（假设每个检测框都代表一处骨折）
                areas = [box.xywh[0][2] * box.xywh[0][3] for box in result.boxes]
                total_area = sum(areas)
                image = st.session_state.history_image[st.session_state.current_index]
                image_np = np.array(image) # 将 PIL Image 对象转换为 Numpy 数组
                image_height, image_width = image_np.shape[:2]
                total_image_area = image_width * image_height
                area_percentage = (total_area / total_image_area) * 100
                # 获取目标类别信息
                class_names = result.names
                detected_classes = set([class_names[box.cls[0].item()] for box in result.boxes])
                class_list = ', '.join(detected_classes)
                # 获取置信度信息
                confidences = [box.conf[0].item() for box in result.boxes]
                average_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                # 显示指标卡
                st.metric(label = ' 总目标数 ', value = total_targets)
                st.metric(label = ' 检测类别 ', value = class_list)
                st.metric(label = ' 平均置信度 ', value = f"{average_confidence:.2f}")
                st.metric(label = ' 骨折面积 ', value = f"{total_area:.0f}") 
                st.metric(label = ' 骨折面积比例 (%) ', value = f"{area_percentage:.2f}") 
elif input_type == ' 视频 ':
    infer_uploaded_video(conf_threshold, box_shape, line_width, label_font_size)

