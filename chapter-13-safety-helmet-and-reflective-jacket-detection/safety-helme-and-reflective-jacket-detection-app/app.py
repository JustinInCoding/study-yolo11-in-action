import streamlit as st
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from utils import detect_objects, generate_detections_df, process_natural_language_query, infer_uploaded_video

# 初始化会话状态变量
if "history" not in st.session_state:
    st.session_state.history = []
if "current_index" not in st.session_state:
    st.session_state.current_index = -1

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

# 设置页面配置
st.set_page_config(
    page_title = ' 安全头盔和反光背心检测平台 ',
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)

# 设置页面标题
st.markdown(
    '<h1 style = "text-align: center; color: blue;"> 安全头盔和反光背心检测平台 </h1>',
    unsafe_allow_html = True
)

# 加载预训练模型
model = YOLO('kaggle/working/runs/detect/train/weights/best.pt')

# 在侧边栏添加配置选项
with st.sidebar:
    st.header(" 配置选项 ")

    # 加载置信度阈值滑块
    conf_threshold = st.slider(
        ' 置信度阈值 ',
        min_value = 0.0,
        max_value = 1.0,
        value = 0.5,
        step = 0.01
    )

    # 添加边框线粗细输入框
    line_width = st.number_input(
        ' 边框线粗细 ',
        min_value = 1,
        max_value = 20,
        value = 2,
        step = 1
    )

    # 添加边界框形式选择框
    box_shape = st.selectbox(
        ' 边界框形式 ',
        [" 矩形框 ", " 圆形框 ", " 四角框 "]
    )

    # 添加标签文字大小
    label_font_size = st.slider(
        " 标签文字大小 ",
        min_value = 0.1,
        max_value = 1.0,
        value = 0.4,
        step = 0.1
    )

    # 添加类别名称选择框
    selected_class_name = st.radio(
        " 选择类别名称 ",
        [" 所有类别 ", " 安全头盔 ", " 反光背心 " ]
    )

    # 用户可以选择输入类型：图片或视频
    input_type = st.radio(
        " 选择输入类型 ",
        [" 图片 ", " 视频 "]
    )

# 主界面布局
# 如果上传了文件并且输入类型是“ 图片 ”
if input_type == " 图片 ":
    # 上传图片文件
    uploaded_file = st.sidebar.file_uploader(
        " 上传图片 ",
        type = ["jpg", "jpeg", "png"]
    )

    # 如果用户上传了图片
    if uploaded_file is not None:
        # 打开上传的图片并转换 numpy 数组
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # 创建三列布局，用于显示原始图片，控制按钮和检测结果图片
        col1, col2, col3 = st.columns([3, 1, 3])
        # 在第一列显示原始图片
        with col1:
            st.image(image, caption = " 原始图片 ", use_container_width = True)
        # 在第二列放置检测按钮和导航按钮
        with col2:
            if st.button(" 检测 "):
                # 生成检测结果表格
                detections_df = generate_detections_df(uploaded_file.name, model, image_np)
                st.session_state.df = pd.concat([st.session_state.df, detections_df], ignore_index = True)

                # 调用 detect_objects 函数进行目标检测并将结果显示在页面上
                detected_image = detect_objects(image_np, conf_threshold, box_shape, selected_class_name, line_width, label_font_size)

                # 将检测结果保存到历史记录中，并更新当前索引到最新检测结果
                st.session_state.history.append(detected_image)
                st.session_state.current_index = len(st.session_state.history) - 1

            # 提供导航按钮以浏览检测历史记录
            if st.button(" 上一个 ") and st.session_state.current_index > 0:
                st.session_state.current_index -= 1

            if st.button(" 下一个 ") and st.session_state.current_index < len(st.session_state.history) - 1:
                st.session_state.current_index += 1

        # 在第三列根据当前索引显示检测结果图片
        with col3:
            if st.session_state.current_index >= 0 and st.session_state.current_index < len(st.session_state.history):
                st.image(st.session_state.history[st.session_state.current_index], caption = f" 检测结果 {st.session_state.current_index + 1} / {len(st.session_state.history)} ", use_container_width = True)

        # 创建两个等宽的容器
        col21, col22 = st.columns([1, 1])
        # 在第一个容器中显示数据框
        with col21:
            st.dataframe(st.session_state.df, use_container_width = True)
        # 在 col22 列中执行自然语言查询处理
        with col22:
            process_natural_language_query(st.session_state.df)

elif input_type == " 视频 ":
    st.title(" 上传视频 ")
    infer_uploaded_video(conf_threshold, box_shape, selected_class_name, line_width, label_font_size)