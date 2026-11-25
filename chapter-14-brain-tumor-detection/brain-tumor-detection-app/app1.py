import streamlit as st
from PIL import Image
import numpy as np
import time
import base64
from pathlib import Path
from utils import detect_objects

# 以本文件所在位置为基准，避免受 streamlit 启动目录影响
BASE_DIR = Path(__file__).resolve().parent

# 设置页面为宽屏模式，以提供更大的工作区域
st.set_page_config(
    page_title = " 脑肿瘤检测平台 ",
    layout = 'wide'
)

# 读取图片文件并编码为 Base64，构造背景图片的 data URL
bg_img_str = base64.b64encode((BASE_DIR.parent / "public" / "bg.jpg").read_bytes()).decode()

# 加载自定义 CSS 样式，显示指定编码为 UTF-8
with open(BASE_DIR / "style.css", encoding = "utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html = True
    )

# 注入背景图：同时作用于 .stApp 和内层容器，避免被新版 Streamlit 的主题背景色遮盖
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bg_img_str}");
    }}
    </style>
    """,
    unsafe_allow_html = True
)

# 添加标题，居中显示，以增强页面的视觉效果和主题明确性
st.markdown(
    "<h1 style = 'text-align: center; font-weight: bold'>基于 YOLO 的脑肿瘤检测平台 </h1>",
    unsafe_allow_html = True
)

# 新增侧边栏，用于配置检测参数和上传文件
with st.sidebar:
    # 置信度阈值滑块，用户可调节以影响检测结果
    confidence_threshold = st.slider(" 置信度阈值 ", min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.01)
    # 模型类型选择框，允许用户根据图片类型选择合适的模型
    model_type = st.selectbox(" 模型选择 ", options = [" 轴向 ", " 冠状位 ", " 失状位 "])
    # 文件上传器，用户可上传待检测的图片
    uploaded_file = st.sidebar.file_uploader(" 上传图片 ", type = ["jpg", "jpeg", "png"])
    # 用户单击 “ 执行检测 ” 按钮后开始进行目标检测
    execute_button = st.button(" 执行检测 ")

# 如果用户上传了图片并且点击了 “执行检测” 按钮
if uploaded_file is not None:
    # 打开上传图片并将其转换为 Numpy 数组
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # 创建三列布局，用于显示原始图片，检测结果和检测概览
    col1, col2, col3 = st.columns([3, 3, 2])
    # 在第一列显示原始图片
    with col1:
        st.image(image, use_column_width = True)
        st.markdown(
            "<p style = 'text-align: center'> 原始图片 </p>",
            unsafe_allow_html = True
        )

    if execute_button:
        # 在第二列根据当前索引显示检测结果
        with col2:
            # 调用 detect_objects 函数进行目标检测并将结果显示在页面上
            start_time = time.time()
            result, frame = detect_objects(image, model_type, confidence_threshold)
            end_time = time.time()
            detection_time = end_time - start_time
            st.image(frame, use_column_width = True)
            st.markdown(
                "<p style = 'text-align: center'> 检测结果 </p>",
                unsafe_allow_html = True
            )
        with col3:
            # 创建容器，用于显示指标卡
            with st.container():
                # 计算检测总目标数
                total_targets = len(result[0].boxes)
                # 计算脑肿瘤总面积占比（假设每个检测框都代表一个脑肿瘤）
                areas = [box.xywh[0][2] * box.xywh[0][3] for box in result[0].boxes]
                total_area = sum(areas)
                image_height, image_width = image_np.shape
                total_image_area = image_height * image_width
                area_percentage = (total_area / total_image_area) * 100
                # 显示指标卡
                # st.metric(label = " 总目标数 ", value = total_targets)

                st.markdown(
                    f"""
                    <div class="custom-metric">
                        <div class="metric-label">总目标数</div>
                        <div class="metric-value">{total_targets}</div>
                    </div>
                    """,
                    unsafe_allow_html = True
                )

                # st.metric(label = " 脑肿瘤面积 ", value = f"{total_area:.0f}")

                st.markdown(
                    f"""
                    <div class="custom-metric">
                        <div class="metric-label">脑肿瘤面积</div>
                        <div class="metric-value">{total_area:.0f}</div>
                    </div>
                    """,
                    unsafe_allow_html = True
                )

                # st.metric(label = " 总面积 ", value = f"{total_image_area:.2f}")

                st.markdown(
                    f"""
                    <div class="custom-metric">
                        <div class="metric-label">总面积</div>
                        <div class="metric-value">{total_image_area:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html = True
                )

                # st.metric(label = " 脑肿瘤面积比例 ", value = f"{area_percentage:.2f}")

                st.markdown(
                    f"""
                    <div class="custom-metric">
                        <div class="metric-label">脑肿瘤面积比例</div>
                        <div class="metric-value">{area_percentage:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html = True
                )

                # st.metric(label = " 消耗时间 (s) ", value = f"{detection_time:.2f}")

                st.markdown(
                    f"""
                    <div class="custom-metric">
                        <div class="metric-label">消耗时间 (s)</div>
                        <div class="metric-value">{detection_time:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html = True
                )
            st.markdown(
                "<p style = 'text-align: center'> 检测概览 </p>",
                unsafe_allow_html = True
            )